"""Write aligned OpenExplorer calibration samples for both S100 models."""

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


TENSOR_SPECS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "depth_image": ("depth_encoder/depth_image", (58, 87)),
    "proprio": ("depth_encoder/proprio", (53,)),
    "h_in": ("depth_encoder/h_in", (1, 512)),
    "actor_obs": ("actor_estimator/actor_obs", (753,)),
    "depth_latent": ("actor_estimator/depth_latent", (32,)),
}


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


class CalibrationRecorder:
    """Save one ``.npy`` file per input and per environment sample."""

    def __init__(
        self,
        output_dir: Path,
        max_samples: int = 300,
        warmup_updates: int = 5,
        update_stride: int = 1,
        overwrite: bool = False,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")
        if update_stride <= 0:
            raise ValueError("update_stride must be positive")
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.warmup_updates = warmup_updates
        self.update_stride = update_stride
        self.sample_count = 0
        self._update_count = 0
        for relative_dir, _ in TENSOR_SPECS.values():
            directory = self.output_dir / relative_dir
            directory.mkdir(parents=True, exist_ok=True)
            existing = list(directory.glob("*.npy"))
            if existing and not overwrite:
                raise FileExistsError(
                    f"{directory} already contains calibration data; "
                    "use overwrite=True to replace it"
                )
            if overwrite:
                for path in existing:
                    path.unlink()
        if overwrite:
            manifest_path = self.output_dir / "manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()

    @property
    def full(self) -> bool:
        return self.sample_count >= self.max_samples

    def record(
        self,
        depth_image: object,
        proprio: object,
        h_in: object,
        actor_obs: object,
        depth_latent: object,
    ) -> int:
        """Record a batched depth update and return the new sample count."""

        update_index = self._update_count
        self._update_count += 1
        if update_index < self.warmup_updates:
            return self.sample_count
        if (update_index - self.warmup_updates) % self.update_stride != 0:
            return self.sample_count

        arrays = {
            "depth_image": _as_numpy(depth_image),
            "proprio": _as_numpy(proprio),
            "h_in": _as_numpy(h_in),
            "actor_obs": _as_numpy(actor_obs),
            "depth_latent": _as_numpy(depth_latent),
        }
        batch_size = self._validate_batch(arrays)
        for batch_index in range(batch_size):
            if self.full:
                break
            file_name = f"{self.sample_count:06d}.npy"
            for name, array in arrays.items():
                relative_dir, _ = TENSOR_SPECS[name]
                np.save(
                    self.output_dir / relative_dir / file_name,
                    np.ascontiguousarray(array[batch_index], dtype=np.float32),
                    allow_pickle=False,
                )
            self.sample_count += 1
        if self.full:
            self.close()
        return self.sample_count

    def _validate_batch(self, arrays: Dict[str, np.ndarray]) -> int:
        batch_sizes = set()
        for name, array in arrays.items():
            _, sample_shape = TENSOR_SPECS[name]
            expected = ("batch",) + sample_shape
            if array.ndim != len(sample_shape) + 1 or array.shape[1:] != sample_shape:
                raise ValueError(
                    f"{name} must have shape {expected}, got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
            batch_sizes.add(array.shape[0])
        if len(batch_sizes) != 1:
            raise ValueError(f"calibration batch sizes differ: {sorted(batch_sizes)}")
        return batch_sizes.pop()

    def close(self) -> None:
        manifest = {
            "sample_count": self.sample_count,
            "max_samples": self.max_samples,
            "warmup_updates": self.warmup_updates,
            "update_stride": self.update_stride,
            "dtype": "float32",
            "inputs": {
                name: {
                    "directory": relative_dir,
                    "shape": list(sample_shape),
                }
                for name, (relative_dir, sample_shape) in TENSOR_SPECS.items()
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


def load_sample(
    calibration_dir: Path,
    tensor_names: Sequence[str],
    sample_name: str,
) -> Dict[str, np.ndarray]:
    result = {}
    for name in tensor_names:
        relative_dir, expected_shape = TENSOR_SPECS[name]
        path = Path(calibration_dir) / relative_dir / sample_name
        value = np.load(path, allow_pickle=False)
        if value.shape != expected_shape:
            raise ValueError(f"{path} has shape {value.shape}, expected {expected_shape}")
        result[name] = np.ascontiguousarray(value[None, ...], dtype=np.float32)
    return result
