"""D-Robotics ``hbm_runtime`` adapter for the two S100 models."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


class HBMBackend:
    """Load and invoke ``depth_encoder.hbm`` and ``actor_estimator.hbm``."""

    def __init__(
        self,
        depth_model_path: Path,
        actor_model_path: Path,
        priority: int = 5,
        bpu_core: int = 0,
    ) -> None:
        try:
            from hbm_runtime import HB_HBMRuntime
        except ImportError as exc:
            raise RuntimeError(
                "hbm_runtime is required and normally comes from the S100 BSP"
            ) from exc

        self._depth = HB_HBMRuntime(str(Path(depth_model_path)))
        self._actor = HB_HBMRuntime(str(Path(actor_model_path)))
        self._depth_name = self._single_model_name(self._depth, "depth")
        self._actor_name = self._single_model_name(self._actor, "actor")
        self._configure(self._depth, self._depth_name, priority, bpu_core)
        self._configure(self._actor, self._actor_name, priority, bpu_core)
        self._require_names(
            self._depth.input_names[self._depth_name],
            ("depth_image", "proprio", "h_in"),
            "depth inputs",
        )
        self._require_names(
            self._actor.input_names[self._actor_name],
            ("actor_obs", "depth_latent"),
            "actor inputs",
        )

    @staticmethod
    def _single_model_name(runtime: object, label: str) -> str:
        names = list(runtime.model_names)
        if len(names) != 1:
            raise ValueError(f"{label} HBM must contain exactly one model: {names}")
        return names[0]

    @staticmethod
    def _configure(
        runtime: object,
        model_name: str,
        priority: int,
        bpu_core: int,
    ) -> None:
        runtime.set_scheduling_params(
            priority={model_name: priority},
            bpu_cores={model_name: [bpu_core]},
        )

    @staticmethod
    def _require_names(
        actual: Iterable[str],
        expected: Tuple[str, ...],
        label: str,
    ) -> None:
        actual_set = set(actual)
        missing = set(expected) - actual_set
        if missing:
            raise ValueError(f"{label} missing {sorted(missing)}; found {sorted(actual_set)}")

    @staticmethod
    def _inputs(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in values.items()
        }

    @staticmethod
    def _outputs(result: object, model_name: str) -> Dict[str, np.ndarray]:
        if not isinstance(result, dict) or model_name not in result:
            raise RuntimeError(f"unexpected hbm_runtime result for {model_name}")
        outputs = result[model_name]
        if not isinstance(outputs, dict):
            raise RuntimeError(f"unexpected output container for {model_name}")
        return outputs

    def infer_depth(
        self,
        depth_image: np.ndarray,
        proprio: np.ndarray,
        h_in: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = self._depth.run(
            self._inputs(
                {
                    "depth_image": depth_image,
                    "proprio": proprio,
                    "h_in": h_in,
                }
            )
        )
        outputs = self._outputs(result, self._depth_name)
        return (
            outputs["depth_latent"],
            outputs["yaw_correction"],
            outputs["h_out"],
        )

    def infer_actor(
        self,
        actor_obs: np.ndarray,
        depth_latent: np.ndarray,
    ) -> np.ndarray:
        result = self._actor.run(
            self._inputs(
                {
                    "actor_obs": actor_obs,
                    "depth_latent": depth_latent,
                }
            )
        )
        return self._outputs(result, self._actor_name)["action"]
