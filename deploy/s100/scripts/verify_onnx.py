#!/usr/bin/env python3
"""Compare PyTorch checkpoint outputs with exported ONNXRuntime outputs."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from s100_models import DEFAULT_CHECKPOINT, REPO_ROOT, build_models, make_sample_inputs

import onnxruntime as ort


DEFAULT_ONNX_DIR = REPO_ROOT / "deploy" / "s100" / "export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help="Directory containing depth_encoder.onnx and actor_estimator.onnx.",
    )
    parser.add_argument("--samples", type=int, default=8, help="Number of random samples to compare.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance.")
    return parser.parse_args()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def make_session(path: Path) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def compare_array(name: str, expected: np.ndarray, actual: np.ndarray, atol: float, rtol: float) -> bool:
    diff = np.abs(expected - actual)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ok = np.allclose(expected, actual, atol=atol, rtol=rtol)
    status = "OK" if ok else "FAIL"
    print(f"{status:4s} {name:24s} max_abs={max_abs:.8f} mean_abs={mean_abs:.8f}")
    return bool(ok)


def run_depth_onnx(
    session: ort.InferenceSession,
    depth_image: torch.Tensor,
    proprio: torch.Tensor,
    h_in: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = session.run(
        None,
        {
            "depth_image": to_numpy(depth_image),
            "proprio": to_numpy(proprio),
            "h_in": to_numpy(h_in),
        },
    )
    return outputs[0], outputs[1], outputs[2]


def run_actor_onnx(
    session: ort.InferenceSession,
    actor_obs: torch.Tensor,
    depth_latent: torch.Tensor,
) -> np.ndarray:
    outputs = session.run(
        None,
        {
            "actor_obs": to_numpy(actor_obs),
            "depth_latent": to_numpy(depth_latent),
        },
    )
    return outputs[0]


def main() -> None:
    args = parse_args()
    depth_onnx_path = args.onnx_dir / "depth_encoder.onnx"
    actor_onnx_path = args.onnx_dir / "actor_estimator.onnx"

    print(f"Loading checkpoint: {args.checkpoint}")
    depth_encoder, actor_estimator = build_models(args.checkpoint)
    depth_session = make_session(depth_onnx_path)
    actor_session = make_session(actor_onnx_path)

    comparisons = []
    with torch.no_grad():
        for idx in range(args.samples):
            inputs = make_sample_inputs(args.seed + idx)

            pt_depth_latent, pt_yaw, pt_h_out = depth_encoder(
                inputs["depth_image"],
                inputs["proprio"],
                inputs["h_in"],
            )
            ort_depth_latent, ort_yaw, ort_h_out = run_depth_onnx(
                depth_session,
                inputs["depth_image"],
                inputs["proprio"],
                inputs["h_in"],
            )

            pt_action = actor_estimator(inputs["actor_obs"], pt_depth_latent)
            ort_action = run_actor_onnx(
                actor_session,
                inputs["actor_obs"],
                torch.from_numpy(ort_depth_latent),
            )

            prefix = f"sample_{idx}"
            comparisons.extend(
                [
                    compare_array(
                        f"{prefix}/depth_latent",
                        to_numpy(pt_depth_latent),
                        ort_depth_latent,
                        args.atol,
                        args.rtol,
                    ),
                    compare_array(
                        f"{prefix}/yaw_correction",
                        to_numpy(pt_yaw),
                        ort_yaw,
                        args.atol,
                        args.rtol,
                    ),
                    compare_array(
                        f"{prefix}/h_out",
                        to_numpy(pt_h_out),
                        ort_h_out,
                        args.atol,
                        args.rtol,
                    ),
                    compare_array(
                        f"{prefix}/action",
                        to_numpy(pt_action),
                        ort_action,
                        args.atol,
                        args.rtol,
                    ),
                ]
            )

    if not all(comparisons):
        raise SystemExit("ONNX verification failed")
    print("ONNX verification passed")


if __name__ == "__main__":
    main()
