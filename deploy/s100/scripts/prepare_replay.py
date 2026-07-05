#!/usr/bin/env python3
"""Generate board replay cases and ONNX reference outputs."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from calibration_data import TENSOR_SPECS, load_sample


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "deploy" / "s100" / "calib"
DEFAULT_ONNX_DIR = REPO_ROOT / "deploy" / "s100" / "export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "deploy" / "s100" / "replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()


def make_session(path: Path) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    sample_dir = args.calibration_dir / TENSOR_SPECS["depth_image"][0]
    sample_names = sorted(path.name for path in sample_dir.glob("*.npy"))
    if len(sample_names) < args.samples:
        raise SystemExit(
            f"requested {args.samples} samples, found {len(sample_names)} in {sample_dir}"
        )

    depth_session = make_session(args.onnx_dir / "depth_encoder.onnx")
    actor_session = make_session(args.onnx_dir / "actor_estimator.onnx")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_name in sample_names[: args.samples]:
        depth_inputs = load_sample(
            args.calibration_dir,
            ("depth_image", "proprio", "h_in"),
            sample_name,
        )
        actor_inputs = load_sample(
            args.calibration_dir,
            ("actor_obs", "depth_latent"),
            sample_name,
        )
        depth_outputs = depth_session.run(None, depth_inputs)
        actor_outputs = actor_session.run(None, actor_inputs)
        chained_actor_obs = actor_inputs["actor_obs"].copy()
        chained_actor_obs[:, 6:8] = 1.5 * depth_outputs[1]
        chained_actor_outputs = actor_session.run(
            None,
            {
                "actor_obs": chained_actor_obs,
                "depth_latent": depth_outputs[0],
            },
        )
        np.savez_compressed(
            args.output_dir / sample_name.replace(".npy", ".npz"),
            **depth_inputs,
            **actor_inputs,
            expected_depth_latent=depth_outputs[0],
            expected_yaw_correction=depth_outputs[1],
            expected_h_out=depth_outputs[2],
            expected_action=actor_outputs[0],
            expected_chained_action=chained_actor_outputs[0],
        )
    print(f"Wrote {args.samples} replay cases to {args.output_dir}")


if __name__ == "__main__":
    main()
