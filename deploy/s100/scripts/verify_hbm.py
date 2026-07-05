#!/usr/bin/env python3
"""Compare S100 HBM outputs with prepared ONNX replay references."""

import argparse
from pathlib import Path
import sys
from typing import Iterable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.s100.runtime.hbm_backend import HBMBackend  # noqa: E402


DEFAULT_HBM_DIR = REPO_ROOT / "deploy" / "s100" / "hbm"
DEFAULT_REPLAY_DIR = REPO_ROOT / "deploy" / "s100" / "replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbm-dir", type=Path, default=DEFAULT_HBM_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--bpu-core", type=int, default=0)
    return parser.parse_args()


def error(name: str, expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape {actual.shape}, expected {expected.shape}")
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    return float(np.max(difference)), float(np.mean(difference))


def main() -> None:
    args = parse_args()
    replay_paths = sorted(args.replay_dir.glob("*.npz"))[: args.samples]
    if not replay_paths:
        raise SystemExit(f"no replay cases found in {args.replay_dir}")
    backend = HBMBackend(
        args.hbm_dir / "depth_encoder.hbm",
        args.hbm_dir / "actor_estimator.hbm",
        priority=args.priority,
        bpu_core=args.bpu_core,
    )

    failures = []
    maxima = {
        "depth_latent": 0.0,
        "yaw_correction": 0.0,
        "h_out": 0.0,
        "action": 0.0,
        "chained_action": 0.0,
    }
    for path in replay_paths:
        with np.load(path, allow_pickle=False) as case:
            depth_latent, yaw, h_out = backend.infer_depth(
                case["depth_image"],
                case["proprio"],
                case["h_in"],
            )
            action = backend.infer_actor(
                case["actor_obs"],
                case["depth_latent"],
            )
            chained_actor_obs = case["actor_obs"].copy()
            chained_actor_obs[:, 6:8] = 1.5 * yaw
            chained_action = backend.infer_actor(
                chained_actor_obs,
                depth_latent,
            )
            comparisons: Iterable[Tuple[str, np.ndarray, np.ndarray]] = (
                ("depth_latent", case["expected_depth_latent"], depth_latent),
                ("yaw_correction", case["expected_yaw_correction"], yaw),
                ("h_out", case["expected_h_out"], h_out),
                ("action", case["expected_action"], action),
                (
                    "chained_action",
                    case["expected_chained_action"],
                    chained_action,
                ),
            )
            for name, expected, actual in comparisons:
                max_abs, _ = error(name, expected, actual)
                maxima[name] = max(maxima[name], max_abs)
                if max_abs > args.atol:
                    failures.append((path.name, name, max_abs))

    for name, max_abs in maxima.items():
        print(f"{name:16s} max_abs={max_abs:.8f}")
    if failures:
        for sample_name, output_name, max_abs in failures[:20]:
            print(f"FAIL {sample_name}/{output_name}: {max_abs:.8f}")
        raise SystemExit(f"HBM verification failed: {len(failures)} comparisons")
    print(f"HBM verification passed: {len(replay_paths)} cases")


if __name__ == "__main__":
    main()
