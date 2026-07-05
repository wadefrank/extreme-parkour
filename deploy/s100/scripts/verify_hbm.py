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
    parser.add_argument(
        "--action-clip",
        type=float,
        default=4.8,
        help="Runtime action clip used for deployment-gating comparisons.",
    )
    parser.add_argument(
        "--strict-raw-action",
        action="store_true",
        help="Also fail when raw, pre-clipping action error exceeds --atol.",
    )
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--bpu-core", type=int, default=0)
    args = parser.parse_args()
    if args.action_clip <= 0:
        parser.error("--action-clip must be positive")
    return args


def error(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float, Tuple[int, ...]]:
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape {actual.shape}, expected {expected.shape}")
    difference = np.abs(expected.astype(np.float32) - actual.astype(np.float32))
    index = tuple(
        int(value)
        for value in np.unravel_index(int(np.argmax(difference)), difference.shape)
    )
    return float(np.max(difference)), float(np.mean(difference)), index


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
    raw_outliers = []
    names = (
        "depth_latent",
        "yaw_correction",
        "h_out",
        "action_raw",
        "chained_action_raw",
        "action",
        "chained_action",
    )
    sample_errors = {name: [] for name in names}
    maxima = {name: 0.0 for name in names}
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
            expected_action = case["expected_action"]
            expected_chained_action = case["expected_chained_action"]
            action_clip = np.float32(args.action_clip)
            comparisons: Iterable[
                Tuple[str, np.ndarray, np.ndarray, bool]
            ] = (
                ("depth_latent", case["expected_depth_latent"], depth_latent, True),
                ("yaw_correction", case["expected_yaw_correction"], yaw, True),
                ("h_out", case["expected_h_out"], h_out, True),
                (
                    "action_raw",
                    expected_action,
                    action,
                    args.strict_raw_action,
                ),
                (
                    "chained_action_raw",
                    expected_chained_action,
                    chained_action,
                    args.strict_raw_action,
                ),
                (
                    "action",
                    np.clip(expected_action, -action_clip, action_clip),
                    np.clip(action, -action_clip, action_clip),
                    True,
                ),
                (
                    "chained_action",
                    np.clip(expected_chained_action, -action_clip, action_clip),
                    np.clip(chained_action, -action_clip, action_clip),
                    True,
                ),
            )
            for name, expected, actual, enforced in comparisons:
                max_abs, _, index = error(name, expected, actual)
                sample_errors[name].append(max_abs)
                maxima[name] = max(maxima[name], max_abs)
                if max_abs > args.atol:
                    record = (
                        path.name,
                        name,
                        max_abs,
                        index,
                        float(expected[index]),
                        float(actual[index]),
                    )
                    if enforced:
                        failures.append(record)
                    else:
                        raw_outliers.append(record)

    for name, max_abs in maxima.items():
        values = np.asarray(sample_errors[name], dtype=np.float32)
        over_atol = int(np.count_nonzero(values > args.atol))
        print(
            f"{name:16s} max_abs={max_abs:.8f} "
            f"p95={np.percentile(values, 95):.8f} "
            f"mean={np.mean(values):.8f} "
            f"over_atol={over_atol}/{len(values)}"
        )
    for (
        sample_name,
        output_name,
        max_abs,
        index,
        expected,
        actual,
    ) in raw_outliers[:20]:
        print(
            f"RAW  {sample_name}/{output_name}{index}: "
            f"max_abs={max_abs:.8f} "
            f"expected={expected:.8f} actual={actual:.8f}"
        )
    if failures:
        for (
            sample_name,
            output_name,
            max_abs,
            index,
            expected,
            actual,
        ) in failures[:20]:
            print(
                f"FAIL {sample_name}/{output_name}{index}: "
                f"max_abs={max_abs:.8f} "
                f"expected={expected:.8f} actual={actual:.8f}"
            )
        raise SystemExit(f"HBM verification failed: {len(failures)} comparisons")
    suffix = "" if args.strict_raw_action else " (raw action is diagnostic only)"
    print(f"HBM verification passed: {len(replay_paths)} cases{suffix}")


if __name__ == "__main__":
    main()
