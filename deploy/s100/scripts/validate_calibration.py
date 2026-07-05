#!/usr/bin/env python3
"""Validate S100 calibration directories, shapes and aligned sample names."""

import argparse
from pathlib import Path
from typing import Dict, Set

import numpy as np

from calibration_data import TENSOR_SPECS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "deploy" / "s100" / "calib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )
    parser.add_argument("--minimum-samples", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_names: Dict[str, Set[str]] = {}
    for name, (relative_dir, expected_shape) in TENSOR_SPECS.items():
        directory = args.calibration_dir / relative_dir
        if not directory.is_dir():
            raise SystemExit(f"missing calibration directory: {directory}")
        paths = sorted(directory.glob("*.npy"))
        file_names[name] = {path.name for path in paths}
        for path in paths:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.shape != expected_shape:
                raise SystemExit(
                    f"{path}: shape {value.shape}, expected {expected_shape}"
                )
            if value.dtype != np.float32:
                raise SystemExit(f"{path}: dtype {value.dtype}, expected float32")
            if not np.all(np.isfinite(value)):
                raise SystemExit(f"{path}: contains NaN or Inf")
        print(f"{name:16s} {len(paths):4d} samples {expected_shape}")

    reference_name, reference_files = next(iter(file_names.items()))
    for name, names in file_names.items():
        if names != reference_files:
            missing = sorted(reference_files - names)[:5]
            extra = sorted(names - reference_files)[:5]
            raise SystemExit(
                f"{name} is not aligned with {reference_name}; "
                f"missing={missing}, extra={extra}"
            )
    if len(reference_files) < args.minimum_samples:
        raise SystemExit(
            f"only {len(reference_files)} aligned samples; "
            f"need at least {args.minimum_samples}"
        )
    print(f"Calibration data valid: {len(reference_files)} aligned samples")


if __name__ == "__main__":
    main()
