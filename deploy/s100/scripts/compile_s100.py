#!/usr/bin/env python3
"""Validate inputs and compile the two ONNX models with OpenExplorer."""

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "depth": REPO_ROOT / "deploy" / "s100" / "depth_encoder_s100.yaml",
    "actor": REPO_ROOT / "deploy" / "s100" / "actor_estimator_s100.yaml",
}
ONNX_MODELS = {
    "depth": REPO_ROOT / "deploy" / "s100" / "export" / "depth_encoder.onnx",
    "actor": REPO_ROOT / "deploy" / "s100" / "export" / "actor_estimator.onnx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", "depth", "actor"),
        default="all",
    )
    parser.add_argument(
        "--fast-perf",
        action="store_true",
        help="Use hb_compile's built-in fast performance configuration; no calibration data.",
    )
    parser.add_argument(
        "--march",
        choices=("nash-e", "nash-m"),
        default="nash-e",
        help="Only used with --fast-perf. nash-e=S100, nash-m=S100P.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"required file not found: {path}")


def main() -> None:
    args = parse_args()
    executable = shutil.which("hb_compile")
    if executable is None:
        raise SystemExit(
            "hb_compile not found; run this inside the D-Robotics OpenExplorer environment"
        )
    selected: List[str]
    if args.model == "all":
        selected = ["depth", "actor"]
    else:
        selected = [args.model]

    for name in selected:
        require_file(ONNX_MODELS[name])
        if args.fast_perf:
            command = [
                executable,
                "--fast-perf",
                "--model",
                str(ONNX_MODELS[name]),
                "--march",
                args.march,
            ]
        else:
            require_file(CONFIGS[name])
            command = [executable, "-c", str(CONFIGS[name])]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    print("S100 compilation complete")


if __name__ == "__main__":
    main()
