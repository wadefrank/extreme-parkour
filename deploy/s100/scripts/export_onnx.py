#!/usr/bin/env python3
"""Export the distilled parkour policy to S100-friendly ONNX models."""

import argparse
from pathlib import Path

import onnx
import torch

from s100_models import (
    DEFAULT_CHECKPOINT,
    REPO_ROOT,
    build_models,
    make_sample_inputs,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "deploy" / "s100" / "export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported ONNX files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker validation after export.",
    )
    return parser.parse_args()


def check_onnx(path: Path) -> None:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def export_depth_encoder(depth_encoder: torch.nn.Module, output_path: Path, opset: int) -> None:
    inputs = make_sample_inputs()
    torch.onnx.export(
        depth_encoder,
        (inputs["depth_image"], inputs["proprio"], inputs["h_in"]),
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=["depth_image", "proprio", "h_in"],
        output_names=["depth_latent", "yaw_correction", "h_out"],
    )


def export_actor_estimator(actor_estimator: torch.nn.Module, output_path: Path, opset: int) -> None:
    inputs = make_sample_inputs()
    torch.onnx.export(
        actor_estimator,
        (inputs["actor_obs"], inputs["depth_latent"]),
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=["actor_obs", "depth_latent"],
        output_names=["action"],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    depth_encoder, actor_estimator = build_models(args.checkpoint)

    depth_path = args.output_dir / "depth_encoder.onnx"
    actor_path = args.output_dir / "actor_estimator.onnx"

    print(f"Exporting depth encoder: {depth_path}")
    export_depth_encoder(depth_encoder, depth_path, args.opset)

    print(f"Exporting actor + estimator: {actor_path}")
    export_actor_estimator(actor_estimator, actor_path, args.opset)

    if not args.no_check:
        print("Checking ONNX models")
        check_onnx(depth_path)
        check_onnx(actor_path)

    print("Export complete")
    print(f"  {depth_path}")
    print(f"  {actor_path}")


if __name__ == "__main__":
    main()
