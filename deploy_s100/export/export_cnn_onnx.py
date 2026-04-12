"""
导出 DepthOnlyFCBackbone58x87 (CNN backbone) 为 ONNX 格式
用于后续通过 hb_mapper 转换为 S100 BPU 可运行的 .bin 模型

用法:
    python export_cnn_onnx.py --exptid xxx-xx --checkpoint 10000

输出:
    deploy_s100/models/cnn_backbone.onnx
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../legged_gym/legged_gym/scripts"))

from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87
from save_jit import get_load_path

sys.path.append(os.path.join(os.path.dirname(__file__), "../runtime"))
from config import (
    N_PROPRIO, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM,
    DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH,
)


class CNNBackboneWrapper(nn.Module):
    """
    封装 DepthOnlyFCBackbone58x87，使其输入为 NCHW 格式 [1, 1, 58, 87]
    原始模型 forward 内部会做 unsqueeze(1)，这里改为直接接收 4D 输入
    """
    def __init__(self, backbone: DepthOnlyFCBackbone58x87):
        super().__init__()
        self.image_compression = backbone.image_compression
        self.output_activation = backbone.output_activation

    def forward(self, depth_image_nchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_image_nchw: [1, 1, 58, 87] float32, 归一化深度图
        Returns:
            cnn_features: [1, 32] float32
        """
        compressed = self.image_compression(depth_image_nchw)
        return self.output_activation(compressed)


def export(args):
    log_root = os.path.join(os.path.dirname(__file__), "../../legged_gym/logs/parkour_new")
    load_run = os.path.join(log_root, args.exptid)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=args.checkpoint)
    print(f"Loading checkpoint: {load_path}")

    device = torch.device("cpu")
    state_dict = torch.load(load_path, map_location=device)

    # 实例化 CNN backbone
    backbone = DepthOnlyFCBackbone58x87(
        prop_dim=N_PROPRIO,
        scandots_output_dim=CNN_OUTPUT_DIM,
        hidden_state_dim=DEPTH_ENCODER_HIDDEN_DIM,
    )

    # 从 depth_encoder_state_dict 中提取 base_backbone 权重
    depth_encoder_sd = state_dict["depth_encoder_state_dict"]
    backbone_sd = {}
    prefix = "base_backbone."
    for k, v in depth_encoder_sd.items():
        if k.startswith(prefix):
            backbone_sd[k[len(prefix):]] = v
    backbone.load_state_dict(backbone_sd)
    backbone.eval()

    # 封装为 NCHW 输入
    wrapper = CNNBackboneWrapper(backbone)
    wrapper.eval()

    # 导出 ONNX
    dummy_input = torch.randn(1, 1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)
    output_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "cnn_backbone.onnx")

    with torch.no_grad():
        # 验证前向传播
        ref_output = wrapper(dummy_input)
        print(f"CNN backbone output shape: {ref_output.shape}")  # [1, 32]

        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=["depth_image"],
            output_names=["cnn_features"],
            dynamic_axes=None,  # 固定 batch=1
        )

    print(f"ONNX model saved to: {os.path.abspath(onnx_path)}")

    # 用 onnxruntime 验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_output = sess.run(None, {"depth_image": dummy_input.numpy()})[0]
        diff = np.abs(ref_output.numpy() - ort_output).max()
        cos_sim = np.dot(ref_output.numpy().flatten(), ort_output.flatten()) / (
            np.linalg.norm(ref_output.numpy()) * np.linalg.norm(ort_output) + 1e-8)
        print(f"Max abs diff (PyTorch vs ONNX Runtime): {diff:.2e}")
        print(f"Cosine similarity: {cos_sim:.6f}")
        if diff < 1e-2 and cos_sim > 0.999:
            print("ONNX export verified successfully!")
        else:
            print("WARNING: large numerical difference, check export")
    except ImportError:
        print("onnxruntime not installed, skipping verification")

    # 保存参考输入输出用于后续 BPU 精度验证
    np.save(os.path.join(output_dir, "cnn_ref_input.npy"), dummy_input.numpy())
    np.save(os.path.join(output_dir, "cnn_ref_output.npy"), ref_output.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    args = parser.parse_args()
    export(args)
