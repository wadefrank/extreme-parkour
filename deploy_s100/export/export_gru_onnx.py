"""
导出 RecurrentDepthBackbone 的 GRU 模块 (combination_mlp + GRU + output_mlp) 为 ONNX
GRU hidden state 作为显式输入/输出，便于设备端管理

用法:
    python export_gru_onnx.py --exptid xxx-xx --checkpoint 10000

输出:
    deploy_s100/models/gru_module.onnx
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../legged_gym/legged_gym/scripts"))

from save_jit import get_load_path

sys.path.append(os.path.join(os.path.dirname(__file__), "../runtime"))
from config import (
    N_PROPRIO, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM,
    DEPTH_LATENT_DIM, YAW_CORRECTION_DIM,
)


class GRUModule(nn.Module):
    """
    从 RecurrentDepthBackbone 中提取的 GRU 部分
    包含: combination_mlp + GRU + output_mlp
    hidden state 显式传入/传出
    """
    def __init__(self):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.combination_mlp = nn.Sequential(
            nn.Linear(CNN_OUTPUT_DIM + N_PROPRIO, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=DEPTH_ENCODER_HIDDEN_DIM, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(DEPTH_ENCODER_HIDDEN_DIM, DEPTH_LATENT_DIM + YAW_CORRECTION_DIM),
            last_activation,
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        proprioception: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple:
        """
        Args:
            cnn_features: [1, 32] CNN backbone 输出
            proprioception: [1, 53] 本体感知 (yaw 已 mask 为 0)
            hidden_state: [1, 1, 512] GRU hidden state
        Returns:
            output: [1, 34] = [depth_latent(32) + yaw_correction(2)]
            new_hidden: [1, 1, 512] 更新后的 GRU hidden state
        """
        combined = self.combination_mlp(torch.cat([cnn_features, proprioception], dim=-1))
        rnn_out, new_hidden = self.rnn(combined[:, None, :], hidden_state)
        output = self.output_mlp(rnn_out.squeeze(1))
        return output, new_hidden


def export(args):
    log_root = os.path.join(os.path.dirname(__file__), "../../legged_gym/logs/parkour_new")
    load_run = os.path.join(log_root, args.exptid)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=args.checkpoint)
    print(f"Loading checkpoint: {load_path}")

    device = torch.device("cpu")
    state_dict = torch.load(load_path, map_location=device)
    depth_encoder_sd = state_dict["depth_encoder_state_dict"]

    # 实例化 GRU 模块
    gru_module = GRUModule()

    # 从 depth_encoder_state_dict 加载权重
    gru_sd = {}
    for k, v in depth_encoder_sd.items():
        if k.startswith("combination_mlp."):
            gru_sd[k] = v
        elif k.startswith("rnn."):
            gru_sd[k] = v
        elif k.startswith("output_mlp."):
            gru_sd[k] = v
    gru_module.load_state_dict(gru_sd)
    gru_module.eval()

    # 导出 ONNX
    dummy_cnn_feat = torch.randn(1, CNN_OUTPUT_DIM)
    dummy_proprio = torch.randn(1, N_PROPRIO)
    dummy_hidden = torch.zeros(1, 1, DEPTH_ENCODER_HIDDEN_DIM)

    output_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "gru_module.onnx")

    with torch.no_grad():
        ref_output, ref_hidden = gru_module(dummy_cnn_feat, dummy_proprio, dummy_hidden)
        print(f"GRU module output shape: {ref_output.shape}")   # [1, 34]
        print(f"GRU hidden state shape: {ref_hidden.shape}")     # [1, 1, 512]

        torch.onnx.export(
            gru_module,
            (dummy_cnn_feat, dummy_proprio, dummy_hidden),
            onnx_path,
            opset_version=11,
            input_names=["cnn_features", "proprioception", "hidden_state"],
            output_names=["output", "new_hidden_state"],
            dynamic_axes=None,
        )

    print(f"ONNX model saved to: {os.path.abspath(onnx_path)}")

    # onnxruntime 验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_outputs = sess.run(None, {
            "cnn_features": dummy_cnn_feat.numpy(),
            "proprioception": dummy_proprio.numpy(),
            "hidden_state": dummy_hidden.numpy(),
        })
        diff_out = np.abs(ref_output.numpy() - ort_outputs[0]).max()
        diff_hid = np.abs(ref_hidden.numpy() - ort_outputs[1]).max()
        print(f"Max abs diff output: {diff_out:.2e}, hidden: {diff_hid:.2e}")
        if diff_out < 1e-5 and diff_hid < 1e-5:
            print("ONNX export verified successfully!")
        else:
            print("WARNING: large numerical difference")
    except ImportError:
        print("onnxruntime not installed, skipping verification")

    # 保存参考数据
    np.save(os.path.join(output_dir, "gru_ref_cnn_feat.npy"), dummy_cnn_feat.numpy())
    np.save(os.path.join(output_dir, "gru_ref_proprio.npy"), dummy_proprio.numpy())
    np.save(os.path.join(output_dir, "gru_ref_output.npy"), ref_output.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    args = parser.parse_args()
    export(args)
