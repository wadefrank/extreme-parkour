"""
端到端验证：对比 PyTorch 原始模型 vs 拆分后的 ONNX 子模型
确保拆分和导出过程不引入数值误差

用法:
    python validate_exports.py --exptid xxx-xx --checkpoint 10000

验证内容:
    1. CNN backbone: PyTorch vs ONNX
    2. GRU module: PyTorch vs ONNX (多步序列)
    3. 端到端: 原始 RecurrentDepthBackbone vs (CNN ONNX + GRU ONNX) 拼接
    4. base_jit.pt: JIT vs 原始 HardwareVisionNN
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../legged_gym/legged_gym/scripts"))

from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from save_jit import get_load_path, HardwareVisionNN

sys.path.append(os.path.join(os.path.dirname(__file__), "../runtime"))
from config import (
    N_PROPRIO, N_SCAN, N_PRIV_EXPLICIT, N_PRIV_LATENT, HISTORY_LEN,
    N_ACTIONS, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM,
    DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, N_OBS_TOTAL, DEPTH_LATENT_DIM,
)


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def validate(args):
    log_root = os.path.join(os.path.dirname(__file__), "../../legged_gym/logs/parkour_new")
    load_run = os.path.join(log_root, args.exptid)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=args.checkpoint)
    print(f"Loading checkpoint: {load_path}")

    device = torch.device("cpu")
    ckpt = torch.load(load_path, map_location=device)
    depth_encoder_sd = ckpt["depth_encoder_state_dict"]

    model_dir = os.path.join(os.path.dirname(__file__), "../models")
    cnn_onnx = os.path.join(model_dir, "cnn_backbone.onnx")
    gru_onnx = os.path.join(model_dir, "gru_module.onnx")
    jit_path = None

    # 查找 base_jit.pt
    traced_dir = os.path.join(os.path.dirname(load_path), "traced")
    if os.path.isdir(traced_dir):
        for f in os.listdir(traced_dir):
            if f.endswith("-base_jit.pt"):
                jit_path = os.path.join(traced_dir, f)
                break

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is required. pip install onnxruntime")
        return

    print("\n" + "=" * 60)
    print("1. CNN Backbone 验证")
    print("=" * 60)

    # 重建 PyTorch CNN
    backbone = DepthOnlyFCBackbone58x87(N_PROPRIO, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM)
    backbone_sd = {k[len("base_backbone."):]: v for k, v in depth_encoder_sd.items() if k.startswith("base_backbone.")}
    backbone.load_state_dict(backbone_sd)
    backbone.eval()

    # 随机测试 10 组
    cnn_sess = ort.InferenceSession(cnn_onnx)
    cnn_diffs = []
    for i in range(10):
        depth = torch.randn(1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH) * 0.5
        with torch.no_grad():
            pt_out = backbone(depth).numpy()
        ort_out = cnn_sess.run(None, {"depth_image": depth.unsqueeze(1).numpy()})[0]
        cnn_diffs.append(np.abs(pt_out - ort_out).max())

    print(f"  Max abs diff (10 samples): {max(cnn_diffs):.2e}")
    print(f"  Mean abs diff: {np.mean(cnn_diffs):.2e}")
    print(f"  Status: {'PASS' if max(cnn_diffs) < 1e-2 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("2. GRU Module 验证 (50步序列)")
    print("=" * 60)

    # 重建完整 RecurrentDepthBackbone
    full_encoder = RecurrentDepthBackbone(
        DepthOnlyFCBackbone58x87(N_PROPRIO, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM),
        env_cfg=None,
    )
    full_encoder.load_state_dict(depth_encoder_sd)
    full_encoder.eval()
    full_encoder.hidden_states = None

    gru_sess = ort.InferenceSession(gru_onnx)
    hidden_ort = np.zeros((1, 1, DEPTH_ENCODER_HIDDEN_DIM), dtype=np.float32)

    output_diffs = []
    for step in range(50):
        depth = torch.randn(1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH) * 0.5
        proprio = torch.randn(1, N_PROPRIO) * 0.3

        with torch.no_grad():
            pt_out = full_encoder(depth, proprio).numpy()

        # ONNX 拆分推理: CNN + GRU
        cnn_feat = cnn_sess.run(None, {"depth_image": depth.unsqueeze(1).numpy()})[0]
        ort_out, hidden_ort = gru_sess.run(None, {
            "cnn_features": cnn_feat,
            "proprioception": proprio.numpy(),
            "hidden_state": hidden_ort,
        })
        output_diffs.append(np.abs(pt_out - ort_out).max())

    print(f"  Max abs diff (50 steps): {max(output_diffs):.2e}")
    print(f"  Mean abs diff: {np.mean(output_diffs):.2e}")
    print(f"  Cosine sim (last step): {cosine_similarity(pt_out, ort_out):.6f}")
    print(f"  Status: {'PASS' if max(output_diffs) < 1e-4 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("3. Base Policy (JIT) 验证")
    print("=" * 60)

    if jit_path is None:
        print("  SKIP: base_jit.pt not found. Run save_jit.py first.")
    else:
        # 重建 PyTorch HardwareVisionNN
        policy_pt = HardwareVisionNN(
            N_PROPRIO, N_SCAN, N_PRIV_LATENT, N_PRIV_EXPLICIT,
            HISTORY_LEN, N_ACTIONS, tanh=False,
        )
        policy_pt.actor.load_state_dict(ckpt["depth_actor_state_dict"], strict=True)
        policy_pt.estimator.load_state_dict(ckpt["estimator_state_dict"])
        policy_pt.eval()

        policy_jit = torch.jit.load(jit_path, map_location=device)
        policy_jit.eval()

        policy_diffs = []
        for i in range(10):
            obs = torch.randn(1, N_OBS_TOTAL)
            depth_lat = torch.randn(1, DEPTH_LATENT_DIM)
            with torch.no_grad():
                pt_act = policy_pt(obs, depth_lat).numpy()
                jit_act = policy_jit(obs, depth_lat).numpy()
            policy_diffs.append(np.abs(pt_act - jit_act).max())

        print(f"  Max abs diff (10 samples): {max(policy_diffs):.2e}")
        print(f"  Status: {'PASS' if max(policy_diffs) < 1e-5 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    args = parser.parse_args()
    validate(args)
