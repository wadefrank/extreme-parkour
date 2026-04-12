"""
从仿真中收集深度图数据，用于 S100 BPU INT8 量化校准

用法:
    python collect_calibration.py --exptid xxx-xx --checkpoint 10000 --num_frames 200

输出:
    deploy_s100/convert/calibration_data/depth_images.npy  [N, 1, 58, 87]

注意: 需要在有 Isaac Gym 的环境中运行（GPU 机器）
      如果没有 Isaac Gym 环境，可以用 --synthetic 生成合成校准数据
"""
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../runtime"))
from config import DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP


def collect_from_sim(args):
    """从仿真环境中收集深度图"""
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../legged_gym/legged_gym/scripts"))
    # 延迟导入，避免没有 Isaac Gym 时报错
    try:
        from isaacgym import gymapi
    except ImportError:
        print("Isaac Gym not available. Use --synthetic for synthetic calibration data.")
        return None

    print("TODO: 从仿真环境中收集深度图")
    print("请在有 Isaac Gym 的机器上运行 play.py 并手动保存深度图")
    print("或使用 --synthetic 选项生成合成校准数据")
    return None


def generate_synthetic(num_frames):
    """
    生成合成校准数据
    模拟训练中深度图的统计分布:
    - 归一化后范围 [-0.5, 0.5]
    - 主要分布在近距离 (大部分值接近 0.5，即近处)
    - 远处区域约占 30% (值接近 -0.5)
    """
    rng = np.random.default_rng(42)
    frames = []

    for _ in range(num_frames):
        # 模拟深度分布: 混合近距和远距
        depth = rng.uniform(-0.5, 0.5, size=(DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)).astype(np.float32)

        # 添加空间结构: 随机矩形障碍物
        num_obstacles = rng.integers(0, 4)
        for _ in range(num_obstacles):
            h1, h2 = sorted(rng.integers(0, DEPTH_IMAGE_HEIGHT, 2))
            w1, w2 = sorted(rng.integers(0, DEPTH_IMAGE_WIDTH, 2))
            obstacle_depth = rng.uniform(-0.3, 0.3)
            depth[h1:h2, w1:w2] = obstacle_depth

        # 添加地面区域 (下半部分偏向固定深度)
        ground_start = DEPTH_IMAGE_HEIGHT // 2
        depth[ground_start:, :] += rng.uniform(-0.1, 0.1)
        depth = np.clip(depth, -0.5, 0.5)

        frames.append(depth)

    return np.stack(frames)[:, np.newaxis, :, :]  # [N, 1, H, W]


def main(args):
    output_dir = os.path.join(os.path.dirname(__file__), "../convert/calibration_data")
    os.makedirs(output_dir, exist_ok=True)

    if args.synthetic:
        print(f"Generating {args.num_frames} synthetic calibration frames...")
        data = generate_synthetic(args.num_frames)
    else:
        data = collect_from_sim(args)
        if data is None:
            print("\nFalling back to synthetic data...")
            data = generate_synthetic(args.num_frames)

    # hb_compile 要求每个样本为独立 .npy 文件, shape 与 input_shape 一致 [1,1,58,87]
    for i in range(data.shape[0]):
        sample_path = os.path.join(output_dir, f"sample_{i:04d}.npy")
        np.save(sample_path, data[i:i+1])  # [1, 1, 58, 87]
    print(f"Calibration data saved: {os.path.abspath(output_dir)}")
    print(f"  Num samples: {data.shape[0]}")
    print(f"  Per-sample shape: {data[0:1].shape}")
    print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str, default="")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true",
                        help="生成合成校准数据 (无需 Isaac Gym)")
    args = parser.parse_args()
    main(args)
