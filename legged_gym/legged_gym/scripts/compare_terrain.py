import argparse
from copy import deepcopy
import importlib
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.terrain import Terrain


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two terrain configs and save top-down images.")
    parser.add_argument("--task-a", type=str, help="Registered task name for config A, e.g. xt_dog")
    parser.add_argument("--task-b", type=str, help="Registered task name for config B, e.g. xt_dog_stage2")
    parser.add_argument("--cfg-a", type=str, help="Import spec for config A, e.g. legged_gym.envs.xt_dog.xt_parkour_config:XTDogParkourCfg")
    parser.add_argument("--cfg-b", type=str, help="Import spec for config B, e.g. legged_gym.envs.xt_dog.xt_parkour_config_2:XTDogParkourStage2Cfg")
    parser.add_argument("--seed", type=int, default=1, help="Seed used for both terrain generations.")
    parser.add_argument("--rows", type=int, default=None, help="Optional override for terrain.num_rows")
    parser.add_argument("--cols", type=int, default=None, help="Optional override for terrain.num_cols")
    parser.add_argument("--focus-row", type=int, default=None, help="Optional row index for a cropped sub-terrain view.")
    parser.add_argument("--focus-col", type=int, default=None, help="Optional col index for a cropped sub-terrain view.")
    parser.add_argument("--out-dir", type=str, default="terrain_compare", help="Directory for output images.")
    return parser.parse_args()


def load_cfg_from_spec(spec):
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    cfg_class = getattr(module, class_name)
    return cfg_class()


def load_cfg(task_name=None, cfg_spec=None):
    if task_name:
        env_cfg, _ = task_registry.get_cfgs(task_name)
        return deepcopy(env_cfg)
    if cfg_spec:
        return load_cfg_from_spec(cfg_spec)
    raise ValueError("Must provide either task name or cfg spec.")


def build_terrain(cfg, seed, rows=None, cols=None):
    cfg = deepcopy(cfg)
    if rows is not None:
        cfg.terrain.num_rows = rows
    if cols is not None:
        cfg.terrain.num_cols = cols

    random.seed(seed)
    np.random.seed(seed)
    return Terrain(cfg.terrain, num_robots=1), cfg


def height_map_in_meters(terrain):
    return terrain.height_field_raw.astype(np.float32) * terrain.cfg.vertical_scale


def all_goals_xy(terrain):
    goals = terrain.goals.reshape(-1, terrain.goals.shape[-1])[:, :2]
    goals = goals + terrain.cfg.border_size
    return goals


def plot_terrain(ax, terrain, title):
    hs = terrain.cfg.horizontal_scale
    height_m = height_map_in_meters(terrain)
    image = ax.imshow(
        height_m.T,
        origin="lower",
        cmap="terrain",
        extent=[0, terrain.tot_rows * hs, 0, terrain.tot_cols * hs],
        aspect="auto",
    )
    goals = all_goals_xy(terrain)
    ax.scatter(goals[:, 0], goals[:, 1], s=4, c="red", alpha=0.45, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return image


def plot_focus(ax, terrain, row, col, title):
    hs = terrain.cfg.horizontal_scale
    start_x = terrain.border + row * terrain.length_per_env_pixels
    end_x = terrain.border + (row + 1) * terrain.length_per_env_pixels
    start_y = terrain.border + col * terrain.width_per_env_pixels
    end_y = terrain.border + (col + 1) * terrain.width_per_env_pixels

    crop = terrain.height_field_raw[start_x:end_x, start_y:end_y].astype(np.float32) * terrain.cfg.vertical_scale
    ax.imshow(
        crop.T,
        origin="lower",
        cmap="terrain",
        extent=[0, terrain.env_length, 0, terrain.env_width],
        aspect="auto",
    )

    local_goals = terrain.goals[row, col, :, :2]
    ax.scatter(local_goals[:, 0] - row * terrain.env_length,
               local_goals[:, 1] - col * terrain.env_width,
               s=20, c="red", alpha=0.7, linewidths=0)
    terrain_idx = int(terrain.terrain_type[row, col])
    ax.set_title(f"{title} | row={row}, col={col}, idx={terrain_idx}")
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")


def save_summary(terrain_a, terrain_b, label_a, label_b, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    image = plot_terrain(axes[0], terrain_a, label_a)
    plot_terrain(axes[1], terrain_b, label_b)
    fig.colorbar(image, ax=axes, shrink=0.8, label="height [m]")
    fig.savefig(os.path.join(out_dir, "terrain_overview.png"), dpi=180)
    plt.close(fig)


def save_focus(terrain_a, terrain_b, label_a, label_b, out_dir, row, col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    plot_focus(axes[0], terrain_a, row, col, label_a)
    plot_focus(axes[1], terrain_b, row, col, label_b)
    fig.savefig(os.path.join(out_dir, f"terrain_focus_r{row}_c{col}.png"), dpi=180)
    plt.close(fig)


def save_metadata(cfg_a, cfg_b, terrain_a, terrain_b, label_a, label_b, out_dir):
    def lines(label, cfg, terrain):
        return [
            f"[{label}]",
            f"num_rows={cfg.terrain.num_rows}, num_cols={cfg.terrain.num_cols}",
            f"y_range={cfg.terrain.y_range}",
            f"height={cfg.terrain.height}",
            f"max_init_terrain_level={cfg.terrain.max_init_terrain_level}",
            f"terrain_dict={cfg.terrain.terrain_dict}",
            f"terrain_type_ids={np.unique(terrain.terrain_type.astype(int)).tolist()}",
            "",
        ]

    with open(os.path.join(out_dir, "terrain_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines(label_a, cfg_a, terrain_a) + lines(label_b, cfg_b, terrain_b)))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_a = load_cfg(task_name=args.task_a, cfg_spec=args.cfg_a)
    cfg_b = load_cfg(task_name=args.task_b, cfg_spec=args.cfg_b)

    label_a = args.task_a or args.cfg_a
    label_b = args.task_b or args.cfg_b

    terrain_a, cfg_a = build_terrain(cfg_a, args.seed, args.rows, args.cols)
    terrain_b, cfg_b = build_terrain(cfg_b, args.seed, args.rows, args.cols)

    save_summary(terrain_a, terrain_b, label_a, label_b, args.out_dir)
    save_metadata(cfg_a, cfg_b, terrain_a, terrain_b, label_a, label_b, args.out_dir)

    if args.focus_row is not None and args.focus_col is not None:
        save_focus(terrain_a, terrain_b, label_a, label_b, args.out_dir, args.focus_row, args.focus_col)

    print(f"Saved terrain comparison to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()


# 可以直接用我刚加的脚本：legged_gym/legged_gym/scripts/compare_terrain.py

# 它会固定同一个随机种子，生成两份 terrain map，然后输出：

# - terrain_overview.png：整张地形俯视对比图
# - terrain_focus_r*_c*.png：指定某个子地形块的局部放大图
# - terrain_summary.txt：两边的 terrain_dict / y_range / height / max_init_terrain_level 等摘要

# 最直接的用法是对比你现在的一阶段和二阶段：

# cd /Users/fengxian/wade/Codes/Robotics/extreme-parkour/legged_gym
# python3 -m legged_gym.scripts.compare_terrain \
# --task-a xt_dog \
# --task-b xt_dog_stage2 \
# --seed 1 \
# --rows 6 \
# --cols 8 \
# --focus-row 2 \
# --focus-col 5 \
# --out-dir /tmp/terrain_cmp_stage1_vs_stage2

# 如果你想对比“之前的老配置”和“现在的配置”，可以直接用备份文件里的类名：

# cd /Users/fengxian/wade/Codes/Robotics/extreme-parkour/legged_gym
# python3 -m legged_gym.scripts.compare_terrain \
# --cfg-a legged_gym.envs.xt_dog.xt_parkour_config_bk_20260316:XTDogParkourCfg \
# --cfg-b legged_gym.envs.xt_dog.xt_parkour_config:XTDogParkourCfg \
# --seed 1 \
# --rows 6 \
# --cols 8 \
# --focus-row 2 \
# --focus-col 5 \
# --out-dir /tmp/terrain_cmp_old_vs_new

# 结果出来后，重点看三件事：

# - terrain_overview.png 里红点是 goals，先看障碍整体密度和 goal 分布是不是更“逼跳”。
# - terrain_focus_*.png 里看 gap 宽度、hurdle 厚度、可落脚带宽度是否明显变化。
# - terrain_summary.txt 里确认是不是确实用了你想对比的参数，而不是看错配置。

# 还有一个坑：当前 legged_gym/legged_gym/scripts/play.py 会强行覆盖 terrain 配置，所以不要拿它来做精确对比。这个脚本才是按配置原样生
# 成地形。
# 如果你愿意，我下一步可以再给你补一个“3D 网格导出/并排 viewer”版本。