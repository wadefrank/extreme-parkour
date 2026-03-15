# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "Extreme Parkour with Legged Robots" — a reinforcement learning framework for training quadruped robots (A1, Go1, XTDog) to perform parkour movements using NVIDIA Isaac Gym for GPU-accelerated simulation and PPO for policy optimization.

**Paper**: https://arxiv.org/abs/2309.14341
**Website**: https://extreme-parkour.github.io

## Setup

Requires Conda (Python 3.8), CUDA 11.3, and NVIDIA Isaac Gym binaries (must be downloaded separately from NVIDIA).

```bash
# Full install from scratch
bash install.sh

# Or manually (after Isaac Gym is installed):
pip install torch==1.10.0+cu113 torchvision==0.9.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e rsl_rl/
pip install -e legged_gym/
pip install numpy<1.24 pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

## Common Commands

All scripts are run from `legged_gym/legged_gym/scripts/`:

```bash
# Train base policy (~10-15k iterations, 8-10 hours)
python train.py --exptid xxx-xx-LABEL --device cuda:0 --task a1

# Train distillation policy with vision (~5-10k iterations, resumes from base policy)
python train.py --exptid yyy-yy-LABEL --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera

# Visualize/play a trained policy
python play.py --exptid xxx-xx [--delay] [--use_camera] [--checkpoint N]

# Export model for deployment (saves JIT-traced model)
python save_jit.py --exptid xxx-xx
```

Checkpoints and logs are saved to `legged_gym/logs/parkour_new/{exptid}/`.

**Key flags:**
- `--task`: Robot name (`a1`, `go1`, `xt_dog`); defaults to `a1`
- `--resume` / `--resumeid xxx-xx`: Resume training from a checkpoint
- `--delay`: Enable action delay (for sim-to-real transfer)
- `--use_camera`: Enable depth camera (required for distillation stage)
- `--debug` / `--no_wandb`: Disable WandB logging
- `--web`: Headless web viewer in play.py

**Viewer controls (play.py):**
- ALT + Mouse drag: Move camera
- `[` / `]`: Switch between robots
- Space: Pause/unpause
- F: Toggle camera mode

## Architecture

### Top-level Structure
```
legged_gym/   - Isaac Gym simulation environments
rsl_rl/       - PPO reinforcement learning implementation
```

### legged_gym
- **`envs/`** — Environment definitions. Each robot has a config class (inheriting `LeggedRobotCfg`) and an env class (inheriting `LeggedRobot`).
  - `base/legged_robot.py` — Core environment: physics stepping, reward computation, observations
  - `base/legged_robot_config.py` — Base config dataclass with all hyperparameters
  - `a1/`, `go1/`, `xt_dog/` — Robot-specific configs and env classes
- **`utils/task_registry.py`** — Central registry; maps string names to (env_class, config_class) pairs. New robots must be registered in `envs/__init__.py`.
- **`utils/terrain/`** — Procedural terrain generation (hurdles, gaps, stairs, platforms, slopes). Supports 20+ terrain types; parkour-specific types: `parkour`, `parkour_hurdle`, `parkour_flat`, `parkour_step`, `parkour_gap`.
- **`resources/robots/`** — URDF files and meshes for all supported robots
- **`scripts/`** — Entry points: `train.py`, `play.py`, `save_jit.py`

### rsl_rl
- **`algorithms/ppo.py`** — PPO implementation
- **`modules/actor_critic.py`** — `ActorCritic` with `StateHistoryEncoder` (1D convolutions over 10-frame history); Actor/Critic MLPs: `[512, 256, 128]`
- **`modules/depth_backbone.py`** — `DepthOnlyFCBackbone58x87` (Conv2d → 32-dim latent); `RecurrentDepthBackbone` (depth encoder + GRU, hidden 512, outputs 32-dim depth latent + 2-dim yaw correction)
- **`modules/estimator.py`** — Privileged state estimator
- **`runners/on_policy_runner.py`** — Training loop: env rollout → PPO update → checkpoint
- **`storage/rollout_storage.py`** — Rollout buffer

### Observation Space

The observation vector is constructed in `compute_observations()`:

| Component | Dims | Notes |
|-----------|------|-------|
| Base angular velocity | 3 | scaled ×0.25 |
| IMU (roll, pitch) | 2 | |
| Yaw delta + next yaw delta | 2 | |
| Command velocity (linear x) | 1 | |
| Env class flags | 2 | non-parkour / parkour |
| DOF positions | 12 | reindexed, scaled |
| DOF velocities | 12 | reindexed, scaled |
| Last action | 12 | reindexed |
| Feet contact filter | 4 | |
| **n_proprio total** | **57** | deployed proprioceptive obs |
| Scan / height measurements | 132 | lidar/scandots |
| Privileged explicit (base lin vel ×3) | 9 | training only |
| Privileged latent (mass, friction, motor) | 33 | training only |
| History buffer | 10 × 57 | temporal encoding |
| **Total (with scan + priv)** | **~888** | |

Asymmetric training: privileged info is available to the critic and estimator during training but not at deployment.

### Training Pipeline
1. **Base policy**: proprioceptive-only; trains motor skills on procedural terrain curriculum (10 difficulty levels)
2. **Distillation**: adds depth camera (58×87); vision encoder distills from base policy's privileged state estimations

**Domain randomization** (applied during training): friction [0.6, 2.0], added mass [0, 3 kg], motor strength [0.8×, 1.2×], optional action delay.

### Adding a New Robot
1. Add URDF + meshes to `legged_gym/resources/robots/`
2. Create config file in `legged_gym/envs/<robot>/` inheriting from `LeggedRobotCfg` and `LeggedRobotCfgPPO`
3. Create env class (or reuse `LeggedRobot`) in `legged_gym/envs/<robot>/`
4. Register in `legged_gym/envs/__init__.py` via `task_registry.register()`

Key config parameters to tune per robot: `stiffness`, `damping` (in `control` section), `init_state.default_joint_angles`, `init_state.pos` (starting height), and reward scales. XTDog uses stiffness=80, damping=2.0, action_scale=0.15 vs A1's stiffness=40, damping=0.6, action_scale=0.25.
