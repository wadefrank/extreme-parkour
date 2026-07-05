# 仓库指南

## 项目结构与模块组织

本仓库包含 Extreme Parkour 腿式机器人强化学习栈。`legged_gym/` 是 Isaac Gym 环境包，包含机器人配置、地形逻辑、脚本、测试和仿真资源。核心环境代码位于 `legged_gym/legged_gym/envs/`，共享辅助工具位于 `legged_gym/legged_gym/utils/`，入口脚本位于 `legged_gym/legged_gym/scripts/`，冒烟测试位于 `legged_gym/legged_gym/tests/`。机器人 URDF、网格、执行器网络和许可证位于 `legged_gym/resources/`。`rsl_rl/` 包含 PPO 训练代码，包括算法、模块、运行器和 rollout 存储。顶层 `images/` 存放 README 媒体文件。

## 构建、测试与开发命令

使用 Python 3.8、兼容 CUDA 11.3 的 PyTorch，并单独安装 NVIDIA Isaac Gym 来配置原始环境：

```bash
bash install.sh
pip install -e rsl_rl/
pip install -e legged_gym/
```

从 `legged_gym/legged_gym/scripts/` 目录运行脚本：

```bash
python train.py --exptid xxx-xx-LABEL --device cuda:0 --task a1
python play.py --exptid xxx-xx
python save_jit.py --exptid xxx-xx
```

使用 `--resume`、`--resumeid`、`--delay` 和 `--use_camera` 进行蒸馏或恢复运行。日志和检查点会写入 `legged_gym/logs/parkour_new/` 下。

## 编码风格与命名约定

使用 Python，并采用 4 空格缩进。模块、函数和变量使用 `snake_case`；类使用 `PascalCase`。遵循现有配置命名，例如 `LeggedRobotCfg`、`<Robot>Cfg` 和 `<Robot>CfgPPO`。奖励方法必须遵循 `_reward_<name>`，这样配置的 reward scales 才能解析它们。当前没有仓库级 formatter 或 linter 配置，因此修改时应与附近文件风格保持一致，并避免无关的格式化改动。

## 测试指南

测试主要是 Isaac Gym 冒烟测试和集成验证。对于环境相关改动，请从 `legged_gym/legged_gym/scripts/` 运行一个小任务检查：

```bash
python ../tests/test_env.py --task a1
```

对于策略、观测、地形或 viewer 相关改动，还应在受影响任务上运行一次简短的 `play.py` 检查或有限训练。报告结果时注明所用 GPU、任务和关键参数。

## 提交与 Pull Request 指南

近期提交使用简短直接的消息，例如 `add depth` 或 `modified legged_robot.py`；提交信息应保持祈使语气并聚焦。Pull Request 应描述目的、受影响的机器人或任务、运行过的命令，以及任何检查点或日志影响。只有在 viewer、地形或行为改动需要视觉验证时，才附上截图或视频。

## 安全与配置提示

不要提交 Isaac Gym 二进制文件、生成的日志、检查点、WandB 凭据或机器特定路径。复制或修改机器人资源时保留对应许可证，并记录任何新的外部模型或网格来源。
