# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""
强化学习训练脚本 - 基于Isaac Gym的四足机器人训练程序
该脚本用于训练强化学习策略，使用PPO算法在仿真环境中训练机器人运动。
支持WandB实验跟踪和日志记录，包含调试模式和无渲染模式配置。
"""

# 标准库导入
import numpy as np              # 数值计算库，用于数组和矩阵运算
import os                       # 操作系统接口，用于文件和目录操作
from datetime import datetime   # 日期时间处理
from shutil import copyfile     # 文件复制功能

# 第三方库导入
import isaacgym                                         # NVIDIA Isaac Gym仿真平台
import torch                                            # PyTorch深度学习框架
import wandb                                            # 实验跟踪和可视化工具

# 本地模块导入
from legged_gym.envs import *                           # 导入自定义环境定义
from legged_gym.utils import get_args, task_registry    # 导入参数处理和任务注册工具


def train(args):
    """
    主要训练函数，负责初始化环境、配置训练流程并启动学习过程
    
    参数:
        args: 命令行参数对象，包含训练配置参数
    """

    # headless 无头模式（headless = True时，不显示图形界面）
    # 启用无头模式（不显示图形界面，适合服务器环境运行）
    args.headless = True

    # 设置日志目录路径
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    
    # 创建日志目录（如果目录已存在则忽略错误）
    try:
        os.makedirs(log_pth)
    except:
        pass

    # 调试模式配置
    if args.debug:
        mode = "disabled"       # 禁用WandB记录
        args.rows = 10          # 设置行数
        args.cols = 8           # 设置列数
        args.num_envs = 64      # 设置并行环境数量
    else:
        mode = "online"         # 启用在线WandB记录
    
    # 如果明确指定不使用WandB，则禁用
    if args.no_wandb:
        mode = "disabled"

    # 初始化WandB实验跟踪
    wandb.init(
        project=args.proj_name,     # 项目名称
        name=args.exptid,           # 实验ID
        entity="wadefrank_2026",    # WandB实体/团队名称
        group=args.exptid[:3],      # 实验分组（取ID前3字符）
        mode=mode,                  # 运行模式（online/disabled）
        dir="../../logs"            # 日志目录
    )

     # 保存关键配置文件到WandB
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    # 创建强化学习环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 创建PPO算法运行器
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root = log_pth,     # 日志根目录
        env=env,                # 环境实例
        name=args.task,         # 任务名称
        args=args               # 训练参数
    )
    
    # 启动PPO学习过程
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,    # 最大学习迭代次数
        init_at_random_ep_len=True                                  # 在随机 episode 长度初始化
    )

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()
    train(args)
