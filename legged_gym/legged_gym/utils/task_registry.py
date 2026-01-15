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

from copy import deepcopy
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

# 导入强化学习框架相关模块
from rsl_rl.env import VecEnv               # 用于并行向量化环境
from rsl_rl.runners import OnPolicyRunner   # PPO等策略梯度算法的运行器

# 导入项目特定路径和配置
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    """
    任务注册表，用于管理、注册和创建不同的机器人强化学习环境及对应的训练配置。
    核心功能包括：
    - 注册任务（机器人环境、配置、训练参数）
    - 创建仿真环境（VecEnv）
    - 初始化强化学习算法运行器（如PPO）
    """
    def __init__(self):
        # 初始化三个字典，用于存储注册的任务信息
        self.task_classes = {}      # 任务名称 -> 环境类（如G1Robot、Anymal等）
        self.env_cfgs = {}          # 任务名称 -> 环境配置（LegoedRobotCfg实例）
        self.train_cfgs = {}        # 任务名称 -> 训练配置（LegoedRobotCfgPPO实例）
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """
        注册一个任务到注册表中。
        
        参数:
            name (str): 任务名称，例如 "g1", "anymal_c_rough"，用于唯一标识该任务。
            task_class (VecEnv): 环境类，继承自VecEnv，用于创建并行环境。
            env_cfg (LeggedRobotCfg): 环境配置对象，包含机器人参数、地形、观测空间等。
            train_cfg (LeggedRobotCfgPPO): 训练配置对象，包含PPO超参数、训练迭代次数等。
        """        
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """根据任务名称获取已注册的环境类。"""
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """
        获取指定任务的环境配置和训练配置。
        确保环境配置中的种子与训练配置一致[5](@ref)。
        """
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        
        # 复制种子值，确保环境随机种子与训练器一致
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.
            创建强化学习仿真环境
            
        步骤:
            1. 解析命令行参数（若未提供）。
            2. 检查任务是否已注册。
            3. 加载环境配置和训练配置。
            4. 用命令行参数覆盖默认配置。
            5. 设置随机种子以确保可重复性。
            6. 解析仿真参数（如物理引擎、设备类型）。
            7. 实例化环境类。    
            
        Args:
            name (string): Name of a registered env.注册的任务名称。
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.命令行参数对象，包含如--task、--headless等参数。
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.可选的环境配置，用于覆盖注册的配置。

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment 创建的环境实例及其配置。
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        # 若未提供args，则从命令行解析
        if args is None:
            args = get_args()
            
        # check if there is a registered env with that name
        # 检查任务是否已注册
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        
        # 若未提供env_cfg，则从注册表中加载默认配置
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
            
        # override cfg from args (if specified)
        # 使用命令行参数更新配置（例如覆盖num_envs、seed等）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        
        # 设置随机种子
        set_seed(env_cfg.seed)
        
        # parse sim params (convert to dict first)
        # 将环境配置中的仿真参数转换为字典，并解析为Isaac Gym所需的sim_params对象
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        
        # 实例化环境类
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, init_wandb=True, log_root="default", **kwargs) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.
            创建并配置强化学习算法运行器（如PPO），用于模型训练或推理
                
        步骤:
            1. 解析命令行参数（若未提供）。
            2. 确定训练配置（优先使用传入的train_cfg，其次根据name从注册表加载）。
            3. 使用命令行参数覆盖训练配置。
            4. 设置日志目录。
            5. 初始化OnPolicyRunner（PPO运行器）。
            6. 如需恢复训练，则加载已有模型检查点。
        
        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)已创建的环境实例。
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.任务名称，用于加载默认训练配置（若train_cfg未提供）。
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.命令行参数对象。
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.可选的训练配置，用于覆盖注册表的配置。
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 日志根目录。"default"表示使用项目logs/目录。
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm              算法运行器实例
            Dict: the corresponding config file     配置
        """
        # if no args passed get command line arguments
        # 若未提供args，则从命令行解析
        if args is None:
            args = get_args()
            
        # if config files are passed use them, otherwise load from the name
        # 确定训练配置：优先使用传入的train_cfg，否则根据name加载
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
                
        # override cfg from args (if specified)
        # 用命令行参数更新训练配置
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        
        # 设置日志目录：格式为<log_root>/<月日_时分秒>_<运行名称>
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = log_root#os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 将配置对象转换为字典，并初始化PPO运行器
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, 
                                train_cfg_dict, 
                                log_dir, 
                                init_wandb=init_wandb,
                                device=args.rl_device, **kwargs)
        
        #save resume path before creating a new log_dir
        # 处理模型恢复：若配置中resume为True，则加载指定检查点
        resume = train_cfg.runner.resume
        if args.resumeid:
            log_root = LEGGED_GYM_ROOT_DIR + f"/logs/{args.proj_name}/" + args.resumeid
            resume = True
            
        # 如果命令行指定了resumeid，覆盖日志路径并强制恢复    
        if resume:
            # load previously trained model
            print(log_root)
            print(train_cfg.runner.load_run)
            # load_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', "rough_a1", train_cfg.runner.load_run)
            # 获取模型检查点路径（如logs/g1/origin_10000/model_10000.pt）
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            
            # 加载模型权重
            runner.load(resume_path)
            
            # 可选：重置策略噪声（用于探索）
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)

        # 根据参数决定是否返回日志路径（用于外部处理）
        if "return_log_dir" in kwargs:
            return runner, train_cfg, os.path.dirname(resume_path)
        else:    
            return runner, train_cfg

# make global task registry
# 创建全局任务注册表实例，供整个项目导入和使用
task_registry = TaskRegistry()
