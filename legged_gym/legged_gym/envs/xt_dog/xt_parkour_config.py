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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class XTDogParkourCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m] 提高初始高度，避免脚部嵌入地面
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # hip关节（髋关节），负责腿部的横向运动
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            # thigh关节（大腿关节），负责控制大腿的前后摆动
            # 前后腿统一为0.8，消除初始俯仰偏移
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,     # [rad]

            # calf关节（小腿关节），决定膝关节的屈伸
            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # XTDog ~28kg，约为A1(12kg)的2.3倍
        # 保持较高刚度以支撑体重，阻尼接近临界阻尼
        stiffness = {'joint': 80.0}  # [N*m/rad]
        damping = {'joint': 2.0}     # [N*m*s/rad]
        # 提高action_scale以获得足够的力矩-体重比
        # 最大力矩 = 80 * 0.25 * 1.2 = 24 N·m (URDF限制40 N·m)
        # 力矩体重比 = 24/(28*9.81) = 0.087 (接近A1的0.102)
        action_scale = 0.3
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/xt_dog/urdf/xt_dog.urdf'
        name = "xt_dog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # 质量随机化按体重比例放大（A1: 3kg/12kg=25%, XTDog: 6kg/28kg=21%）
        added_mass_range = [0., 6.]
        # 推力增大以匹配更大的体重
        max_push_vel_xy = 0.4

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25  # 与自然站立高度匹配
        class scales( LeggedRobotCfg.rewards.scales ):
            # 跟踪奖励
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # 正则化惩罚（针对重型机器人调整）
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.08       # 略微放宽，重型机器人动作变化更大
            delta_torques = -1.0e-7
            torques = -0.000005       # 放宽力矩惩罚，允许更大力矩输出
            hip_pos = -0.5
            dof_error = -0.02         # 放宽关节误差惩罚，允许更大运动幅度
            feet_stumble = -1
            feet_edge = -1

class XTDogParkourCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'parkour_xt_dog'