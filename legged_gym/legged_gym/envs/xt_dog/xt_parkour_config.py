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
        pos = [0.0, 0.0, 0.42] # x,y,z [m] 略高于自然站立高度(~0.39m)，让机器人自然落地
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # hip关节（髋关节），负责腿部的横向运动
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            # thigh关节（大腿关节），负责控制大腿的前后摆动
            # 前后腿统一为0.8，消除初始俯仰偏移
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,     # [rad]

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
        # 保持较高刚度以支撑体重，阻尼提高以抑制振荡
        stiffness = {'joint': 90.0}  # [N*m/rad]
        damping = {'joint': 2.5}     # [N*m*s/rad] 提高阻尼，28kg需要更强阻尼抑制振荡
        # 最大力矩 = 90 * 0.25 * 1.2 = 27 N·m (URDF限制40 N·m，利用率67%)
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class depth( LeggedRobotCfg.depth ):
        position = [0.31, 0, 0.03]  # XTDog体型更大，相机位置前移（A1为0.27）

    class terrain( LeggedRobotCfg.terrain ):
        # XTDog 机身更长更宽，先降低横向随机和地面粗糙度，让策略先学会直线起跳。
        y_range = [-0.2, 0.2]
        height = [0.01, 0.04]
        max_init_terrain_level = 0
        num_rows = 8

        # 保持 12×11=132 维度不变（无需改 n_scan），但覆盖更大范围以匹配 xt_dog 体型
        measured_points_x = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        measured_points_y = [-0.9, -0.72, -0.54, -0.36, -0.18, 0., 0.18, 0.36, 0.54, 0.72, 0.9]

        # 保留少量 smooth flat 作为热身地形
        terrain_dict = {“smooth slope”: 0.,
                        “rough slope up”: 0.0,
                        “rough slope down”: 0.0,
                        “rough stairs up”: 0.,
                        “rough stairs down”: 0.,
                        “discrete”: 0.,
                        “stepping stones”: 0.0,
                        “gaps”: 0.,
                        “smooth flat”: 0.15,
                        “pit”: 0.0,
                        “wall”: 0.0,
                        “platform”: 0.,
                        “large stairs up”: 0.,
                        “large stairs down”: 0.,
                        “parkour”: 0.10,
                        “parkour_hurdle”: 0.20,
                        “parkour_flat”: 0.10,
                        “parkour_step”: 0.20,
                        “parkour_gap”: 0.25,
                        “demo”: 0.0,}
        terrain_proportions = list(terrain_dict.values())

        # 下面这些覆盖只给 XTDog 用，terrain.py 里通过 getattr 读取。
        parkour_hurdle_stone_len = [0.18, 0.58]
        parkour_hurdle_height_low = [0.10, 0.18]
        parkour_hurdle_height_high = [0.12, 0.30]
        parkour_hurdle_x_range = [1.5, 2.7]
        parkour_hurdle_half_valid_width = [0.55, 1.0]

        parkour_flat_stone_len = [0.18, 0.58]
        parkour_flat_hurdle_height_low = [0.08, 0.16]
        parkour_flat_hurdle_height_high = [0.12, 0.22]
        parkour_flat_half_valid_width = [0.60, 1.10]

        parkour_step_height = [0.10, 0.42]
        parkour_step_x_range = [0.5, 1.8]
        parkour_step_half_valid_width = [0.60, 1.10]

        parkour_gap_size = [0.18, 0.75]
        parkour_gap_x_range = [1.0, 1.8]
        parkour_gap_half_valid_width = [0.70, 1.30]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/xt_dog/urdf/xt_dog.urdf'
        name = "xt_dog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0., 1.2]       # 28kg 机器人加速慢，降低速度上限（A1: [0., 1.5]）

        class max_ranges( LeggedRobotCfg.commands.max_ranges ):
            lin_vel_x = [0.2, 0.7]      # 课程学习最大速度范围（A1: [0.3, 0.8]）

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # 质量随机化按体重比例放大（A1: 3kg/12kg=25%, XTDog: 6kg/28kg=21%）
        added_mass_range = [0., 6.]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.35
        class scales( LeggedRobotCfg.rewards.scales ):
            # === 正奖励：加大激励 ===
            tracking_goal_vel = 2.0     # 1.2→2.0，给更强的前进激励
            tracking_yaw = 0.5          # 保持

            # === 最大惩罚项：大幅缩小 ===
            collision = -2.             # -10→-2，原值-1.2量级，体型大初期碰撞不可避免
            action_rate = -0.01         # base -0.1→-0.01，原值-0.35量级，允许探索
            dof_acc = -5.0e-8           # base -2.5e-7→5e-8，原值-0.08量级

            # === 中等惩罚项：减半 ===
            ang_vel_xy = -0.02          # base -0.05→-0.02
            lin_vel_z = -0.5            # base -1.0→-0.5
            hip_pos = -0.2             # base -0.5→-0.2
            dof_error = -0.02           # base -0.04→-0.02
            orientation = -0.5          # base -1.0→-0.5

            # === 小惩罚项：缩小 ===
            torques = -0.000002         # base -0.00001
            delta_torques = -2.0e-8     # base -1.0e-7
            feet_stumble = -0.5         # base -1.0
            feet_edge = -0.5            # base -1.0

class XTDogParkourCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'parkour_xt_dog'