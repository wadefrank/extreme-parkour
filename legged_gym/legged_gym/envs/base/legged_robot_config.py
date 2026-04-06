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

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn
class LeggedRobotCfg(BaseConfig):
    """四足机器人环境的基础配置类，所有机器人特定配置（A1、Go1、XTDog）都继承自此类"""
    
    class play:
        # 是否加载学生（蒸馏）策略配置
        load_student_config = False

        # 是否遮蔽特权观测
        mask_priv_obs = False
    
    class env:
        # 并行仿真环境数量（GPU 并行训练的核心参数）
        num_envs = 6144

        # 在机器人周围采样地面高度（scandots）的维度，默认为132，表示为12行11列的网格，采样点分别为：measured_points_x和measured_points_y
        n_scan = 132

        # 特权显式观测维度，9（base_lin_vel 3 + 6个零占位）
        n_priv = 3+3 +3
        
         # n_priv_latent 特权隐式观测维度 （默认为29）
         # 1.质量（4维）：机身附加质量（1）和机身质心漂移（3）
         # 2.摩擦（1维）
         # 3.电机强度（24维)：Kp（12维）+Kd（12维）
        n_priv_latent = 4 + 1 + 12 +12
        
        # n_proprio 本体感知维度（默认为53）
        # 1.机身坐标系下的角速度：base_ang_vel（3维）
        #    - wx
        #    - wy
        #    - wz
        # 2.IMU：imu_obs（2维）
        #    - roll
        #    - pitch
        # 3.偏航（3维）
        #    - delta_yaw（到当前目标点的偏航差）
        #    - delta_next_yaw：到下一目标点的偏航差
        #    - 1维零占位
        # 4.接触（4维），对应四条腿是否接触地面
        # 5.关节相关（36 维）
        #    - 12 维关节角偏差 dof_pos - default_dof_pos
        #    - 12 维关节角速度 dof_vel
        #    - 12 维上一步动作 action_history_buf[:, -1]
        # 6.指令/任务标志(5维)
        #    - 1 维前进速度指令
        #    - 1 维非跑酷地形标志
        #    - 1 维跑酷地形标志
        #    - 2 维零占位
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5
        
        # 历史帧数，用于时序编码
        history_len = 10

        # 总观测维度 = 本体感知53 + 扫描132 + 历史10×53 + 特权隐式29 + 特权显式9 = 753
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        
        # 特权观测维度，None 表示 critic 直接使用 obs_buf
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        
        # 动作维度（4条腿 × 3个关节）
        num_actions = 12
        
        # 环境间距（使用地形时不生效）
        env_spacing = 3.  # not used with heightfields/trimeshes 

        # 是否将超时信息发送给算法（用于 bootstrapping）
        send_timeouts = True # send time out information to the algorithm
        
        # 1秒内的 episode 时长
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"


        
        
        # 是否启用历史编码器（DAgger 蒸馏用）
        # 作用：控制环境里是否维护一段本体观测历史缓冲 obs_history_buf，每步把最近 history_len 帧 n_proprio 观测拼到 obs 末尾
        # 直白地说，它让策略不只看“这一帧”，还看“最近 10 帧我是怎么动的”，从而推断速度、接触变化、动力学特性等。
        history_encoding = True

        # 是否重排关节顺序（URDF顺序 → 策略顺序）
        reorder_dofs = True
        
        
        # action_delay_range = [0, 5]

        # additional visual inputs 

        # action_delay_range = [0, 5]

        # additional visual inputs 
        # 是否在观测中包含足部接触信息
        include_foot_contacts = True

        # 起始状态随机化（用于增强鲁棒性）
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        # 接触历史缓冲区长度
        contact_buf_len = 100

        # 路径点（waypoint）导航参数
        # 到达当前路径点的距离阈值（米）
        next_goal_threshold = 0.2
        # 到达路径点后的延迟时间
        reach_goal_delay = 0.1
        # 观测中包含的未来路径点数量
        num_future_goal_obs = 2

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2
    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        """
        地形系统配置
        - 地形网格: num_rows(难度级别) × num_cols(地形类型)，每块地形 terrain_length × terrain_width 米
        - curriculum: 课程学习，机器人表现好则升级到更难地形，表现差则降级
        """
        # 地形网格类型: none/plane/heightfield/trimesh
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        
        # 高度场转三角网格的方法: grid(精确) 或 fast(pydelatin 加速)
        hf2mesh_method = "grid"  # grid or fast

        # fast 方法的最大误差
        max_error = 0.1 # for fast
        max_error_camera = 2                    # 相机模式下的最大误差

        y_range = [-0.4, 0.4]                   # 用来控制部分 parkour 地形在横向 y 方向上的随机偏移范围，不是整个地形的总宽度。
        
        edge_width_thresh = 0.05                # 离台阶边多近，才算是在踩边上（单位：米）
        
        # 水平上每个像素的物理尺寸（单位：米），越小越精细，计算量越大
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        
        horizontal_scale_camera = 0.1           # 相机模式下，水平上每个像素的物理尺寸（单位：米），使用更粗的分辨率以加速
        vertical_scale = 0.005                  # 垂直方向上每个像素的物理尺寸（单位：米）
        border_size = 5                         # 边界的长度（单位：米）
        height = [0.02, 0.06]                   # 地面粗糙度的上下起伏范围（单位：米）
        simplify_grid = False                   # 是否做三角化网格减面，以提高渲染速度（相机模式开启，减到 5% 三角形）
        gap_size = [0.02, 0.1]                  # 缝隙宽度范围
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075               # 粗糙地面的“横向颗粒度”
        curriculum = True                       # 是否启用课程学习

        all_vertical = False                    # 墙是不是直接拉满成垂直
        no_flat = True                          # 低难度时要不要出现平地/无墙版本
        
        static_friction = 1.0                   # 静摩擦系数（物体“还没滑起来”时，阻止它开始滑动的能力）
        dynamic_friction = 1.0                  # 动摩擦系数（物体“已经在滑了”之后，继续阻碍滑动的能力）
        restitution = 0.                        # 弹性恢复系数（撞上地面后，会反弹多少）
        measure_heights = True                  # 是否测量地形高度（scandots 需要）
        # 扫描点网格: 12×11 = 132 个点，覆盖机体周围 1.65m × 1.5m 区域
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        # 是否选择单一地形类型
        selected = False # select a unique terrain type and pass all arguments
        
        # 选定地形的参数字典
        terrain_kwargs = None # Dict of arguments for selected terrain

        # 初始课程难度上限
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.                    # 每块地形长度 [m]
        terrain_width = 4                       # 每块地形宽度 [m]
        
        # 地形行数（难度级别 0-9）
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        # 地形列数（地形类型数量）
        num_cols = 40 # number of terrain cols (types)
        
        # 地形类型比例分配（所有值之和应为 1.0）
        # 默认配置: 跑酷相关地形各占 20%
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,         # 综合跑酷（idx=15）
                        "parkour_hurdle": 0.2,  # 跨栏（idx=16）
                        "parkour_flat": 0.2,    # 平地跑酷（idx=17，用作 env_class 标志）
                        "parkour_step": 0.2,    # 台阶跑酷（idx=18）
                        "parkour_gap": 0.2,     # 跳跃缝隙（idx=19）
                        "demo": 0.0,}           # 演示地形（idx=20）
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        # 超过此角度的斜面将被修正为垂直面
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True                    # 原点高度固定为 0

        num_goals = 8                           # 每条地形上的路径点（waypoint）数量

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4
        # Easy ranges
        class ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [0.3, 0.8] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-1.6, 1.6]

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        """
        域随机化配置（Domain Randomization）
        随机化物理参数以提高策略的 sim-to-real 迁移能力
        """
        randomize_friction = True               # 随机化地面摩擦系数
        friction_range = [0.6, 2.]              # 摩擦系数范围
        randomize_base_mass = True              # 随机化机体附加质量
        added_mass_range = [0., 3.]             # 附加质量范围 [kg]
        randomize_base_com = True               # 随机化质心偏移
        added_com_range = [-0.2, 0.2]           # 质心偏移范围 [m]
        push_robots = True                      # 是否随机推动机器人
        push_interval_s = 8                     # 推动间隔 [s]
        max_push_vel_xy = 0.5                   # 最大推动速度 [m/s]

        randomize_motor = True                  # 随机化电机强度
        motor_strength_range = [0.8, 1.2]       # 电机强度倍率范围

        # 动作延迟配置（模拟真实通信延迟，关键 sim-to-real 参数）
        delay_update_global_steps = 24 * 8000   # 延迟更新的全局步数阈值（=192000）
        action_delay = False                    # 是否启用动作延迟
        action_curr_step = [1, 1]               # 蒸馏阶段延迟步数范围（固定 1 步）
        action_curr_step_scratch = [0, 1]       # base 训练延迟步数范围（0~1步随机）
        action_delay_view = 1                   # 可视化时的延迟步数
        action_buf_len = 8                      # 动作历史缓冲区长度
        
    class rewards:
        """
        奖励函数配置
        正奖励鼓励目标行为（跟踪速度、跟踪偏航），负奖励惩罚不良行为（碰撞、姿态偏差）
        所有 scale 值在初始化时会乘以 dt，奖励函数定义为 LeggedRobot 上的 _reward_<name> 方法
        """
        class scales:
            # tracking rewards
            # === 目标跟踪奖励（正值）===
            tracking_goal_vel = 1.5     # 速度跟踪奖励（exp(-误差²/sigma)）
            tracking_yaw = 0.5          # 偏航角跟踪奖励
            
            # regularization rewards
            # === 正则化惩罚（负值）===
            lin_vel_z = -1.0            # z轴线速度惩罚（抑制弹跳）
            ang_vel_xy = -0.05          # x轴（横滚角）和y轴（俯仰角）的角速度惩罚（抑制角度不稳）
            orientation = -1.           # 重力方向在机器人坐标系下的投影惩罚（抑制倾斜不稳）
            dof_acc = -2.5e-7           # 关节加速度惩罚（平滑运动）
            collision = -10.            # 碰撞惩罚（身体/大腿/小腿碰地）
            action_rate = -0.1          # 动作变化率惩罚（平滑控制）
            delta_torques = -1.0e-7     # 力矩变化惩罚
            torques = -0.00001          # 力矩大小惩罚（节能）
            hip_pos = -0.5              # 髋关节位置惩罚（防止劈叉）
            dof_error = -0.04           # 关节误差惩罚
            feet_stumble = -1           # 绊倒惩罚
            feet_edge = -1              # 踩边缘惩罚

        only_positive_rewards = True    # 是否将负总奖励裁剪为零（避免早终止问题）
        tracking_sigma = 0.2            # 跟踪奖励的 sigma 参数: reward = exp(-error²/sigma)
        soft_dof_pos_limit = 1.         # 关节软限位（URDF 限位的百分比）
        soft_dof_vel_limit = 1          # 
        soft_torque_limit = 0.4
        base_height_target = 1.         # 目标机体高度 [m]
        max_contact_force = 40.         # 最大接触力阈值 [N]





    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt