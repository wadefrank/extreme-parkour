"""
XTDog S100 部署配置常量
所有维度、缩放因子、关节映射必须与训练代码完全一致
参考: legged_robot_config.py, xt_parkour_config.py, legged_robot.py
"""

# ============================================================
# 观测维度
# ============================================================
N_PROPRIO = 53          # 3+2+3+4+36+5 (see legged_robot_config.py:84)
N_SCAN = 132            # scandots (部署时视觉模式下填零)
N_PRIV_EXPLICIT = 9     # 3+3+3 (base_lin_vel * 3 组)
N_PRIV_LATENT = 29      # 4+1+12+12 (mass, friction, motor strength)
HISTORY_LEN = 10        # 历史帧数
N_ACTIONS = 12          # 关节数 (4腿 × 3关节)
DEPTH_LATENT_DIM = 32   # 深度编码器输出维度
YAW_CORRECTION_DIM = 2  # 偏航修正维度

# 完整观测维度 = proprio + scan + priv_explicit + priv_latent + history
N_OBS_TOTAL = N_PROPRIO + N_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO
# = 53 + 132 + 9 + 29 + 530 = 753

# ============================================================
# 关节重排序
# ============================================================
# URDF 顺序: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#             RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
# Policy 顺序: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#              RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
REINDEX_URDF_TO_POLICY = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
REINDEX_POLICY_TO_URDF = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # 此排列是自逆的
REINDEX_FEET_URDF_TO_POLICY = [1, 0, 3, 2]  # [FR,FL,RR,RL] -> [FL,FR,RL,RR]

# ============================================================
# 观测缩放因子 (legged_robot_config.py:196-201)
# ============================================================
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_HEIGHT = 5.0
OBS_CLIP = 100.0
ACTION_CLIP = 1.2

# ============================================================
# XTDog 控制参数 (xt_parkour_config.py)
# ============================================================
ACTION_SCALE = 0.3
STIFFNESS = 80.0    # N*m/rad
DAMPING = 3.0       # N*m*s/rad
DECIMATION = 4
DT = 0.005          # 仿真步长 5ms
POLICY_DT = DT * DECIMATION  # 20ms = 50Hz

# XTDog 默认关节角度 (Policy 顺序: FL, FR, RL, RR)
DEFAULT_DOF_POS_POLICY = [
    0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 0.8, -1.5,   # FR: hip, thigh, calf
    0.1, 1.0, -1.5,    # RL: hip, thigh, calf
    -0.1, 1.0, -1.5,   # RR: hip, thigh, calf
]

# XTDog 默认关节角度 (URDF 顺序: FR, FL, RR, RL)
DEFAULT_DOF_POS_URDF = [
    -0.1, 0.8, -1.5,   # FR: hip, thigh, calf
    0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 1.0, -1.5,   # RR: hip, thigh, calf
    0.1, 1.0, -1.5,    # RL: hip, thigh, calf
]

# ============================================================
# 深度相机参数
# ============================================================
DEPTH_IMAGE_HEIGHT = 58
DEPTH_IMAGE_WIDTH = 87
DEPTH_NEAR_CLIP = 0.0   # meters
DEPTH_FAR_CLIP = 2.0    # meters
DEPTH_UPDATE_INTERVAL = 5  # 每 5 个 sim step 更新一次

# 偏航修正缩放 (on_policy_runner.py 中应用)
YAW_CORRECTION_SCALE = 1.5

# ============================================================
# 网络架构参数 (用于重建模型)
# ============================================================
SCAN_ENCODER_DIMS = [128, 64, 32]
ACTOR_HIDDEN_DIMS = [512, 256, 128]
PRIV_ENCODER_DIMS = [64, 20]
ESTIMATOR_HIDDEN_DIMS = [128, 64]
DEPTH_ENCODER_HIDDEN_DIM = 512  # GRU hidden size
CNN_OUTPUT_DIM = 32

# ============================================================
# 设备端运行参数
# ============================================================
CONTROL_FREQ_HZ = 50       # 控制频率
VISION_FREQ_HZ = 30        # 视觉处理频率 (Gemini 335 帧率)
SENSOR_FREQ_HZ = 1000      # 传感器采集频率
GRU_RESET_INTERVAL = 500   # 每 500 步重置 GRU hidden state (安全机制)
