"""
观测向量构建器
严格复现 legged_robot.py:compute_observations() 的逻辑
将传感器原始数据转换为策略网络所需的 753 维观测向量
"""
import numpy as np
from config import (
    N_PROPRIO, N_SCAN, N_PRIV_EXPLICIT, N_PRIV_LATENT,
    HISTORY_LEN, N_OBS_TOTAL, N_ACTIONS,
    OBS_SCALE_ANG_VEL, OBS_SCALE_DOF_POS, OBS_SCALE_DOF_VEL,
    OBS_CLIP, YAW_CORRECTION_SCALE,
    REINDEX_URDF_TO_POLICY, REINDEX_FEET_URDF_TO_POLICY,
    DEFAULT_DOF_POS_POLICY,
)


class ObservationBuilder:
    """
    构建策略网络的观测向量

    obs 内存布局 [753]:
      [0:53]    proprioception
      [53:185]  scandots (视觉模式下填零)
      [185:194] priv_explicit (由 Estimator 在模型内部估计)
      [194:223] priv_latent (视觉模式下填零)
      [223:753] history (10 × 53)
    """

    def __init__(self):
        self.default_dof_pos = np.array(DEFAULT_DOF_POS_POLICY, dtype=np.float32)
        self.history_buf = np.zeros((HISTORY_LEN, N_PROPRIO), dtype=np.float32)
        self.last_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.step_count = 0
        self._initialized = False

    def reset(self):
        """episode 重置: 清空历史缓冲区"""
        self.history_buf[:] = 0
        self.last_action[:] = 0
        self.step_count = 0
        self._initialized = False

    def build(
        self,
        ang_vel: np.ndarray,         # [3] 角速度 (rad/s), body frame
        roll: float,                  # roll 角 (rad)
        pitch: float,                 # pitch 角 (rad)
        dof_pos_urdf: np.ndarray,     # [12] 关节角度 (rad), URDF 顺序
        dof_vel_urdf: np.ndarray,     # [12] 关节角速度 (rad/s), URDF 顺序
        contact_states_urdf: np.ndarray,  # [4] 足底接触 (bool/0-1), URDF [FR,FL,RR,RL]
        cmd_vel_x: float,             # 前进速度指令
        yaw_correction: np.ndarray,   # [2] 深度编码器偏航修正
        is_parkour: bool = True,      # 是否为 parkour 地形
    ) -> np.ndarray:
        """
        构建完整观测向量 [753]

        Args:
            ang_vel: IMU 角速度 [wx, wy, wz]
            roll, pitch: IMU 姿态角
            dof_pos_urdf: URDF 顺序关节角度
            dof_vel_urdf: URDF 顺序关节角速度
            contact_states_urdf: URDF 顺序足底接触 [FR, FL, RR, RL]
            cmd_vel_x: 前进速度指令
            yaw_correction: 深度编码器偏航修正 [delta_yaw, delta_next_yaw]
            is_parkour: parkour 地形标志

        Returns:
            obs: [753] float32
        """
        # 关节重排序: URDF -> Policy
        dof_pos = dof_pos_urdf[REINDEX_URDF_TO_POLICY]
        dof_vel = dof_vel_urdf[REINDEX_URDF_TO_POLICY]
        contact = contact_states_urdf[REINDEX_FEET_URDF_TO_POLICY].astype(np.float32)
        action_reindexed = self.last_action  # 已经是 policy 顺序

        # 构建 proprio [53]
        proprio = np.zeros(N_PROPRIO, dtype=np.float32)
        proprio[0:3] = ang_vel * OBS_SCALE_ANG_VEL                    # 角速度
        proprio[3] = roll                                               # roll
        proprio[4] = pitch                                              # pitch
        proprio[5] = 0.0                                                # 占位
        proprio[6] = YAW_CORRECTION_SCALE * yaw_correction[0]         # delta_yaw
        proprio[7] = YAW_CORRECTION_SCALE * yaw_correction[1]         # delta_next_yaw
        proprio[8:10] = 0.0                                            # 占位
        proprio[10] = cmd_vel_x                                        # 前进速度指令
        proprio[11] = 0.0 if is_parkour else 1.0                      # 非 parkour 标志
        proprio[12] = 1.0 if is_parkour else 0.0                      # parkour 标志
        proprio[13:25] = (dof_pos - self.default_dof_pos) * OBS_SCALE_DOF_POS  # 关节角偏差
        proprio[25:37] = dof_vel * OBS_SCALE_DOF_VEL                   # 关节角速度
        proprio[37:49] = action_reindexed                               # 上一步动作
        proprio[49:53] = contact - 0.5                                  # 足底接触（居中）

        # clip
        proprio = np.clip(proprio, -OBS_CLIP, OBS_CLIP)

        # 更新历史缓冲区 (写入前 mask yaw)
        proprio_for_history = proprio.copy()
        proprio_for_history[6:8] = 0.0  # mask yaw in history

        if not self._initialized:
            # episode 开始: 所有历史帧填充当前帧
            self.history_buf[:] = proprio_for_history[np.newaxis, :]
            self._initialized = True
        else:
            # 滑动窗口
            self.history_buf[:-1] = self.history_buf[1:]
            self.history_buf[-1] = proprio_for_history

        # 拼接完整观测 [753]
        obs = np.zeros(N_OBS_TOTAL, dtype=np.float32)
        offset = 0

        # proprio [0:53]
        obs[offset:offset + N_PROPRIO] = proprio
        offset += N_PROPRIO

        # scandots [53:185] 视觉模式下填零 (被 depth_latent 替代)
        offset += N_SCAN

        # priv_explicit [185:194] 填零 (模型内部由 Estimator 估计)
        offset += N_PRIV_EXPLICIT

        # priv_latent [194:223] 填零 (视觉模式下由 history_encoder 替代)
        offset += N_PRIV_LATENT

        # history [223:753]
        obs[offset:offset + HISTORY_LEN * N_PROPRIO] = self.history_buf.flatten()

        self.step_count += 1
        return obs

    def set_last_action(self, action: np.ndarray):
        """记录上一步输出的动作 (policy 顺序)"""
        self.last_action = action.copy()
