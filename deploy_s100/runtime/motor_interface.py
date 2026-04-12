"""
电机控制接口
将策略输出的动作转换为关节目标位置并发送到电机控制器
"""
import numpy as np
from config import (
    N_ACTIONS, ACTION_SCALE, ACTION_CLIP,
    REINDEX_POLICY_TO_URDF,
    DEFAULT_DOF_POS_URDF,
)


class MotorInterface:
    """电机控制抽象接口"""

    def __init__(self):
        self.default_dof_pos_urdf = np.array(DEFAULT_DOF_POS_URDF, dtype=np.float32)

    def action_to_target(self, action_policy: np.ndarray) -> np.ndarray:
        """
        将策略输出的动作转换为 URDF 顺序的关节目标位置

        Args:
            action_policy: [12] float32, policy 顺序的动作

        Returns:
            target_pos_urdf: [12] float32, URDF 顺序的关节目标角度 (rad)
        """
        # clip action
        action_clipped = np.clip(action_policy, -ACTION_CLIP, ACTION_CLIP)

        # 逆重排: policy [FL,FR,RL,RR] -> URDF [FR,FL,RR,RL]
        action_urdf = action_clipped[REINDEX_POLICY_TO_URDF]

        # 计算目标位置: default_pos + action * scale
        target_pos = self.default_dof_pos_urdf + action_urdf * ACTION_SCALE

        return target_pos

    def send_command(self, target_pos_urdf: np.ndarray):
        """
        发送关节目标位置到电机控制器
        需要根据实际硬件接口实现

        Args:
            target_pos_urdf: [12] URDF 顺序关节目标角度 (rad)
        """
        raise NotImplementedError("需要根据实际硬件通信协议实现")


class DummyMotorInterface(MotorInterface):
    """模拟电机接口，用于测试"""

    def __init__(self):
        super().__init__()
        self.last_command = None

    def send_command(self, target_pos_urdf: np.ndarray):
        self.last_command = target_pos_urdf.copy()


class CANMotorInterface(MotorInterface):
    """CAN 总线电机接口模板"""

    def __init__(self, can_channel="can0", bitrate=1000000):
        super().__init__()
        self.can_channel = can_channel
        self.bitrate = bitrate
        self._bus = None

    def connect(self):
        """连接 CAN 总线"""
        try:
            import can
            self._bus = can.interface.Bus(
                channel=self.can_channel,
                bustype="socketcan",
                bitrate=self.bitrate,
            )
            print(f"CAN bus connected: {self.can_channel}")
        except ImportError:
            raise RuntimeError("python-can not installed. pip install python-can")

    def send_command(self, target_pos_urdf: np.ndarray):
        """通过 CAN 发送关节目标位置"""
        if self._bus is None:
            raise RuntimeError("CAN bus not connected. Call connect() first.")
        # TODO: 根据具体电机驱动协议编码并发送
        # 这里需要根据 XTDog 的电机驱动板协议实现
        pass

    def disconnect(self):
        if self._bus:
            self._bus.shutdown()
