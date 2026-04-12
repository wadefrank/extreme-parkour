"""
主推理循环节点
三线程架构:
  - 视觉线程: 深度相机采集 + BPU CNN + CPU GRU
  - 控制线程: 观测构建 + 策略推理 + 电机指令
  - 主线程: 初始化 + 信号处理

用法:
    # 完整运行 (需要 S100 硬件)
    python inference_node.py --cnn_model ../models/cnn_backbone.bin \
                              --gru_model ../models/gru_module.onnx \
                              --policy_model ../models/base_jit.pt

    # 开发机测试 (ONNX fallback + dummy 传感器)
    python inference_node.py --cnn_model ../models/cnn_backbone.onnx \
                              --gru_model ../models/gru_module.onnx \
                              --policy_model ../models/base_jit.pt \
                              --dummy
"""
import os
import sys
import argparse
import time
import threading
import signal
import numpy as np

sys.path.append(os.path.dirname(__file__))

from config import (
    N_PROPRIO, N_OBS_TOTAL, DEPTH_LATENT_DIM,
    CONTROL_FREQ_HZ, POLICY_DT,
)
from depth_camera import DepthCamera, DummyDepthCamera
from observation_builder import ObservationBuilder
from bpu_inference import BPUInference, DummyBPUInference
from gru_inference import GRUInference
from motor_interface import MotorInterface, DummyMotorInterface


class InferenceNode:
    """主推理节点"""

    def __init__(self, args):
        self.args = args
        self._running = False

        # 共享数据 (线程间通信)
        self._depth_latent = np.zeros(DEPTH_LATENT_DIM, dtype=np.float32)
        self._yaw_correction = np.zeros(2, dtype=np.float32)
        self._depth_lock = threading.Lock()

        # 传感器数据 (需要根据实际硬件实现)
        self._sensor_data = {
            "ang_vel": np.zeros(3, dtype=np.float32),
            "roll": 0.0,
            "pitch": 0.0,
            "dof_pos": np.zeros(12, dtype=np.float32),
            "dof_vel": np.zeros(12, dtype=np.float32),
            "contact": np.zeros(4, dtype=np.float32),
        }
        self._sensor_lock = threading.Lock()

        # 初始化组件
        self._init_components()

        # 性能统计
        self._control_times = []
        self._vision_times = []

    def _init_components(self):
        """初始化所有组件"""
        # 深度相机
        if self.args.dummy:
            self.camera = DummyDepthCamera()
        else:
            self.camera = DepthCamera()

        # BPU CNN
        if self.args.dummy and not os.path.exists(self.args.cnn_model):
            self.bpu = DummyBPUInference()
        else:
            backend = "bpu" if self.args.cnn_model.endswith(".bin") else "onnx"
            self.bpu = BPUInference(self.args.cnn_model, backend=backend)

        # GRU
        self.gru = GRUInference(self.args.gru_model)

        # 策略网络
        import torch
        self.policy = torch.jit.load(self.args.policy_model, map_location="cpu")
        self.policy.eval()
        print(f"Policy model loaded: {self.args.policy_model}")

        # 观测构建器
        self.obs_builder = ObservationBuilder()

        # 电机接口
        if self.args.dummy:
            self.motor = DummyMotorInterface()
        else:
            self.motor = DummyMotorInterface()  # TODO: 替换为实际接口

        # 控制状态
        self.cmd_vel_x = self.args.cmd_vel

    def start(self):
        """启动所有线程"""
        self._running = True

        # 启动相机
        self.camera.start()

        # 视觉线程
        self._vision_thread = threading.Thread(
            target=self._vision_loop, daemon=True, name="vision"
        )
        self._vision_thread.start()

        # 控制线程
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="control"
        )
        self._control_thread.start()

        print(f"\nInference node started")
        print(f"  Control freq: {CONTROL_FREQ_HZ} Hz")
        print(f"  Command vel:  {self.cmd_vel_x} m/s")
        print(f"  Dummy mode:   {self.args.dummy}")
        print(f"  Press Ctrl+C to stop\n")

    def stop(self):
        """停止所有线程"""
        self._running = False
        self.camera.stop()
        time.sleep(0.1)
        self._print_stats()

    def wait(self):
        """等待直到停止信号"""
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()

    def _vision_loop(self):
        """视觉线程: 相机采集 + CNN (BPU) + GRU (CPU)"""
        while self._running:
            t0 = time.perf_counter()

            # 获取深度图
            depth_frame = self.camera.get_latest_frame()
            if depth_frame is None:
                time.sleep(0.01)
                continue

            # CNN backbone 推理 (BPU)
            cnn_features = self.bpu.infer(depth_frame)

            # 构建 GRU 用的本体感知 (yaw masked)
            with self._sensor_lock:
                proprio_for_gru = self._build_proprio_for_gru()

            # GRU 推理 (CPU)
            depth_latent, yaw_correction = self.gru.infer(cnn_features, proprio_for_gru)

            # 更新共享数据
            with self._depth_lock:
                self._depth_latent = depth_latent
                self._yaw_correction = yaw_correction

            elapsed = time.perf_counter() - t0
            self._vision_times.append(elapsed)

            # 控制帧率
            sleep_time = 1.0 / 30.0 - elapsed  # ~30Hz
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _control_loop(self):
        """控制线程: 观测构建 + 策略推理 + 电机指令"""
        import torch

        while self._running:
            t0 = time.perf_counter()

            # 读取传感器数据
            with self._sensor_lock:
                ang_vel = self._sensor_data["ang_vel"].copy()
                roll = float(self._sensor_data["roll"])
                pitch = float(self._sensor_data["pitch"])
                dof_pos = self._sensor_data["dof_pos"].copy()
                dof_vel = self._sensor_data["dof_vel"].copy()
                contact = self._sensor_data["contact"].copy()

            # 读取最新 depth latent
            with self._depth_lock:
                depth_latent = self._depth_latent.copy()
                yaw_correction = self._yaw_correction.copy()

            # 构建观测
            obs = self.obs_builder.build(
                ang_vel=ang_vel,
                roll=roll,
                pitch=pitch,
                dof_pos_urdf=dof_pos,
                dof_vel_urdf=dof_vel,
                contact_states_urdf=contact,
                cmd_vel_x=self.cmd_vel_x,
                yaw_correction=yaw_correction,
                is_parkour=True,
            )

            # 策略推理
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                depth_latent_tensor = torch.from_numpy(depth_latent).unsqueeze(0).float()
                action = self.policy(obs_tensor, depth_latent_tensor).squeeze(0).numpy()

            # 记录动作到观测构建器
            self.obs_builder.set_last_action(action)

            # 转换并发送电机指令
            target_pos = self.motor.action_to_target(action)
            self.motor.send_command(target_pos)

            elapsed = time.perf_counter() - t0
            self._control_times.append(elapsed)

            # 控制帧率 50Hz
            sleep_time = POLICY_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _build_proprio_for_gru(self) -> np.ndarray:
        """构建 GRU 输入用的本体感知 (yaw 位置 mask 为 0)"""
        proprio = np.zeros(N_PROPRIO, dtype=np.float32)
        from config import OBS_SCALE_ANG_VEL, OBS_SCALE_DOF_POS, OBS_SCALE_DOF_VEL
        from config import REINDEX_URDF_TO_POLICY, REINDEX_FEET_URDF_TO_POLICY
        from config import DEFAULT_DOF_POS_POLICY

        default_dof = np.array(DEFAULT_DOF_POS_POLICY, dtype=np.float32)
        dof_pos = self._sensor_data["dof_pos"][REINDEX_URDF_TO_POLICY]
        dof_vel = self._sensor_data["dof_vel"][REINDEX_URDF_TO_POLICY]
        contact = self._sensor_data["contact"][REINDEX_FEET_URDF_TO_POLICY]

        proprio[0:3] = self._sensor_data["ang_vel"] * OBS_SCALE_ANG_VEL
        proprio[3] = self._sensor_data["roll"]
        proprio[4] = self._sensor_data["pitch"]
        # [5] = 0, [6:8] = 0 (yaw masked for GRU input)
        # [8:10] = 0
        proprio[10] = self.cmd_vel_x
        proprio[11] = 0.0  # non-parkour flag
        proprio[12] = 1.0  # parkour flag
        proprio[13:25] = (dof_pos - default_dof) * OBS_SCALE_DOF_POS
        proprio[25:37] = dof_vel * OBS_SCALE_DOF_VEL
        proprio[37:49] = self.obs_builder.last_action
        proprio[49:53] = contact.astype(np.float32) - 0.5

        return proprio

    def update_sensors(
        self,
        ang_vel: np.ndarray,
        roll: float,
        pitch: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        contact: np.ndarray,
    ):
        """
        外部调用: 更新传感器数据
        在实际部署中，由传感器线程调用
        """
        with self._sensor_lock:
            self._sensor_data["ang_vel"] = ang_vel.astype(np.float32)
            self._sensor_data["roll"] = float(roll)
            self._sensor_data["pitch"] = float(pitch)
            self._sensor_data["dof_pos"] = dof_pos.astype(np.float32)
            self._sensor_data["dof_vel"] = dof_vel.astype(np.float32)
            self._sensor_data["contact"] = contact.astype(np.float32)

    def _print_stats(self):
        """打印性能统计"""
        if self._control_times:
            ct = np.array(self._control_times) * 1000
            print(f"\nControl loop timing (ms):")
            print(f"  Mean: {ct.mean():.2f}, Max: {ct.max():.2f}, Std: {ct.std():.2f}")
            print(f"  Steps: {len(ct)}")

        if self._vision_times:
            vt = np.array(self._vision_times) * 1000
            print(f"Vision loop timing (ms):")
            print(f"  Mean: {vt.mean():.2f}, Max: {vt.max():.2f}, Std: {vt.std():.2f}")
            print(f"  Steps: {len(vt)}")


def main():
    parser = argparse.ArgumentParser(description="XTDog S100 Inference Node")
    parser.add_argument("--cnn_model", type=str, default="../models/cnn_backbone.onnx")
    parser.add_argument("--gru_model", type=str, default="../models/gru_module.onnx")
    parser.add_argument("--policy_model", type=str, default="../models/base_jit.pt")
    parser.add_argument("--cmd_vel", type=float, default=1.0, help="前进速度指令 (m/s)")
    parser.add_argument("--dummy", action="store_true", help="使用模拟传感器和电机")
    args = parser.parse_args()

    node = InferenceNode(args)

    # 注册信号处理
    def signal_handler(sig, frame):
        node.stop()

    signal.signal(signal.SIGINT, signal_handler)

    node.start()
    node.wait()


if __name__ == "__main__":
    main()
