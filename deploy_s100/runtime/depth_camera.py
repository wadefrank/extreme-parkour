"""
奥比中光 Gemini 335 深度相机采集与预处理
将原始深度帧转换为模型所需的 [58, 87] 归一化输入
"""
import numpy as np
import cv2
import threading
import time

from config import (
    DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH,
    DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP,
)


class DepthCamera:
    """Gemini 335 深度相机封装"""

    def __init__(self, device_index=0, target_fps=30):
        self.target_fps = target_fps
        self.device_index = device_index

        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Orbbec SDK pipeline (延迟初始化)
        self._pipeline = None

    def start(self):
        """启动相机采集线程"""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止相机采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipeline:
            self._pipeline.stop()

    def get_latest_frame(self) -> np.ndarray | None:
        """
        获取最新预处理后的深度图
        Returns:
            depth: [58, 87] float32, 范围 [-0.5, 0.5]，或 None
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _capture_loop(self):
        """相机采集线程主循环"""
        self._init_camera()
        while self._running:
            raw_depth = self._read_frame()
            if raw_depth is not None:
                processed = self.preprocess_depth(raw_depth)
                with self._lock:
                    self._latest_frame = processed
            time.sleep(1.0 / self.target_fps)

    def _init_camera(self):
        """
        初始化相机
        支持两种方式:
        1. Orbbec SDK (推荐, 需要 pyorbbecsdk)
        2. OpenCV (备用, 通过 OpenNI2 后端)
        """
        try:
            # 尝试 Orbbec SDK
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
            self._pipeline = Pipeline()
            config = Config()
            # 配置深度流
            depth_profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            self._pipeline.start(config)
            self._use_orbbec_sdk = True
            print(f"Gemini 335 initialized via Orbbec SDK")
        except (ImportError, Exception) as e:
            print(f"Orbbec SDK not available ({e}), trying OpenCV...")
            # 备用: OpenCV + OpenNI2
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_OPENNI2)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.device_index)
            self._use_orbbec_sdk = False
            print(f"Camera initialized via OpenCV (device {self.device_index})")

    def _read_frame(self) -> np.ndarray | None:
        """读取一帧原始深度图 (uint16, mm)"""
        if self._use_orbbec_sdk:
            frameset = self._pipeline.wait_for_frames(100)
            if frameset is None:
                return None
            depth_frame = frameset.get_depth_frame()
            if depth_frame is None:
                return None
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            return data.reshape(depth_frame.get_height(), depth_frame.get_width())
        else:
            ret, frame = self._cap.read()
            if not ret:
                return None
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            return frame.astype(np.uint16)

    @staticmethod
    def preprocess_depth(raw_depth: np.ndarray) -> np.ndarray:
        """
        深度图预处理，与仿真训练流程对齐
        参考: legged_robot.py 的 crop_depth_image + normalize_depth_image

        Args:
            raw_depth: [H, W] uint16, 单位 mm

        Returns:
            processed: [58, 87] float32, 范围 [-0.5, 0.5]
        """
        # uint16 mm -> float32 meters
        depth_m = raw_depth.astype(np.float32) / 1000.0

        # resize 到目标尺寸 (width=87, height=58)
        depth_resized = cv2.resize(
            depth_m,
            (DEPTH_IMAGE_WIDTH, DEPTH_IMAGE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # clip 到有效范围 [near_clip, far_clip] = [0, 2] meters
        depth_clipped = np.clip(depth_resized, DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP)

        # 归一化到 [-0.5, 0.5]
        # 训练中: depth 存为负值 → 取反 → normalize
        # 等价于: (depth - near) / (far - near) - 0.5
        depth_range = DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP
        if depth_range > 0:
            normalized = (depth_clipped - DEPTH_NEAR_CLIP) / depth_range - 0.5
        else:
            normalized = np.zeros_like(depth_clipped)

        return normalized


class DummyDepthCamera:
    """模拟深度相机，用于无相机环境的测试"""

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def start(self):
        pass

    def stop(self):
        pass

    def get_latest_frame(self) -> np.ndarray:
        return self._rng.uniform(-0.5, 0.5, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH)).astype(np.float32)
