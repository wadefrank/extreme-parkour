"""
S100 BPU 推理封装
加载量化后的 CNN backbone .bin 模型，在 BPU 上执行推理
支持 hobot_dnn (S100 Python API) 和 onnxruntime (开发机 fallback)
"""
import numpy as np
from config import DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, CNN_OUTPUT_DIM


class BPUInference:
    """BPU CNN backbone 推理"""

    def __init__(self, model_path: str, backend: str = "auto"):
        """
        Args:
            model_path: .bin (BPU) 或 .onnx (fallback) 模型路径
            backend: "bpu", "onnx", 或 "auto" (自动检测)
        """
        self.model_path = model_path
        self._model = None

        if backend == "auto":
            backend = "bpu" if model_path.endswith(".bin") else "onnx"

        self.backend = backend
        self._init_model()

    def _init_model(self):
        if self.backend == "bpu":
            try:
                from hobot_dnn import pyeasy_dnn
                self._models = pyeasy_dnn.load(self.model_path)
                self._model = self._models[0]
                print(f"BPU model loaded: {self.model_path}")
                # 打印模型输入输出信息
                for i, inp in enumerate(self._model.inputs):
                    print(f"  Input[{i}]: {inp.properties.shape}, {inp.properties.dtype}")
                for i, out in enumerate(self._model.outputs):
                    print(f"  Output[{i}]: {out.properties.shape}, {out.properties.dtype}")
            except ImportError:
                raise RuntimeError("hobot_dnn not available. Install on S100 board.")

        elif self.backend == "onnx":
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path)
            print(f"ONNX fallback model loaded: {self.model_path}")

    def infer(self, depth_image: np.ndarray) -> np.ndarray:
        """
        执行 CNN backbone 推理

        Args:
            depth_image: [58, 87] float32, 归一化深度图

        Returns:
            cnn_features: [32] float32
        """
        # 添加 batch 和 channel 维度: [58, 87] -> [1, 1, 58, 87]
        input_tensor = depth_image[np.newaxis, np.newaxis, :, :].astype(np.float32)

        if self.backend == "bpu":
            outputs = self._model.forward(input_tensor)
            return np.array(outputs[0].buffer).flatten()[:CNN_OUTPUT_DIM]

        elif self.backend == "onnx":
            outputs = self._session.run(None, {"depth_image": input_tensor})
            return outputs[0].flatten()


class DummyBPUInference:
    """模拟 BPU 推理，用于测试"""

    def infer(self, depth_image: np.ndarray) -> np.ndarray:
        return np.zeros(CNN_OUTPUT_DIM, dtype=np.float32)
