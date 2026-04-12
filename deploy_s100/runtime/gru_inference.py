"""
GRU 模块 CPU 端推理
管理 GRU hidden state 的生命周期
"""
import numpy as np
import onnxruntime as ort

from config import (
    N_PROPRIO, CNN_OUTPUT_DIM, DEPTH_ENCODER_HIDDEN_DIM,
    DEPTH_LATENT_DIM, YAW_CORRECTION_DIM, GRU_RESET_INTERVAL,
)


class GRUInference:
    """GRU depth encoder 推理 (combination_mlp + GRU + output_mlp)"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: gru_module.onnx 路径
        """
        self._session = ort.InferenceSession(model_path)
        self._hidden_state = np.zeros(
            (1, 1, DEPTH_ENCODER_HIDDEN_DIM), dtype=np.float32
        )
        self._step_count = 0
        print(f"GRU module loaded: {model_path}")

    def reset_hidden(self):
        """重置 GRU hidden state (episode 开始时调用)"""
        self._hidden_state = np.zeros(
            (1, 1, DEPTH_ENCODER_HIDDEN_DIM), dtype=np.float32
        )
        self._step_count = 0

    def infer(
        self, cnn_features: np.ndarray, proprioception: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        执行 GRU 模块推理

        Args:
            cnn_features: [32] float32, CNN backbone 输出
            proprioception: [53] float32, 本体感知 (yaw 已 mask)

        Returns:
            depth_latent: [32] float32
            yaw_correction: [2] float32
        """
        # 安全机制: 周期性重置 hidden state
        if GRU_RESET_INTERVAL > 0 and self._step_count > 0 and self._step_count % GRU_RESET_INTERVAL == 0:
            self.reset_hidden()

        # 准备输入 (添加 batch 维度)
        cnn_feat_batch = cnn_features[np.newaxis, :].astype(np.float32)
        proprio_batch = proprioception[np.newaxis, :].astype(np.float32)

        outputs = self._session.run(
            None,
            {
                "cnn_features": cnn_feat_batch,
                "proprioception": proprio_batch,
                "hidden_state": self._hidden_state,
            },
        )

        output = outputs[0].flatten()  # [34]
        self._hidden_state = outputs[1]  # [1, 1, 512] 更新 hidden state

        depth_latent = output[:DEPTH_LATENT_DIM]
        yaw_correction = output[DEPTH_LATENT_DIM:DEPTH_LATENT_DIM + YAW_CORRECTION_DIM]

        self._step_count += 1
        return depth_latent, yaw_correction
