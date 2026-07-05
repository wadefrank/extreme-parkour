
  - export_onnx.py：从 model_7000.pt 导出 depth_encoder.onnx 和actor_estimator.onnx
  - verify_onnx.py：用 ONNXRuntime 对比 PyTorch 输出
  - s100_models.py：共享模型包装和维度常量