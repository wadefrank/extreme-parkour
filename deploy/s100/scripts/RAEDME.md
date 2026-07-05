完整说明见 [`../README.md`](../README.md)。

- `export_onnx.py`：从 checkpoint 导出两个 ONNX 模型。
- `verify_onnx.py`：比较 PyTorch 与 ONNXRuntime 输出。
- `calibration_data.py`：保存 OpenExplorer 多输入校准数据。
- `validate_calibration.py`：检查校准数据数量、形状和对齐关系。
- `compile_s100.py`：调用 OpenExplorer 编译 S100/S100P 模型。
- `prepare_replay.py`：生成带 ONNX 参考输出的离线 replay。
- `verify_hbm.py`：在板端比较 HBM 与 ONNX 参考输出。
