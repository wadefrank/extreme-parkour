#!/bin/bash
# S100 BPU 模型转换脚本
# 前置条件: 已安装 Horizon OpenExplorer 工具链 (hb_mapper)
# 用法: bash convert.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

echo "=========================================="
echo "Step 1: ONNX 模型简化"
echo "=========================================="
if ! command -v python -m onnxsim &> /dev/null; then
    echo "Installing onnx-simplifier..."
    pip install onnx-simplifier
fi

python -m onnxsim \
    "${MODEL_DIR}/cnn_backbone.onnx" \
    "${MODEL_DIR}/cnn_backbone_sim.onnx"

echo "Simplified ONNX saved to: ${MODEL_DIR}/cnn_backbone_sim.onnx"

echo ""
echo "=========================================="
echo "Step 2: 检查校准数据"
echo "=========================================="
CAL_DIR="${SCRIPT_DIR}/calibration_data"
if [ ! -f "${CAL_DIR}/depth_images.npy" ]; then
    echo "校准数据不存在，生成合成数据..."
    python "${SCRIPT_DIR}/../export/collect_calibration.py" --synthetic --num_frames 200
fi
echo "校准数据: $(ls -la ${CAL_DIR}/depth_images.npy)"

echo ""
echo "=========================================="
echo "Step 3: hb_mapper BPU 转换"
echo "=========================================="
if ! command -v hb_mapper &> /dev/null; then
    echo "ERROR: hb_mapper not found. 请安装 Horizon OpenExplorer 工具链"
    echo "参考: https://developer.horizon.ai/"
    exit 1
fi

cd "${SCRIPT_DIR}"
hb_mapper makertbin --config cnn_config.yaml --model-type onnx

echo ""
echo "=========================================="
echo "Step 4: 复制输出到 models 目录"
echo "=========================================="
# hb_mapper 默认输出到当前目录的 model_output/
if [ -d "model_output" ]; then
    cp model_output/*.bin "${MODEL_DIR}/cnn_backbone.bin" 2>/dev/null || true
    echo "BPU model saved to: ${MODEL_DIR}/cnn_backbone.bin"
fi

echo ""
echo "=========================================="
echo "转换完成!"
echo "=========================================="
echo "模型文件:"
echo "  BPU CNN:  ${MODEL_DIR}/cnn_backbone.bin"
echo "  CPU GRU:  ${MODEL_DIR}/gru_module.onnx"
echo "  CPU Policy: ${MODEL_DIR}/base_jit.pt"
