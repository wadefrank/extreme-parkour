#!/bin/bash
# S100 BPU 模型转换脚本
# 前置条件: 已安装地平线 OpenExplorer 工具链 (hb_compile)
# 参考: https://toolchain.d-robotics.cc/guide/ptq/ptq_tool/hb_compile/convert.html
#
# 安装工具链:
#   方式1 (Docker): docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
#   方式2 (OE包):   wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/oe-package-3.7.0-s100-s600.tgz

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

echo "=========================================="
echo "Step 1: ONNX 模型简化"
echo "=========================================="
python -m onnxsim \
    "${MODEL_DIR}/cnn_backbone.onnx" \
    "${MODEL_DIR}/cnn_backbone_sim.onnx"
echo "Simplified ONNX saved to: ${MODEL_DIR}/cnn_backbone_sim.onnx"

echo ""
echo "=========================================="
echo "Step 2: 检查校准数据"
echo "=========================================="
CAL_DIR="${SCRIPT_DIR}/calibration_data"
if [ ! -d "${CAL_DIR}" ] || [ -z "$(ls ${CAL_DIR}/*.npy 2>/dev/null)" ]; then
    echo "校准数据不存在，生成合成数据..."
    python "${SCRIPT_DIR}/../export/collect_calibration.py" --synthetic --num_frames 100
fi
echo "校准数据目录: ${CAL_DIR}"
echo "样本数量: $(ls ${CAL_DIR}/*.npy 2>/dev/null | wc -l)"

echo ""
echo "=========================================="
echo "Step 3: hb_compile BPU 转换"
echo "=========================================="
if ! command -v hb_compile &> /dev/null; then
    echo "ERROR: hb_compile not found."
    echo ""
    echo "安装方式:"
    echo "  Docker: docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0"
    echo "  OE包:   wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/oe-package-3.7.0-s100-s600.tgz"
    exit 1
fi

cd "${SCRIPT_DIR}"
hb_compile -c cnn_config.yaml

echo ""
echo "=========================================="
echo "Step 4: 复制输出到 models 目录"
echo "=========================================="
if [ -d "model_output" ]; then
    # hb_compile 输出 .hbm 文件
    cp model_output/cnn_backbone.hbm "${MODEL_DIR}/" 2>/dev/null && \
        echo "BPU model saved to: ${MODEL_DIR}/cnn_backbone.hbm" || \
        echo "WARNING: .hbm file not found in model_output/, check hb_compile output"
    echo ""
    echo "转换产物:"
    ls -la model_output/
fi

echo ""
echo "=========================================="
echo "转换完成!"
echo "=========================================="
echo ""
echo "部署所需模型文件:"
echo "  BPU CNN:     ${MODEL_DIR}/cnn_backbone.hbm"
echo "  CPU GRU:     ${MODEL_DIR}/gru_module.onnx"
echo "  CPU Policy:  ${MODEL_DIR}/base_jit.pt"
