# S100 部署

本目录实现 `doc/S100_DEPLOYMENT.md` 中的训练机导出、校准、S100 编译、
离线 replay 和板端策略 runtime。机器人传感器与电机 SDK 由具体硬件决定，
需实现 `runtime/control_loop.py` 中的 `SensorSource` 和 `CommandSink` 接口。

## 1. 导出并验证 ONNX

在仓库根目录和 Extreme Parkour Python 环境中执行：

```bash
python deploy/s100/scripts/export_onnx.py
python deploy/s100/scripts/verify_onnx.py
```

输出位于 `deploy/s100/export/`。验证阈值默认为 `atol=rtol=1e-4`。

## 2. 从仿真采集校准数据

从脚本目录运行带相机的蒸馏策略：

```bash
cd legged_gym/legged_gym/scripts
python play.py \
  --exptid 020-00-distill \
  --checkpoint 7000 \
  --delay \
  --use_camera \
  --task a1 \
  --s100_calib_dir ../../../deploy/s100/calib \
  --s100_calib_samples 300
```

如需替换已有数据，显式增加 `--s100_calib_overwrite`。回到仓库根目录检查：

```bash
python deploy/s100/scripts/validate_calibration.py --minimum-samples 100
```

校准样本按 OpenExplorer 多输入约定分别存放，每个输入目录中的同名 `.npy`
属于同一时刻、同一仿真环境。应分别采集平地、台阶、hurdle、gap 和 parkour，
不要只采静止状态。

## 3. 编译 S100 HBM

以下命令必须在安装了 OpenExplorer 的环境中运行：

```bash
# 快速检查算子和性能，不使用正式校准配置
python deploy/s100/scripts/compile_s100.py --fast-perf

# 使用校准数据正式量化编译
python deploy/s100/scripts/compile_s100.py
```

默认目标是 S100 的 `nash-e`。S100P 应把两个 YAML 中的 `march` 改为
`nash-m`；快速检查可直接传 `--march nash-m`。正式产物为：

```text
deploy/s100/hbm/depth_encoder.hbm
deploy/s100/hbm/actor_estimator.hbm
```

## 4. 准备并执行板端 replay

训练机生成 ONNX 参考输出：

```bash
python deploy/s100/scripts/prepare_replay.py --samples 100
```

将 `deploy/s100/hbm/`、`deploy/s100/replay/` 和本目录代码复制到 S100 后，
运行：

```bash
python deploy/s100/scripts/verify_hbm.py --samples 100 --atol 0.05
```

同时使用 `hrt_model_exec model_info` 和 `perf` 检查模型接口与延迟。

## 5. 接入控制程序

`runtime/hbm_backend.py` 封装两个 HBM 模型，`runtime/policy_runtime.py`
维护 GRU、10 帧历史和安全状态：

```python
from deploy.s100.runtime.hbm_backend import HBMBackend
from deploy.s100.runtime.policy_runtime import RuntimeConfig, S100PolicyRuntime

backend = HBMBackend(
    "deploy/s100/hbm/depth_encoder.hbm",
    "deploy/s100/hbm/actor_estimator.hbm",
)
runtime = S100PolicyRuntime(
    backend,
    RuntimeConfig(action_limit_fraction=0.25),  # 悬空测试阶段
)
```

传给 runtime 的 `proprio` 必须通过 `build_proprio()` 的布局和缩放构造，
深度图必须先执行与训练一致的裁剪、bicubic resize 和归一化。输出目标角为
策略顺序 `FL, FR, RL, RR`；电机接口若使用 URDF 顺序 `FR, FL, RR, RL`，
需调用 `policy_to_hardware_order()`。`PolicyInput.timestamp_s` 必须来自
`time.monotonic()`，否则传感器超时判断无效。

板端 Python 依赖至少包括 `numpy`、`Pillow` 和 BSP 提供的 `hbm_runtime`。

Python 循环不是硬实时安全层。电机驱动侧仍须独立实现通信 watchdog、力矩/
位置硬限位和物理急停。
