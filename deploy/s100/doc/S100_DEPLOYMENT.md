# 地瓜机器人 S100 部署方案

## 1. 目标

将已经训练并蒸馏完成的模型：

```text
legged_gym/logs/parkour_new/020-00-distill/model_7000.pt
```

部署到地瓜机器人 S100 芯片上运行。推荐路线是先在训练机上导出 ONNX，再使用地瓜 OpenExplorer 工具链编译为 S100 可执行的 `.hbm` 模型，最后在板端通过 D-Robotics Runtime/UCP 或配套推理接口接入机器人控制循环。

S100 的模型编译目标建议使用：

```text
march: nash-e
```

如果实际硬件是 S100P，则应改为：

```text
march: nash-m
```

## 2. 当前模型结构

`model_7000.pt` 是视觉蒸馏 checkpoint，包含以下关键权重：

```text
depth_actor_state_dict
depth_encoder_state_dict
estimator_state_dict
model_state_dict
optimizer_state_dict
```

部署时不需要 critic 和 optimizer，只需要：

- `depth_encoder_state_dict`：深度图编码器，输入深度图和本体观测，输出视觉 latent 与 yaw 修正。
- `depth_actor_state_dict`：蒸馏后的策略 actor，使用历史本体观测和视觉 latent 输出动作。
- `estimator_state_dict`：从本体观测估计部署时不可直接获得的显式特权状态。

建议将部署模型拆成两个子模型：

1. `depth_encoder`
   - 输入：
     - `depth_image`: `[1, 58, 87]`
     - `proprio`: `[1, 53]`
     - `h_in`: `[1, 1, 512]`
   - 输出：
     - `depth_latent`: `[1, 32]`
     - `yaw`: `[1, 2]`
     - `h_out`: `[1, 1, 512]`

2. `actor_estimator`
   - 输入：
     - `actor_obs`: `[1, 753]`
     - `depth_latent`: `[1, 32]`
   - 输出：
     - `action`: `[1, 12]`

其中 `actor_obs` 的维度来自：

```text
n_proprio + n_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio
= 53 + 132 + 9 + 29 + 10 * 53
= 753
```

## 3. 离线导出流程

部署目录由脚本按需创建，生成物位于：

```bash
mkdir -p deploy/s100/export deploy/s100/calib deploy/s100/hbm deploy/s100/replay
```

在仓库根目录执行：

```bash
python deploy/s100/scripts/export_onnx.py
```

导出脚本应完成以下工作：

1. 加载 checkpoint：

```python
ckpt = torch.load(
    "legged_gym/logs/parkour_new/020-00-distill/model_7000.pt",
    map_location="cpu",
)
```

2. 构造与训练一致的网络：

- `DepthOnlyFCBackbone58x87`
- `RecurrentDepthBackbone`
- `HardwareVisionNN` 或等价的 actor + estimator 包装模型

3. 加载权重：

```python
depth_encoder.load_state_dict(ckpt["depth_encoder_state_dict"])
actor.load_state_dict(ckpt["depth_actor_state_dict"])
estimator.load_state_dict(ckpt["estimator_state_dict"])
```

4. 导出 ONNX：

```python
torch.onnx.export(
    depth_encoder_wrapper,
    (depth_image, proprio, h_in),
    "deploy/s100/export/depth_encoder.onnx",
    input_names=["depth_image", "proprio", "h_in"],
    output_names=["depth_latent", "yaw", "h_out"],
    opset_version=17,
)

torch.onnx.export(
    actor_estimator,
    (actor_obs, depth_latent),
    "deploy/s100/export/actor_estimator.onnx",
    input_names=["actor_obs", "depth_latent"],
    output_names=["action"],
    opset_version=17,
)
```

注意：当前仓库里的 `save_jit.py` 已经能导出 TorchScript，但 S100 工具链通常以 ONNX 作为转换入口，因此需要补充 ONNX 导出脚本。

## 4. 校准数据采集

量化编译前需要准备校准数据。建议从仿真中采集 100-500 组样本，覆盖平地、台阶、gap、hurdle、parkour 等典型场景。

推荐从脚本目录运行蒸馏策略：

```bash
cd legged_gym/legged_gym/scripts
python play.py \
  --exptid 020-00-distill \
  --checkpoint 7000 \
  --delay \
  --use_camera \
  --task xt_dog \
  --s100_calib_dir ../../../deploy/s100/calib \
  --s100_calib_samples 300
```

采集后从仓库根目录验证目录对齐、shape、dtype 和有限值：

```bash
python deploy/s100/scripts/validate_calibration.py --minimum-samples 100
```

采集内容至少包括：

- `depth_image`: 预处理后的 `[58, 87]` 深度图。
- `proprio`: 53 维本体观测。
- `h_in`: GRU 隐状态。
- `actor_obs`: 753 维 actor 输入。
- `depth_latent`: depth encoder 输出的 32 维 latent。

校准数据不要只采静止站立状态，否则量化后在障碍场景下误差可能明显变大。

## 5. ONNX 与量化一致性验证

导出后先用 ONNXRuntime 验证数值一致性：

```bash
python deploy/s100/scripts/verify_onnx.py \
  --checkpoint legged_gym/logs/parkour_new/020-00-distill/model_7000.pt \
  --onnx-dir deploy/s100/export
```

验收标准建议：

- PyTorch vs ONNXRuntime：`max_abs_error < 1e-4`
- PyTorch vs `.hbm`：动作输出单维误差尽量小于 `0.05`

如果 `.hbm` 误差过大，优先增加校准数据覆盖范围；如果 GRU 量化误差仍不可接受，可以先将 `depth_encoder` 放在 CPU/ONNXRuntime 上运行，只将 `actor_estimator` 编译到 BPU。

## 6. S100 编译流程

本机使用以下地瓜 OpenExplorer CPU Docker 镜像：

```text
registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu
```

以下命令都从 Extreme Parkour 仓库根目录执行。进入容器前，确认 ONNX 已经
导出；正式量化编译还要求 `deploy/s100/calib/` 中已有通过检查的校准数据：

```bash
test -f deploy/s100/export/depth_encoder.onnx
test -f deploy/s100/export/actor_estimator.onnx
python deploy/s100/scripts/validate_calibration.py --minimum-samples 100
```

### 6.1 启动 OpenExplorer 容器

将当前仓库挂载到容器的 `/open_explorer `：

```bash
sudo docker run --gpus all -it \
  -v /home/wade/wade/Code/robotics/extreme-parkour:/open_explorer \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

使用宿主机 UID/GID 可以避免生成的 `.hbm` 和中间文件归属 root。如果镜像内
工具因权限问题无法启动，可以去掉 `--user` 和 `-e HOME=/tmp`，编译完成后
再在宿主机修正 `deploy/s100/hbm/` 的文件权限。

进入容器后先检查工具：

```bash
hb_compile --help
```

### 6.2 快速检查

快速模式用于检查模型能否转换和初步评估性能，不使用正式校准配置。在容器内
执行：

```bash
python3 deploy/s100/scripts/compile_s100.py --fast-perf
```

等价的底层命令为：

```bash
hb_compile --fast-perf \
  --model deploy/s100/export/depth_encoder.onnx \
  --march nash-e
hb_compile --fast-perf \
  --model deploy/s100/export/actor_estimator.onnx \
  --march nash-e
```

### 6.3 正式量化编译

准备两个 `hb_compile` 配置文件：

```text
deploy/s100/depth_encoder_s100.yaml
deploy/s100/actor_estimator_s100.yaml
```

关键配置：

```yaml
march: nash-e
calibration_parameters:
  quant_config:
    model_config:
      all_node_type: 'int16'
      model_output_type: 'int16'
      modelwise_search:
        metric: 'chebyshev'
      layerwise_search:
        metric: 'chebyshev'
    node_config:
      '<optimized Conv node name>':
        input1: 'ec'
```

默认 INT8 在该循环策略上的累计误差较大，尤其会影响 GRU hidden state 和最终
动作。全模型 INT16 虽能显著改善精度，但本模型实测动作最大误差仍约为
`0.15`，且 GRU hidden state 仍会略微超出 `0.05`。正式配置因此保持全模型
INT16，并同时启用 modelwise 和 layerwise Chebyshev 搜索：前者选择整网候选
校准方案，后者再逐节点选择阈值，直接优化最大绝对误差。逐层搜索会显著增加
编译时间。

全模型 INT16 下，Conv/Gemm 的权重仍可能保持 INT8。两份正式配置根据
`*_optimized_float_model.onnx` 和 `*_quant_info.json` 中的优化后节点名，对
Conv 权重输入启用 `ec`（error compensate）。它通过增加补偿计算降低权重量化
误差，因此需要重新检查 HBM 的延迟和内存。重新导出 ONNX 或升级工具链后，
优化节点名可能变化；此时必须同步更新 `node_config`。

不要把全模型 Float16 作为默认配置。S100 BPU 的 Conv、Gemm 和 GRU 主要支持
INT8/INT16；不支持 Float16 的节点会回退到 Float32，可能牺牲实时性。必须以
板端 replay 的动作误差和实测延迟共同验收，不能只看编译器报告的 cosine
similarity。这里将 `quant_config` 直接内联到 YAML；该字段的文件路径解析
方式与 `onnx_model`、`cal_data_dir` 不一致，不使用外部 JSON 可以避免容器
工作目录变化导致路径失效。

确认校准数据有效后，在容器内执行：

```bash
python3 deploy/s100/scripts/validate_calibration.py --minimum-samples 100
python3 deploy/s100/scripts/compile_s100.py
```

等价的底层命令为：

```bash
hb_compile -c deploy/s100/depth_encoder_s100.yaml
hb_compile -c deploy/s100/actor_estimator_s100.yaml
```

预期生成：

```text
deploy/s100/hbm/depth_encoder.hbm
deploy/s100/hbm/actor_estimator.hbm
```

### 6.4 板端模型检查

退出容器后，将 `.hbm` 文件复制到 S100 板端。以下命令应在安装了
`hrt_model_exec` 的 S100 板端运行：

```bash
hrt_model_exec model_info --model_file deploy/s100/hbm/depth_encoder.hbm
hrt_model_exec model_info --model_file deploy/s100/hbm/actor_estimator.hbm
hrt_model_exec perf --model_file deploy/s100/hbm/depth_encoder.hbm --frame_count 200
hrt_model_exec perf --model_file deploy/s100/hbm/actor_estimator.hbm --frame_count 200
```

## 7. 板端运行流程

板端控制程序建议分为三层：

1. 传感器层
   - 读取深度相机。
   - 读取 IMU。
   - 读取 12 个关节的位置、速度和状态。
   - 读取或估计足端接触状态。

2. 推理层
   - 维护 10 帧 `proprio` 历史队列。
   - 维护 `depth_encoder` 的 GRU hidden state。
   - 按 `depth.update_interval` 更新 depth latent。
   - 调用 `actor_estimator.hbm` 输出 12 维动作。

3. 控制层
   - 对动作做 NaN 检查和限幅。
   - 将动作转换为 PD 目标关节角。
   - 下发给电机控制器。
   - 处理急停、超时和跌倒保护。

部署时的关键动作转换公式应与仿真一致：

```text
target_joint_angle = default_joint_angle + action_scale * action
```

当前 parkour 配置中常见 `action_scale` 为 `0.25`，但最终必须以训练该 checkpoint 使用的 robot config 为准。

## 8. 实机安全验证顺序

不要直接上地面跑 parkour。建议按以下顺序验证：

1. 离线 replay
   - 将仿真保存的输入喂给 S100。
   - 比较板端 `.hbm` 输出与 PyTorch 输出。

2. 悬空电机测试
   - 机器人悬空。
   - 动作限幅到训练范围的 20%-30%。
   - 验证关节顺序、方向、零位、PD 增益。

3. 支架测试
   - 机器人接触地面但有支撑保护。
   - 给低速命令，观察是否有持续饱和、抖动或反向关节。

4. 平地低速测试
   - 先禁用大障碍。
   - 逐步增加速度命令和动作限幅。

5. 小障碍测试
   - 从低台阶、低 hurdle 开始。
   - 再验证 gap 和复杂 parkour。

每阶段都应记录：

- 策略推理延迟。
- 控制循环频率。
- 动作饱和率。
- 电机温度和电流。
- 摔倒或急停触发原因。

## 9. 安全保护要求

板端运行必须实现以下保护：

- 输入传感器超时则进入阻尼或站立安全模式。
- BPU 推理超时则复用上一帧安全动作，连续超时触发急停。
- 输出动作包含 NaN 或 Inf 时立即急停。
- 动作超过限幅时裁剪并记录。
- 机器人姿态 roll/pitch 超限时触发跌倒保护。
- 深度图异常时保留最近一次可信 `depth_latent`，连续异常则降级或停机。

## 10. 参考资料

- 地瓜 OpenExplorer 工具链概览：<https://toolchain.d-robotics.cc/en/guide/preface/toolchain_overview.html>
- 地瓜模型量化编译说明：<https://toolchain.d-robotics.cc/guide/ptq/ptq_tool/hb_compile/convert.html>
- S100 ONNX 算子支持表：<https://toolchain.d-robotics.cc/guide/appendix/supported_op_list/operator_support/onnx_operator_support_j6em.html>
- RDK S100 硬件介绍：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Quick_start/hardware_introduction/rdk_s100/>

## 11. 已实现文件

训练机侧导出、校准、编译与 replay：

```text
deploy/s100/scripts/export_onnx.py
deploy/s100/scripts/verify_onnx.py
deploy/s100/scripts/calibration_data.py
deploy/s100/scripts/validate_calibration.py
deploy/s100/scripts/compile_s100.py
deploy/s100/scripts/prepare_replay.py
deploy/s100/scripts/verify_hbm.py
deploy/s100/depth_encoder_s100.yaml
deploy/s100/actor_estimator_s100.yaml
```

板端 runtime：

```text
deploy/s100/runtime/policy_runtime.py
deploy/s100/runtime/hbm_backend.py
deploy/s100/runtime/depth_preprocess.py
deploy/s100/runtime/control_loop.py
```

完整命令见 `deploy/s100/README.md`。由于机器人传感器、电机 SDK 和硬件关节
映射不属于本仓库，实机集成仍需实现 `SensorSource`、`CommandSink`，并在
电机驱动层独立配置 watchdog、硬限位和物理急停。
