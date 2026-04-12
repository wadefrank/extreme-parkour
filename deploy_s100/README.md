# XTDog Parkour 部署方案：地平线 S100 + 奥比中光 Gemini 335

## 1. 概述

将 extreme-parkour 项目训练的 distill 视觉策略部署到搭载**地平线 S100** 芯片的 XTDog 四足机器狗上，深度相机为**奥比中光 Gemini 335**（USB 接口），实现 **50Hz** 实时控制频率的 parkour 运动。

### 系统架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                        S100 开发板                            │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Gemini 335  │───>│  视觉线程     │───>│   控制线程      │  │
│  │ 深度相机     │    │  30-40Hz     │    │   50Hz         │  │
│  │ (USB)       │    │              │    │                │  │
│  └─────────────┘    │ BPU: CNN     │    │ CPU: Policy    │  │
│                     │ CPU: GRU     │    │ (base_jit.pt)  │  │
│  ┌─────────────┐    │              │    │                │  │
│  │ IMU/编码器   │    │ depth_latent │    │ actions[12]    │  │
│  │ 足底传感器   │───>│ yaw_corr[2]  │───>│                │──>│ 电机
│  └─────────────┘    └──────────────┘    └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 模型架构

部署涉及两个协同工作的模型：

### 2.1 深度编码器 (RecurrentDepthBackbone)

```
depth_image [1, 58, 87]  +  proprioception [1, 53] (yaw masked)
        │                           │
        ▼                           │
  CNN Backbone (BPU)                │
  Conv2d(1→32, k=5)                │
  MaxPool(k=2, s=2)                │
  Conv2d(32→64, k=3)              │
  FC(62400→128→32)                 │
        │                           │
        ▼                           │
  cnn_features [32] ───────┐       │
                            ▼       ▼
                    Combination MLP (CPU)
                    concat(32+53=85) → FC(85→128→32)
                            │
                            ▼
                      GRU (CPU)
                      input=32, hidden=512
                      hidden state 跨步维持
                            │
                            ▼
                      Output MLP (CPU)
                      FC(512→34) + Tanh
                            │
                            ▼
              depth_latent [32]  +  yaw_correction [2]
```

### 2.2 基础策略 (HardwareVisionNN)

```
obs [1, 753]  +  depth_latent [1, 32]
      │                   │
      ▼                   │
  Estimator (CPU)         │
  proprio[53] → FC(53→128→64→9)   → priv_explicit_est[9]
      │                   │
      ▼                   │
  HistoryEncoder (CPU)    │
  history[10×53] → Conv1d → FC → hist_latent[20]
      │                   │
      ▼                   ▼
  Actor Backbone (CPU)
  concat[53 + 32 + 9 + 20 = 114]
  → FC(114→512→256→128→12)
      │
      ▼
  actions [12]
```

### 2.3 观测向量内存布局 (753 维)

| 区段 | 索引范围 | 维度 | 内容 | 部署来源 |
|------|---------|------|------|---------|
| proprio | `[0:53]` | 53 | 本体感知 | IMU + 关节编码器 |
| scandots | `[53:185]` | 132 | 地形扫描 | **填零** (被 depth_latent 替代) |
| priv_explicit | `[185:194]` | 9 | 特权显式 | **填零** (模型内 Estimator 估计) |
| priv_latent | `[194:223]` | 29 | 特权隐式 | **填零** (被 history_encoder 替代) |
| history | `[223:753]` | 530 | 10帧历史 | 滑动窗口缓冲区 |

### 2.4 本体感知向量 (53 维)

| 索引 | 维度 | 内容 | 来源 | 缩放 |
|------|------|------|------|------|
| 0:3 | 3 | 角速度 (wx, wy, wz) | IMU 陀螺仪 | ×0.25 |
| 3:5 | 2 | roll, pitch | IMU 姿态 | ×1.0 |
| 5 | 1 | 零占位 | 常量 0 | - |
| 6 | 1 | delta_yaw | 深度编码器 | ×1.5 |
| 7 | 1 | delta_next_yaw | 深度编码器 | ×1.5 |
| 8:10 | 2 | 零占位 | 常量 0 | - |
| 10 | 1 | 前进速度指令 | 遥控器/规划 | ×1.0 |
| 11 | 1 | 非 parkour 标志 | 地形类型 | 0 或 1 |
| 12 | 1 | parkour 标志 | 地形类型 | 0 或 1 |
| 13:25 | 12 | 关节角度偏差 | 编码器 (重排序) | ×1.0 |
| 25:37 | 12 | 关节角速度 | 编码器 (重排序) | ×0.05 |
| 37:49 | 12 | 上一步动作 | 策略输出 (重排序) | ×1.0 |
| 49:53 | 4 | 足底接触 | 足底传感器 (重排序) | -0.5 居中 |

---

## 3. 模型拆分与部署策略

将模型拆分为 3 个子模型，分别部署到 BPU 和 CPU：

| 子模型 | 运行位置 | 格式 | 参数量 | 延迟估计 |
|--------|---------|------|--------|---------|
| CNN Backbone | S100 BPU | .bin (INT8) | ~2M (Conv2d+FC) | 3-5ms |
| GRU 模块 | ARM CPU | ONNX (FP32) | ~200K | ~1ms |
| 基础策略 | ARM CPU | TorchScript (.pt) | ~500K | 2-3ms |

**拆分理由：**
- CNN 是唯一计算密集部分，适合 BPU 加速 + INT8 量化
- GRU 有状态操作 (hidden state)，BPU 不支持
- 基础策略全是小 MLP + 1D Conv，CPU 足够快

---

## 4. 实施步骤

### Phase 1：模型导出（在训练机上）

#### 4.1 导出 CNN Backbone 为 ONNX

CNN — 看。把深度图压缩成 32 维视觉特征，回答"前面地形长什么样"

```bash
pip install onnxruntime
cd deploy_s100/export
# python export_cnn_onnx.py --exptid 014-00-distill --checkpoint 2100
python export_cnn_onnx.py --exptid xxx-xx --checkpoint 10000
```

- 从 checkpoint 提取 `depth_encoder_state_dict` 中的 `base_backbone.*` 权重
- 输入: `[1, 1, 58, 87]` float32 (NCHW)
- 输出: `[1, 32]` float32
- 保存到 `deploy_s100/models/cnn_backbone.onnx`

#### 4.2 导出 GRU 模块为 ONNX

GRU — 记。结合当前视觉 + 当前体感 + 之前的记忆，理解地形的时序变化（比如"正在接近台阶"），同时修正偏航角

```bash
# python export_gru_onnx.py --exptid 014-00-distill --checkpoint 2100
python export_gru_onnx.py --exptid xxx-xx --checkpoint 10000
```

- 提取 `combination_mlp.*` / `rnn.*` / `output_mlp.*` 权重
- GRU hidden state 作为显式输入/输出: `[1, 1, 512]`
- 保存到 `deploy_s100/models/gru_module.onnx`

#### 4.3 导出基础策略 (复用现有脚本)

策略网络 — 决策。综合视觉理解、身体状态、历史动作，输出 12 个关节该怎么动

```bash
# cd /root/wade/extreme-parkour/legged_gym/legged_gym/scripts
cd ../../legged_gym/legged_gym/scripts
# python save_jit.py --exptid 014-00-distill --checkpoint 2100
# Saved traced_actor at  /root/wade/extreme-parkour/legged_gym/logs/parkour_new/014-00-distill/traced/014-00-distill-2100-base_jit.pt
python save_jit.py --exptid xxx-xx --checkpoint 10000
# 将 base_jit.pt 复制到 deploy_s100/models/
cp /root/wade/extreme-parkour/legged_gym/logs/parkour_new/014-00-distill/traced/014-00-distill-2100-base_jit.pt /root/wade/extreme-parkour/deploy_s100/models/
```

#### 4.4 收集量化校准数据

```bash
cd deploy_s100/export
# 从仿真收集 (需要 Isaac Gym 环境)
# python collect_calibration.py --exptid 014-00-distill --num_frames 200
python collect_calibration.py --exptid xxx-xx --num_frames 200

# 或生成合成数据 (无需 Isaac Gym)
# python collect_calibration.py --synthetic --num_frames 200
python collect_calibration.py --synthetic --num_frames 200
```

#### 4.5 数值验证

```bash
# python validate_exports.py --exptid 014-00-distill --checkpoint 2100
python validate_exports.py --exptid xxx-xx --checkpoint 10000
```

验证内容：
1. CNN backbone: PyTorch vs ONNX，10 组随机输入，max abs diff < 1e-5
2. GRU 模块: 50 步序列对比，cosine similarity > 0.999
3. base_jit.pt: JIT vs PyTorch，max abs diff < 1e-5

### Phase 2：S100 BPU 模型转换

#### 4.6 ONNX 简化 + BPU 编译

```bash
cd deploy_s100/convert
bash convert.sh
```

脚本自动执行：
1. `onnxsim` 简化 ONNX 图
2. 检查/生成校准数据
3. `hb_mapper makertbin` 转换为 BPU `.bin` 格式

#### 4.7 量化精度验证

转换后需验证 INT8 量化精度：
- 对比 BPU 量化输出 vs float32 参考输出
- 32 维向量 cosine similarity > 0.98
- 若 ELU 导致精度问题，替换为 ReLU 并微调

### Phase 3：设备端部署

#### 4.8 部署文件清单

将以下文件复制到 S100 开发板：

```
deploy_s100/
├── models/
│   ├── cnn_backbone.bin       # BPU 量化模型
│   ├── gru_module.onnx        # CPU GRU 模型
│   └── base_jit.pt            # CPU 策略模型
└── runtime/
    ├── config.py              # 配置常量
    ├── inference_node.py      # 主推理循环
    ├── depth_camera.py        # Gemini 335 驱动
    ├── observation_builder.py # 观测构建
    ├── bpu_inference.py       # BPU 推理封装
    ├── gru_inference.py       # GRU 推理
    └── motor_interface.py     # 电机控制
```

#### 4.9 运行

```bash
# 完整运行 (S100 板上)
cd deploy_s100/runtime
python inference_node.py \
    --cnn_model ../models/cnn_backbone.bin \
    --gru_model ../models/gru_module.onnx \
    --policy_model ../models/base_jit.pt \
    --cmd_vel 1.0

# 开发机 dummy 测试 (无需硬件)
python inference_node.py \
    --cnn_model ../models/cnn_backbone.onnx \
    --gru_model ../models/gru_module.onnx \
    --policy_model ../models/base_jit.pt \
    --dummy
```

---

## 5. 运行时架构

### 5.1 三线程设计

```
┌─────────────────────────────────────────────────────────┐
│  线程 1: 传感器线程 (1000Hz)                              │
│  - IMU 读取 (角速度, roll, pitch)                        │
│  - 关节编码器 (位置, 速度)                                │
│  - 足底接触传感器                                         │
│  - 写入 lock-free 双缓冲                                 │
├─────────────────────────────────────────────────────────┤
│  线程 2: 视觉线程 (~30Hz)                                │
│  - Gemini 335 采集深度帧                                 │
│  - 预处理: crop → resize(87,58) → clip → normalize      │
│  - BPU 推理 CNN backbone        (~3-5ms)                │
│  - CPU 推理 GRU + MLP           (~1ms)                  │
│  - 输出 depth_latent[32] + yaw_correction[2]            │
├─────────────────────────────────────────────────────────┤
│  线程 3: 控制线程 (50Hz, 实时优先级)                       │
│  - 读传感器 → 构建 proprio[53]                           │
│  - 更新历史缓冲区 (10帧滑窗)                              │
│  - 读 depth_latent → 构建 obs[753]                      │
│  - CPU 推理 base_jit.pt         (~2-3ms)                │
│  - 逆重排序 → 计算目标角度 → 电机指令                      │
└─────────────────────────────────────────────────────────┘
```

### 5.2 单周期时序 (20ms)

```
t=0ms    ─── 定时器触发，读取传感器
t=0.5ms  ─── 构建观测 + 历史更新
t=1ms    ─── base_jit.pt 推理
t=3.5ms  ─── 逆重排序，计算目标关节角度
t=4ms    ─── 发送电机指令
t=4-20ms ─── 空闲 / 视觉线程处理下一帧
```

### 5.3 数据流

```
[Gemini 335] ──depth frame──> [预处理] ──[1,58,87]──> [BPU: CNN]
                                                          │
                                                    cnn_feat[32]
                                                          │
[IMU/编码器] ──proprio[53]──> [GRU] <─────────────────────┘
                                │
                          depth_latent[32]
                          yaw_corr[2]
                                │
[IMU/编码器] ──sensor data──> [ObservationBuilder] ──obs[753]──> [base_jit.pt]
                                                                      │
                                                                 actions[12]
                                                                      │
                                                          [MotorInterface]
                                                                      │
                                                               target_pos[12]
                                                                      │
                                                                [CAN/电机]
```

---

## 6. 关键实现细节

### 6.1 关节重排序

硬件 (URDF) 和策略 (Policy) 使用不同的关节顺序：

```
URDF 顺序:   [FR_hip, FR_thigh, FR_calf,   index 0-2
              FL_hip, FL_thigh, FL_calf,   index 3-5
              RR_hip, RR_thigh, RR_calf,   index 6-8
              RL_hip, RL_thigh, RL_calf]   index 9-11

Policy 顺序: [FL_hip, FL_thigh, FL_calf,   index 0-2
              FR_hip, FR_thigh, FR_calf,   index 3-5
              RL_hip, RL_thigh, RL_calf,   index 6-8
              RR_hip, RR_thigh, RR_calf]   index 9-11

重排映射:     policy[i] = urdf[[3,4,5, 0,1,2, 9,10,11, 6,7,8][i]]
```

此排列是**自逆**的（执行两次得到原始顺序），因此正向和反向使用相同的映射。

足底重排：`[FR, FL, RR, RL] → [FL, FR, RL, RR]`，映射为 `[1, 0, 3, 2]`。

### 6.2 深度图预处理

必须与训练流程 (`legged_robot.py`) 完全对齐：

```python
# Gemini 335 原始帧 (640×480, uint16, mm)
depth_m = raw_depth.astype(np.float32) / 1000.0     # → meters
depth_resized = cv2.resize(depth_m, (87, 58))        # → (W=87, H=58)
depth_clipped = np.clip(depth_resized, 0.0, 2.0)     # → [near, far]
normalized = depth_clipped / 2.0 - 0.5               # → [-0.5, 0.5]
```

### 6.3 偏航修正

深度编码器输出 34 维，其中最后 2 维是偏航修正：
- `yaw_correction[0]` → 替换 `obs[6]` (delta_yaw)
- `yaw_correction[1]` → 替换 `obs[7]` (delta_next_yaw)
- 应用缩放: `obs[6:8] = 1.5 * yaw_correction`
- 写入历史缓冲前: `obs[6:8] = 0` (mask yaw)

### 6.4 GRU Hidden State 管理

- 大小: `[1, 1, 512]` float32
- **初始化**: 全零
- **跨步维持**: 每次推理后保留 hidden state 供下次使用
- **重置时机**: episode 边界（站立失败/翻倒后）
- **安全机制**: 每 500 步强制重置一次，防止数值发散

### 6.5 动作输出转换

```python
action_clipped = np.clip(action, -1.2, 1.2)          # clip
action_urdf = action_clipped[REINDEX]                  # policy → URDF 顺序
target_pos = default_dof_pos + action_urdf * 0.3       # action_scale = 0.3
# target_pos 通过 CAN 总线发送给各关节电机控制器
```

### 6.6 XTDog 默认关节角度

| 关节 | FL | FR | RL | RR |
|------|-----|-----|-----|-----|
| hip (rad) | 0.1 | -0.1 | 0.1 | -0.1 |
| thigh (rad) | 0.8 | 0.8 | 1.0 | 1.0 |
| calf (rad) | -1.5 | -1.5 | -1.5 | -1.5 |

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| CNN INT8 量化精度损失 | depth_latent 偏差经 GRU 累积 | cosine sim 验证 > 0.98；必要时替换 ELU 为 ReLU 并微调 |
| GRU hidden state 发散 | 行为异常、动作抖动 | 长序列对比验证；周期性 reset（500步） |
| Gemini 335 与仿真深度图差异 | sim-to-real gap | FOV/分辨率标定对齐；必要时域适应微调 |
| 20ms 控制截止时间违规 | 步态不稳定 | 视觉线程流水线化，depth_latent 允许滞后 1 帧 |
| 观测缩放/重排序不匹配 | 策略输出完全错误 | 录制仿真数据逐元素对比 |

---

## 8. 依赖项

### 开发机 (模型导出)
- Python 3.8+
- PyTorch 1.10+
- onnx, onnxruntime, onnx-simplifier
- numpy, opencv-python

### S100 开发板 (部署运行)
- Python 3.8+
- hobot_dnn (地平线 BPU Python SDK)
- onnxruntime (CPU 版)
- torch (CPU 版, 用于 TorchScript 加载)
- pyorbbecsdk 或 OpenCV with OpenNI2 (Gemini 335 驱动)
- numpy, opencv-python

### 安装 (S100 板)
```bash
pip install onnxruntime numpy opencv-python
# hobot_dnn 随 S100 SDK 预装
# pyorbbecsdk 从 https://github.com/orbbec/pyorbbecsdk 安装
# torch CPU 版:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 9. 需要根据硬件补充的部分

以下部分需要根据 XTDog 的实际硬件接口实现：

1. **`motor_interface.py` → `send_command()`**
   - 根据电机驱动板通信协议（CAN/EtherCAT/串口）实现关节指令发送
   - 需要知道电机 ID 映射和通信帧格式

2. **`inference_node.py` → 传感器读取**
   - 对接 IMU 驱动获取角速度和姿态角
   - 对接关节编码器获取位置和速度
   - 对接足底接触传感器

3. **`depth_camera.py` → FOV 裁剪标定**
   - Gemini 335 水平 FOV 与仿真 87° 对齐
   - 确定 ROI 裁剪区域使深度视野匹配训练设置

---

## 10. 文件清单

```
deploy_s100/
├── README.md                        # 本文档
├── export/                          # 模型导出脚本 (训练机上运行)
│   ├── export_cnn_onnx.py          # CNN backbone → ONNX
│   ├── export_gru_onnx.py          # GRU 模块 → ONNX
│   ├── collect_calibration.py      # BPU 量化校准数据
│   └── validate_exports.py         # 数值一致性验证
├── convert/                         # BPU 转换
│   ├── cnn_config.yaml             # hb_mapper 配置
│   └── convert.sh                  # 转换脚本
├── runtime/                         # 设备端推理代码
│   ├── config.py                   # 常量配置
│   ├── inference_node.py           # 主推理循环 (三线程)
│   ├── depth_camera.py             # Gemini 335 采集+预处理
│   ├── observation_builder.py      # 观测向量构建
│   ├── bpu_inference.py            # BPU/ONNX CNN 推理
│   ├── gru_inference.py            # GRU hidden state 管理
│   └── motor_interface.py          # 电机控制接口
└── models/                          # 模型文件 (导出后存放)
    ├── cnn_backbone.onnx           # CNN ONNX (导出产物)
    ├── cnn_backbone.bin            # CNN BPU (转换产物)
    ├── gru_module.onnx             # GRU ONNX
    └── base_jit.pt                 # 策略 TorchScript
```
