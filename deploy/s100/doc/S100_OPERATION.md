# 第一步

先在训练机上验证 model_7000.pt 能被正确加载和导出，不要先上 S100 板子。

```shell
# 在仓库根目录执行
conda activate extreme_parkour
cd legged_gym/legged_gym/scripts
python save_jit.py --exptid 020-00-distill --checkpoint 7000

# 成功后应生成类似:

legged_gym/logs/parkour_new/020-00-distill/traced/
├── 020-00-distill-7000-base_jit.pt
└── 020-00-distill-7000-vision_weight.pt
```

这一步的意义是确认三件事：

1. checkpoint 路径和权重是完整的。
2. depth_actor_state_dict、estimator_state_dict、depth_encoder_state_dict 能正常加载。
3. 当前 Python 环境能重建部署所需网络。

## 第一步生成文件说明

### `020-00-distill-7000-base_jit.pt`

这是可以通过 `torch.jit.load()` 直接加载的 TorchScript 模型，内部包含：

- `estimator`：根据当前 53 维本体观测估计 9 维显式特权状态。
- `actor`：根据完整 actor 观测和视觉 latent 输出 12 维关节动作。

模型接口为：

```text
输入:
  obs:          [1, 753]
  depth_latent: [1, 32]

输出:
  action:       [1, 12]
```

它不包含深度图编码器，因此不能直接输入深度图。

### `020-00-distill-7000-vision_weight.pt`

这是深度图编码器的权重文件，只包含：

```text
depth_encoder_state_dict
```

它不是可直接执行的 TorchScript 模型。使用时必须先重建
`DepthOnlyFCBackbone58x87 + RecurrentDepthBackbone` 网络，再通过
`load_state_dict()` 加载权重。

深度编码器的部署接口为：

```text
输入:
  depth_image: [1, 58, 87]
  proprio:     [1, 53]
  h_in:        [1, 1, 512]

输出:
  depth_latent:   [1, 32]
  yaw_correction: [1, 2]
  h_out:          [1, 1, 512]
```

两个文件在推理流程中的关系如下：

```text
depth_image + proprio + h_in
                │
                ▼
       depth encoder
  （vision_weight 对应权重）
                │
                ├── depth_latent [1, 32]
                ├── yaw_correction [1, 2]
                └── h_out [1, 1, 512]
                         │
obs [1, 753] ────────────┤
                         ▼
                     base_jit
                         │
                         ▼
                   action [1, 12]
```

这两个 `.pt` 文件是训练机上的加载与导出验证产物，不是最终交付给
S100 的模型。当前 ONNX 导出脚本直接从 `model_7000.pt` 加载三组原始
权重，不依赖这两个中间文件。概念上的对应关系为：

```text
vision_weight 中的深度编码器权重 → depth_encoder.onnx
base_jit 中的 actor + estimator  → actor_estimator.onnx
```

### base_jit.pt

可直接执行的 TorchScript 模型。
- 内含 actor + estimator。
- 输入：
	- obs：1, 753
	- depth_latent：1, 32
- 输出：12 维关节动作。
	- estimator 会先根据 53 维本体观测估计 9 维特权信息，再由 actor 生成动作。见 legged_gym/legged_gym/scripts/save_jit.py:64。


### vision_weight.pt

- 深度视觉编码器的纯权重文件。
- 只保存 depth_encoder_state_dict，并不是可直接运行的 TorchScript 模型。见 legged_gym/legged_gym/scripts/save_jit.py:97。
- 加载时需要重建 DepthOnlyFCBackbone58x87 + RecurrentDepthBackbone。
- 编码器将深度图、本体观测和 GRU 隐状态转换为：
	- 32 维 depth_latent
	- 2 维 yaw_correction
	- 新的 GRU 隐状态。见 deploy/s100/scripts/s100_models.py:48。

## Proprio 输入说明

`proprio` 是当前时刻机器人自身状态和控制指令组成的 53 维向量：

| 维度 | 内容 |
| --- | --- |
| 3 | 机身角速度 `wx, wy, wz` |
| 2 | IMU 的 `roll, pitch` |
| 3 | 1 个占位值、当前目标偏航差、下一目标偏航差 |
| 2 | 指令占位值 |
| 1 | 前进速度指令 |
| 2 | 地形类型标志 |
| 12 | 关节角度相对默认角度的偏差 |
| 12 | 关节速度 |
| 12 | 上一次策略输出动作 |
| 4 | 四只脚的接触状态 |
| **合计** | **53** |

输入深度编码器前，需要将偏航差对应的两个位置清零：

```python
proprio = obs[:, :53].clone()
proprio[:, 6:8] = 0
```

深度编码器会根据视觉信息预测这两个偏航修正值。`proprio` 的缩放、
关节顺序和足端顺序必须与仿真中的 `compute_observations()` 完全一致。

## GRU 状态说明

`h_in` 和 `h_out` 是深度编码器的 512 维时序记忆，不是传感器直接测得
的数据。板端应在首次启动时将其初始化为全零，并在后续推理中把本次
的 `h_out` 作为下一次的 `h_in`：

```python
h = np.zeros((1, 1, 512), dtype=np.float32)

depth_latent, yaw_correction, h = depth_encoder(
    depth_image,
    proprio,
    h,
)
```

机器人重新启动、控制状态重置或摔倒恢复时，应重新将 `h` 清零。

GRU 状态与 actor 输入中的 `10 × 53` 历史观测不是同一份数据：

- GRU 状态是深度编码器内部维护的视觉时序记忆。
- `10 × 53` 历史观测是 actor 使用的显式本体观测历史。

更独立的输入说明见 [S100_MODEL_INPUTS.md](S100_MODEL_INPUTS.md)。

这一步通过后，第二步才是将模型拆成 `depth_encoder.onnx` 和
`actor_estimator.onnx`。

# 第二步

```shell
# 1.回到extreme-parkour根目录
cd ../../..

# 2.检查ONNX依赖
python -c "import onnx, onnxruntime; print('onnx ok')"

# 3.如果报错，先配置onnx环境；没有报错则略过该步骤
pip install onnx onnxruntime

# 4.导出脚本位于：deploy/s100/scripts/export_onnx.py
#  这个脚本要从legged_gym/logs/parkour_new/020-00-distill/model_7000.pt  导出两个文件：
#  deploy/s100/export/depth_encoder.onnx
#  deploy/s100/export/actor_estimator.onnx
```
