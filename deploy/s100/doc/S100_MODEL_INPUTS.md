# S100 模型输入说明：Proprio 与 GRU 状态

这里的 `proprio + GRU 状态` 是深度编码器除深度图之外的两个输入。

## 1. Proprio：53 维机器人状态

`proprio` 是当前时刻机器人自身状态和控制指令组成的向量：

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

具体构造见 `legged_gym/legged_gym/envs/base/legged_robot.py` 中的 `compute_observations()`。

注意：输入深度编码器前，偏航差的两个位置会被清零：

```python
obs_student = obs[:, :53].clone()
obs_student[:, 6:8] = 0
```

这是因为深度编码器本身需要根据视觉预测这两个偏航信息。

## 2. GRU 状态：512 维时序记忆

GRU 状态不是直接由传感器测得的数据，而是深度编码器内部产生的“历史记忆”。

S100 ONNX 接口中的形状是：

```text
h_in:  [1, 1, 512]   # 上一次推理留下的状态
h_out: [1, 1, 512]   # 本次推理更新后的状态
```

运行方式：

```text
首次启动：
h_in = 全零

第 1 帧：
深度图 + proprio + h_in → depth_latent + yaw + h_out

第 2 帧：
深度图 + proprio + 上一帧 h_out → 新的输出和 h_out
```

它让编码器结合多帧信息，理解机器人运动趋势、障碍物相对运动和相机噪声，而不只是处理单张深度图。

板端部署时应维护并传递 GRU 状态：

```python
h = zeros([1, 1, 512])

depth_latent, yaw, h = depth_encoder(depth_image, proprio, h)
```

机器人重新启动、控制状态重置或摔倒恢复时，应将 `h` 重新清零。

## 3. GRU 状态与 10 帧 Proprio 历史的区别

GRU 状态和 actor 输入中的“最近 10 帧 proprio 历史”是两套不同的时序信息：

- GRU 状态：深度编码器的内部视觉记忆。
- `10 × 53` 历史观测：actor 使用的显式本体状态历史。
