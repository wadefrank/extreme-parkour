import torch
import torch.nn as nn
import sys
import torchvision

class RecurrentDepthBackbone(nn.Module):
    """
    循环深度编码器（第二阶段视觉蒸馏的核心网络）

    数据流:
      depth_image(58×87) → CNN(base_backbone) → 32维
      → concat(32 + proprio 53) → MLP → 32维
      → GRU(hidden=512) → 时序融合
      → MLP → 34维 = [32维深度潜在向量 + 2维偏航角修正]

    输出的 32 维深度潜在向量替代 scan_encoder 的输出
    输出的 2 维偏航修正替代观测中被遮蔽的偏航角信息
    """
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone  # CNN 骨干: DepthOnlyFCBackbone58x87
        # 组合 MLP: 将 CNN 输出(32) 与本体感知(53) 融合 → 32 维
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        # GRU 循环层: 捕获时序依赖（深度图每 5 帧更新一次）
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        # 输出 MLP: 512 → 34 (32 深度潜在 + 2 偏航修正)，Tanh 限制输出范围
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = None  # GRU 隐藏状态（跨步保持）

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)       # CNN 编码: [batch, 58, 87] → [batch, 32]
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))  # 融合: [batch, 85] → [batch, 32]
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)  # GRU: [batch, 1, 32] → [batch, 1, 512]
        depth_latent = self.output_mlp(depth_latent.squeeze(1))  # 输出: [batch, 512] → [batch, 34]
        
        return depth_latent  # [:, :32] = 深度潜在, [:, 32:] = 偏航修正

    def detach_hidden_states(self):
        """每次迭代后 detach 隐藏状态，截断 GRU 的反向传播"""
        self.hidden_states = self.hidden_states.detach().clone()

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    """
    深度图 CNN 骨干网络
    输入: 58×87 单通道深度图
    输出: 32 维视觉潜在向量（与 scan_encoder 输出维度匹配）

    网络结构:
      Conv2d(1→32, k=5) → [32, 54, 83]
      MaxPool2d(k=2, s=2) → [32, 27, 41]
      Conv2d(32→64, k=3) → [64, 25, 39]
      Flatten → Linear(62400→128) → Linear(128→32)
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # 输入: [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # → [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # → [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # → [64, 25, 39]
            activation,
            nn.Flatten(),
            # → [64 * 25 * 39 = 62400]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)  # → 32 维
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        # images: [batch, 58, 87] → unsqueeze → [batch, 1, 58, 87]
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent  # [batch, 32]