"""Shared S100 deployment model wrappers."""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = REPO_ROOT / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.modules.actor_critic import Actor, get_activation  # noqa: E402
from rsl_rl.modules.depth_backbone import (  # noqa: E402
    DepthOnlyFCBackbone58x87,
    RecurrentDepthBackbone,
)
from rsl_rl.modules.estimator import Estimator  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "legged_gym" / "logs" / "parkour_new" / "020-00-distill" / "model_7000.pt"
)

NUM_PROP = 53
NUM_SCAN = 132
NUM_PRIV_EXPLICIT = 9
NUM_PRIV_LATENT = 29
NUM_HIST = 10
NUM_ACTIONS = 12
ACTOR_OBS_DIM = NUM_PROP + NUM_SCAN + NUM_PRIV_EXPLICIT + NUM_PRIV_LATENT + NUM_HIST * NUM_PROP

DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
DEPTH_LATENT_DIM = 32
DEPTH_YAW_DIM = 2
GRU_HIDDEN_DIM = 512

SCAN_ENCODER_DIMS = [128, 64, DEPTH_LATENT_DIM]
ACTOR_HIDDEN_DIMS = [512, 256, 128]
PRIV_ENCODER_DIMS = [64, 20]
ESTIMATOR_HIDDEN_DIMS = [128, 64]


class DepthEncoderONNX(nn.Module):
    """Depth encoder with explicit GRU hidden-state inputs and outputs."""

    def __init__(self, depth_encoder: RecurrentDepthBackbone):
        super().__init__()
        self.base_backbone = depth_encoder.base_backbone
        self.combination_mlp = depth_encoder.combination_mlp
        self.rnn = depth_encoder.rnn
        self.output_mlp = depth_encoder.output_mlp

    def forward(
        self,
        depth_image: torch.Tensor,
        proprio: torch.Tensor,
        h_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth_features = self.base_backbone(depth_image)
        fused = self.combination_mlp(torch.cat((depth_features, proprio), dim=-1))
        rnn_out, h_out = self.rnn(fused[:, None, :], h_in)
        output = self.output_mlp(rnn_out[:, 0, :])
        depth_latent = output[:, :DEPTH_LATENT_DIM]
        yaw_correction = output[:, DEPTH_LATENT_DIM : DEPTH_LATENT_DIM + DEPTH_YAW_DIM]
        return depth_latent, yaw_correction, h_out


class ActorEstimatorONNX(nn.Module):
    """Actor wrapper that estimates explicit privileged states before policy inference."""

    def __init__(self, actor: Actor, estimator: Estimator):
        super().__init__()
        self.actor = actor
        self.estimator = estimator
        self._priv_start = NUM_PROP + NUM_SCAN
        self._priv_end = self._priv_start + NUM_PRIV_EXPLICIT

    def forward(self, actor_obs: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        proprio = actor_obs[:, :NUM_PROP]
        estimated_priv = self.estimator(proprio)
        obs = torch.cat(
            (
                actor_obs[:, : self._priv_start],
                estimated_priv,
                actor_obs[:, self._priv_end :],
            ),
            dim=1,
        )
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def build_depth_encoder(ckpt: Dict[str, object]) -> DepthEncoderONNX:
    if "depth_encoder_state_dict" not in ckpt:
        raise KeyError("checkpoint is missing depth_encoder_state_dict")

    depth_backbone = DepthOnlyFCBackbone58x87(
        NUM_PROP,
        DEPTH_LATENT_DIM,
        GRU_HIDDEN_DIM,
    )
    depth_encoder = RecurrentDepthBackbone(depth_backbone, env_cfg=None)
    depth_encoder.load_state_dict(ckpt["depth_encoder_state_dict"])
    model = DepthEncoderONNX(depth_encoder)
    model.eval()
    return model


def build_actor_estimator(ckpt: Dict[str, object]) -> ActorEstimatorONNX:
    if "depth_actor_state_dict" not in ckpt:
        raise KeyError("checkpoint is missing depth_actor_state_dict")
    if "estimator_state_dict" not in ckpt:
        raise KeyError("checkpoint is missing estimator_state_dict")

    activation = get_activation("elu")
    actor = Actor(
        NUM_PROP,
        NUM_SCAN,
        NUM_ACTIONS,
        SCAN_ENCODER_DIMS,
        ACTOR_HIDDEN_DIMS,
        PRIV_ENCODER_DIMS,
        NUM_PRIV_LATENT,
        NUM_PRIV_EXPLICIT,
        NUM_HIST,
        activation,
        tanh_encoder_output=False,
    )
    actor.load_state_dict(ckpt["depth_actor_state_dict"], strict=True)

    estimator = Estimator(
        input_dim=NUM_PROP,
        output_dim=NUM_PRIV_EXPLICIT,
        hidden_dims=ESTIMATOR_HIDDEN_DIMS,
    )
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)

    model = ActorEstimatorONNX(actor, estimator)
    model.eval()
    return model


def build_models(checkpoint_path: Path) -> Tuple[DepthEncoderONNX, ActorEstimatorONNX]:
    ckpt = load_checkpoint(checkpoint_path)
    return build_depth_encoder(ckpt), build_actor_estimator(ckpt)


def make_sample_inputs(seed: int = 1) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return {
        "depth_image": torch.randn(1, DEPTH_HEIGHT, DEPTH_WIDTH, generator=generator),
        "proprio": torch.randn(1, NUM_PROP, generator=generator),
        "h_in": torch.randn(1, 1, GRU_HIDDEN_DIM, generator=generator),
        "actor_obs": torch.randn(1, ACTOR_OBS_DIM, generator=generator),
        "depth_latent": torch.randn(1, DEPTH_LATENT_DIM, generator=generator),
    }
