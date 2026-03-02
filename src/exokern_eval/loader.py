"""Policy checkpoint loader with auto-detection of architecture."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.mish(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.mish(h)
        return h + self.residual_conv(x)


class TemporalUNet1D(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        obs_horizon: int,
        base_channels: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_horizon * obs_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.Mish(),
        )
        self.input_proj = nn.Conv1d(action_dim, base_channels, 1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1D(ch, out_ch, cond_dim),
                        ConditionalResBlock1D(out_ch, out_ch, cond_dim),
                    ]
                )
            )
            self.downsamples.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        self.mid_block1 = ConditionalResBlock1D(ch, ch, cond_dim)
        self.mid_block2 = ConditionalResBlock1D(ch, ch, cond_dim)

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1))
            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1D(ch + out_ch, out_ch, cond_dim),
                        ConditionalResBlock1D(out_ch, out_ch, cond_dim),
                    ]
                )
            )
            ch = out_ch

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.Mish(),
            nn.Conv1d(base_channels, action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        t_emb = self.time_embed(timestep)
        obs_flat = obs_cond.reshape(batch_size, -1)
        obs_emb = self.obs_encoder(obs_flat)
        cond = self.cond_proj(torch.cat([t_emb, obs_emb], dim=-1))

        x = noisy_actions.permute(0, 2, 1)
        x = self.input_proj(x)

        skip_connections = []
        for (res1, res2), downsample in zip(self.encoder_blocks, self.downsamples):
            x = res1(x, cond)
            x = res2(x, cond)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for (res1, res2), upsample in zip(self.decoder_blocks, self.upsamples):
            x = upsample(x)
            skip = skip_connections.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = x[:, :, : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        x = self.output_proj(x)
        return x.permute(0, 2, 1)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / num_steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999).float()


class DDIMSampler:
    """DDIM sampler for inference."""

    def __init__(
        self,
        num_train_steps: int,
        num_inference_steps: int,
        schedule: str = "cosine",
        device: str = "cpu",
    ):
        betas = cosine_beta_schedule(num_train_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        step_ratio = num_train_steps // num_inference_steps
        self.timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long().to(device)
        self.device = device

    @torch.no_grad()
    def sample(
        self, model: nn.Module, obs_cond: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        timesteps = self.timesteps.flip(0)
        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])
            noise_pred = model(x, t_batch, obs_cond)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=self.device)
            alpha_prev = (
                self.alphas_cumprod[t_prev].view(-1, 1, 1) if t_prev >= 0 else torch.ones_like(alpha_t)
            )
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
            x = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        return x


def load_policy(checkpoint_path: Union[str, Path], device: torch.device) -> dict[str, Any]:
    """Load a trained policy checkpoint and return all components needed for evaluation."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]
    condition = ckpt.get("condition", "unknown")
    args = ckpt.get("args", {})
    stats = ckpt["stats"]

    model = TemporalUNet1D(
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_horizon=args.get("obs_horizon", 10),
        base_channels=args.get("base_channels", 256),
        channel_mults=(1, 2, 4),
        cond_dim=args.get("cond_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sampler = DDIMSampler(
        num_train_steps=args.get("num_diffusion_steps", 100),
        num_inference_steps=args.get("num_inference_steps", 16),
        schedule=args.get("noise_schedule", "cosine"),
        device=str(device),
    )

    return {
        "model": model,
        "sampler": sampler,
        "stats": stats,
        "condition": condition,
        "args": args,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "val_loss": ckpt.get("val_loss", None),
    }
