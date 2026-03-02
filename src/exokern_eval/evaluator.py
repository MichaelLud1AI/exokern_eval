"""Core evaluation logic: run rollouts and collect metrics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class EvalResults:
    """Aggregated evaluation results for a single policy."""

    condition: str
    n_episodes: int = 0
    successes: list[bool] = field(default_factory=list)
    avg_forces: list[float] = field(default_factory=list)
    max_forces: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    completion_times: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes) * 100) if self.successes else 0.0

    @property
    def mean_force(self) -> float:
        return float(np.mean(self.avg_forces)) if self.avg_forces else 0.0

    @property
    def mean_peak_force(self) -> float:
        return float(np.mean(self.max_forces)) if self.max_forces else 0.0

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.completion_times)) if self.completion_times else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "n_episodes": self.n_episodes,
            "success_rate_pct": round(self.success_rate, 1),
            "avg_force_N": round(self.mean_force, 2),
            "peak_force_N": round(self.mean_peak_force, 2),
            "avg_time_s": round(self.mean_time, 1),
            "success_rate_ci95": self._ci95(self.successes),
        }

    @staticmethod
    def _ci95(successes: list[bool]) -> tuple[float, float]:
        """Wilson score interval for binomial proportion."""
        n = len(successes)
        if n == 0:
            return (0.0, 0.0)
        p = np.mean(successes)
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return (round(max(0, centre - margin) * 100, 1), round(min(1, centre + margin) * 100, 1))


def create_env(env_name: str, num_envs: int = 1):
    """Create an Isaac Lab environment. Returns None if Isaac Lab is not available."""
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": True})
        _ = app_launcher.app

        import isaaclab_tasks  # noqa: F401
        import gymnasium as gym

        try:
            from isaaclab_tasks.utils import parse_env_cfg
        except ImportError:
            from omni.isaac.lab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(env_name, device="cuda:0", num_envs=num_envs)
        env = gym.make(env_name, cfg=env_cfg)
        return env
    except ImportError as e:
        print(f"  Isaac Lab not available: {e}")
        return None


def _extract_obs(env_obs, condition: str, device: torch.device) -> torch.Tensor:
    if isinstance(env_obs, dict):
        obs_tensor = env_obs.get("policy", env_obs.get("obs", list(env_obs.values())[0]))
    else:
        obs_tensor = env_obs
    if not isinstance(obs_tensor, torch.Tensor):
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)
    obs_tensor = obs_tensor.to(device)
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    state_base = obs_tensor[:, :16]
    if condition == "full_ft":
        wrench = obs_tensor[:, 16:22]
        return torch.cat([state_base, wrench], dim=-1)
    return state_base


@torch.no_grad()
def run_rollouts(
    policy: dict[str, Any],
    env,
    num_episodes: int,
    device: torch.device,
    max_steps: int = 200,
    verbose: bool = True,
) -> EvalResults:
    """Run sim rollouts with action chunking and collect metrics."""
    model = policy["model"]
    sampler = policy["sampler"]
    stats = policy["stats"]
    condition = policy["condition"]
    args = policy["args"]

    obs_horizon = args.get("obs_horizon", 10)
    pred_horizon = args.get("pred_horizon", 16)
    action_horizon = args.get("action_horizon", 8)
    action_dim = policy["action_dim"]

    obs_min = torch.tensor(stats["obs_min"], dtype=torch.float32, device=device)
    obs_range = torch.tensor(stats["obs_range"], dtype=torch.float32, device=device)
    action_min = torch.tensor(stats["action_min"], dtype=torch.float32, device=device)
    action_range = torch.tensor(stats["action_range"], dtype=torch.float32, device=device)

    def normalize_obs(obs):
        return 2.0 * (obs - obs_min) / obs_range - 1.0

    def denormalize_action(a):
        return (a + 1.0) / 2.0 * action_range + action_min

    results = EvalResults(condition=condition, n_episodes=num_episodes)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        forces = []
        t0 = time.time()
        obs_window: deque = deque(maxlen=obs_horizon)
        action_buffer: list = []

        while not done and step < max_steps:
            obs_vec = _extract_obs(obs, condition, device)
            obs_norm = normalize_obs(obs_vec)

            if step == 0:
                for _ in range(obs_horizon):
                    obs_window.append(obs_norm)
            else:
                obs_window.append(obs_norm)

            if len(action_buffer) == 0:
                obs_seq = torch.stack(list(obs_window), dim=1)
                shape = (1, pred_horizon, action_dim)
                action_traj_norm = sampler.sample(model, obs_seq, shape)
                action_traj = denormalize_action(action_traj_norm[0])
                action_buffer = list(action_traj[:action_horizon])

            action = action_buffer.pop(0).unsqueeze(0)
            obs, _, terminated, _, info = env.step(action)
            done = terminated.any().item() if isinstance(terminated, torch.Tensor) else terminated

            try:
                fs = env.unwrapped.force_sensor_smooth
                if fs is not None:
                    forces.append(torch.norm(fs[:, :3], dim=-1).item())
            except (AttributeError, Exception):
                pass

            step += 1

        success = False
        try:
            if isinstance(info, dict):
                success = info.get("is_success", False)
            if not success and isinstance(terminated, torch.Tensor):
                success = terminated.any().item() and step < max_steps - 5
        except Exception:
            pass

        results.successes.append(success)
        results.episode_lengths.append(step)
        results.completion_times.append(time.time() - t0)
        if forces:
            results.avg_forces.append(float(np.mean(forces)))
            results.max_forces.append(float(np.max(forces)))

        if verbose and (ep + 1) % 10 == 0:
            print(f"    Episode {ep + 1}/{num_episodes} | Success rate: {results.success_rate:.1f}%")

    return results
