import json

import torch

from exokern_eval.cli import main
from exokern_eval.loader import TemporalUNet1D


def _write_dummy_checkpoint(path):
    obs_dim = 22
    action_dim = 7
    args = {
        "obs_horizon": 10,
        "pred_horizon": 16,
        "action_horizon": 8,
        "base_channels": 32,
        "cond_dim": 64,
        "num_diffusion_steps": 100,
        "num_inference_steps": 16,
        "noise_schedule": "cosine",
    }
    model = TemporalUNet1D(
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_horizon=args["obs_horizon"],
        base_channels=args["base_channels"],
        cond_dim=args["cond_dim"],
    )
    checkpoint = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "condition": "full_ft",
        "args": args,
        "stats": {
            "obs_min": [-1.0] * obs_dim,
            "obs_range": [2.0] * obs_dim,
            "action_min": [-1.0] * action_dim,
            "action_range": [2.0] * action_dim,
        },
        "model_state_dict": model.state_dict(),
        "val_loss": 0.123456,
    }
    torch.save(checkpoint, path)


def test_offline_cli_writes_checkpoint_metadata_only(tmp_path):
    checkpoint_path = tmp_path / "dummy.pt"
    report_path = tmp_path / "report.json"
    _write_dummy_checkpoint(checkpoint_path)

    exit_code = main(
        [
            "--policy",
            str(checkpoint_path),
            "--offline",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(report_path.read_text())
    assert data["mode"] == "offline"
    assert "results" not in data
    assert "baseline" not in data
    assert data["checkpoint"]["condition"] == "full_ft"
    assert data["checkpoint"]["val_loss"] == 0.123456
