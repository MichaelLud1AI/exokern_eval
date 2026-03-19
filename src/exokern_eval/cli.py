"""CLI entry point: exokern-eval"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from exokern_eval import __version__
from typing import Optional

from exokern_eval.baselines import get_baseline
from exokern_eval.evaluator import EvalResults, create_env, run_rollouts
from exokern_eval.loader import load_policy
from exokern_eval.report import generate_json_report, generate_report


BANNER = f"""\
  ╔══════════════════════════════════════════╗
  ║   EXOKERN-EVAL v{__version__:<26s}║
  ║   Policy Report Card Generator          ║
  ╚══════════════════════════════════════════╝
"""


def _print_table(results: EvalResults, baseline: Optional[dict]) -> None:
    """Print a compact results table to stdout."""
    print()
    print("  ┌─────────────────────┬────────────────┐")
    print("  │ Metric              │ Value          │")
    print("  ├─────────────────────┼────────────────┤")
    print(f"  │ Success Rate        │ {results.success_rate:>10.1f}%    │")
    ci = results.to_dict()["success_rate_ci95"]
    print(f"  │   95% CI            │  [{ci[0]:.1f}%, {ci[1]:.1f}%]  │")
    print(f"  │ Avg Force           │ {results.mean_force:>10.1f} N    │")
    print(f"  │ Peak Force          │ {results.mean_peak_force:>10.1f} N    │")
    print(f"  │ Avg Time            │ {results.mean_time:>10.1f} s    │")
    print(f"  │ Episodes            │ {results.n_episodes:>10d}      │")
    print("  └─────────────────────┴────────────────┘")

    if baseline:
        sr_delta = results.success_rate - baseline["success_rate"]
        force_delta = results.mean_force - baseline["avg_force"]
        print()
        print("  vs EXOKERN Baseline:")
        sr_sym = "+" if sr_delta >= 0 else ""
        f_sym = "+" if force_delta > 0 else ""
        print(f"    Success Rate:  {sr_sym}{sr_delta:.1f}%")
        print(f"    Avg Force:     {f_sym}{force_delta:.1f} N {'(worse)' if force_delta > 0 else '(better)'}")
    print()


def _print_offline_summary(policy: dict) -> None:
    """Print checkpoint metadata when no sim rollouts are executed."""
    print()
    print("  ┌─────────────────────┬────────────────┐")
    print("  │ Checkpoint          │ Value          │")
    print("  ├─────────────────────┼────────────────┤")
    print(f"  │ Condition           │ {policy['condition']:<14} │")
    print(f"  │ Observation Dim     │ {policy['obs_dim']:>14d} │")
    print(f"  │ Action Dim          │ {policy['action_dim']:>14d} │")
    val_loss = policy["val_loss"]
    val_loss_text = f"{val_loss:.6f}" if val_loss is not None else "n/a"
    print(f"  │ Validation Loss     │ {val_loss_text:>14} │")
    print("  └─────────────────────┴────────────────┘")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exokern-eval",
        description="Generate a Policy Report Card by evaluating a checkpoint in Isaac Lab.",
    )
    parser.add_argument("--policy", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--env", type=str, default="Isaac-Forge-PegInsert-Direct-v0", help="Gym env name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of rollout episodes")
    parser.add_argument("--output", type=str, default="report.html", help="Output path (.html or .json)")
    parser.add_argument("--baseline", type=str, default="auto",
                        help="Baseline comparison: 'auto' (EXOKERN SOTA models), path to JSON (your own previous best), or 'none'",)
    parser.add_argument("--condition", type=str, default=None,
                        help="Override condition label (full_ft / no_ft)")
    parser.add_argument("--offline", action="store_true",
                        help="Skip sim rollouts, only report checkpoint metadata")
    parser.add_argument("--version", action="version", version=f"exokern-eval {__version__}")
    args = parser.parse_args(argv)

    print(BANNER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Policy: {args.policy}")
    print(f"  Env:    {args.env}")
    print(f"  Episodes: {args.episodes}")
    print()

    print("  Loading policy...")
    policy = load_policy(args.policy, device)
    condition = args.condition or policy["condition"]
    policy["condition"] = condition
    print(f"  Loaded: condition={condition} obs_dim={policy['obs_dim']} action_dim={policy['action_dim']}")
    if policy["val_loss"] is not None:
        print(f"  Val loss: {policy['val_loss']:.6f}")

    baseline = None
    report_metadata = None

    if args.offline:
        print("\n  OFFLINE MODE: no sim rollouts, reporting checkpoint info only.")
        results = EvalResults(condition=condition, n_episodes=0)
        report_metadata = {
            "mode": "offline",
            "checkpoint": {
                "condition": condition,
                "obs_dim": policy["obs_dim"],
                "action_dim": policy["action_dim"],
                "val_loss": policy["val_loss"],
            },
        }
        if args.baseline != "none":
            print("  Baseline comparison skipped in offline mode.")
        _print_offline_summary(policy)
    else:
        if args.baseline == "auto":
            baseline = get_baseline(args.env, condition)
            if baseline:
                print(f"  Baseline: EXOKERN {args.env} ({condition})")
        elif args.baseline != "none":
            baseline = json.loads(Path(args.baseline).read_text())
            print(f"  Baseline: {args.baseline}")

        print(f"\n  Creating environment: {args.env}")
        env = create_env(args.env)
        if env is None:
            print("  Isaac Lab not available. Use --offline for checkpoint-only report.")
            print("  Or run inside an Isaac Lab container.")
            return 1

        print(f"\n  Running {args.episodes} rollouts...")
        results = run_rollouts(policy, env, args.episodes, device)
        env.close()
        _print_table(results, baseline)

    output_path = Path(args.output)
    if output_path.suffix == ".json":
        out = generate_json_report(
            results, args.env, baseline, output_path, metadata=report_metadata
        )
    else:
        out = generate_report(
            results,
            args.env,
            policy["obs_dim"],
            policy["action_dim"],
            baseline,
            output_path,
            metadata=report_metadata,
        )
    print(f"  Report saved: {out}")

    if baseline and results.n_episodes > 0:
        if results.success_rate >= baseline["success_rate"]:
            print("  Your policy meets or exceeds the EXOKERN baseline.")
        else:
            gap = baseline["success_rate"] - results.success_rate
            print(f"  Your policy is {gap:.1f}% below the EXOKERN baseline.")
            print("  Try the pre-trained skill: https://huggingface.co/EXOKERN/skill-forge-peginsert-v0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
