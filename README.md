# exokern-eval

**Policy Report Card generator for contact-rich manipulation skills.**

Evaluate any policy checkpoint against standardized sim scenarios and get a publication-ready report with success rate, force profiles, and automatic comparison against EXOKERN baselines.

## Quick Start

```bash
pip install exokern-eval

exokern-eval \
  --policy checkpoint.pt \
  --env Isaac-Forge-PegInsert-Direct-v0 \
  --episodes 100 \
  --output report.html
```

This produces an HTML report card with:
- Success rate with 95% confidence interval
- Average and peak contact force
- Completion time
- Automatic comparison against the EXOKERN baseline (if available)
- Grade badges (Excellent / Good / Needs Work)

## Installation

**Basic (checkpoint metadata only):**
```bash
pip install exokern-eval
```

**With Isaac Lab (full sim rollouts):**
```bash
pip install exokern-eval[isaaclab]
```

**From source:**
```bash
git clone https://github.com/MichaelLud1AI/exokern_eval.git
cd exokern_eval
pip install -e ".[dev]"
```

## Usage

### Full evaluation (requires Isaac Lab + GPU)

```bash
exokern-eval \
  --policy /path/to/best_model.pt \
  --env Isaac-Forge-PegInsert-Direct-v0 \
  --episodes 200 \
  --output report.html
```

### Offline mode (checkpoint info only, no GPU needed)

```bash
exokern-eval \
  --policy checkpoint.pt \
  --offline \
  --output report.json
```

### JSON output for CI/CD

```bash
exokern-eval \
  --policy checkpoint.pt \
  --episodes 100 \
  --output results.json
```

### Custom baseline comparison

```bash
exokern-eval \
  --policy checkpoint.pt \
  --baseline my_baseline.json \
  --output report.html
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | (required) | Path to `.pt` checkpoint |
| `--env` | `Isaac-Forge-PegInsert-Direct-v0` | Gymnasium environment name |
| `--episodes` | `100` | Number of rollout episodes |
| `--output` | `report.html` | Output path (`.html` or `.json`) |
| `--baseline` | `auto` | `auto` (built-in EXOKERN), path to JSON, or `none` |
| `--condition` | auto-detect | Override condition label (`full_ft` / `no_ft`) |
| `--offline` | `false` | Skip sim, report checkpoint metadata only |

## Built-in Baselines

exokern-eval ships with EXOKERN's validated baselines. When `--baseline auto` (the default), your policy is automatically compared:

| Environment | Condition | Success Rate | Avg Force | Source |
|-------------|-----------|:---:|:---:|--------|
| PegInsert | full_ft | 100% | 3.2 N | [EXOKERN/skill-forge-peginsert-v0](https://huggingface.co/EXOKERN/skill-forge-peginsert-v0) |
| PegInsert | no_ft | 100% | 5.2 N | [EXOKERN/skill-forge-peginsert-v0](https://huggingface.co/EXOKERN/skill-forge-peginsert-v0) |

## Example Output

### Terminal

```
  ╔══════════════════════════════════════════╗
  ║   EXOKERN-EVAL v0.1.0                   ║
  ║   Policy Report Card Generator          ║
  ╚══════════════════════════════════════════╝

  Device: cuda
  Policy: results/full_ft_seed42/best_model.pt
  Env:    Isaac-Forge-PegInsert-Direct-v0
  Episodes: 100

  Loading policy...
  Loaded: condition=full_ft obs_dim=22 action_dim=7
  Baseline: EXOKERN Isaac-Forge-PegInsert-Direct-v0 (full_ft)

  Running 100 rollouts...
    Episode 10/100 | Success rate: 100.0%
    ...
    Episode 100/100 | Success rate: 100.0%

  ┌─────────────────────┬────────────────┐
  │ Metric              │ Value          │
  ├─────────────────────┼────────────────┤
  │ Success Rate        │      100.0%    │
  │   95% CI            │  [96.3%, 100.0%]│
  │ Avg Force           │        3.7 N    │
  │ Peak Force          │       10.8 N    │
  │ Avg Time            │       26.9 s    │
  │ Episodes            │        100      │
  └─────────────────────┴────────────────┘

  vs EXOKERN Baseline:
    Success Rate:  +0.0%
    Avg Force:     +0.5 N (worse)

  Report saved: report.html
```

### HTML Report

The HTML report is a single self-contained file with a dark-themed dashboard showing metrics, grades, and baseline comparison bars.

## How It Works

1. **Load** a trained Diffusion Policy checkpoint (`.pt` file with model weights + normalization stats)
2. **Create** an Isaac Lab environment (headless)
3. **Run** N rollouts with DDIM sampling and action chunking
4. **Collect** success, force, and timing metrics per episode
5. **Compare** against built-in EXOKERN baselines
6. **Generate** an HTML or JSON report

## Supported Environments

Currently tested with NVIDIA Isaac Lab FORGE environments:

- `Isaac-Forge-PegInsert-Direct-v0`

More environments will be added as the [EXOKERN Skill Catalog](https://huggingface.co/EXOKERN) grows.

## License

Apache 2.0

## Links

- [EXOKERN on HuggingFace](https://huggingface.co/EXOKERN)
- [EXOKERN Skill: Peg Insert v0](https://huggingface.co/EXOKERN/skill-forge-peginsert-v0)
- [EXOKERN Dataset: ContactBench v0](https://huggingface.co/datasets/EXOKERN/contactbench-forge-peginsert-v0)
