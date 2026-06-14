# Apertus-300M evaluation setup

This folder is a customization of [swiss-ai/evals](https://github.com/swiss-ai/evals) for running benchmark evals on a custom ~300M-parameter Apertus model.

## Prerequisites

- Access to the CSCS cluster (Clariden) with GPU allocation
- Python deps via `uv sync` from this `evals/` directory
- Environment variables:
  - `WANDB_API_KEY` — required when uploading results to Weights & Biases
  - `HF_TOKEN` — if downloading private Hugging Face assets

## One-time setup

From the repo root:

```bash
cd evals
uv sync
cp configs/eval_300m.example.json configs/eval_300m.json
# edit logs_root, model path, wandb_entity, etc.
```

Required paths in `configs/eval_300m.json`:

| Field | Description |
|-------|-------------|
| `logs_root` | Where harness + wandb logs are written (use `$SCRATCH/eval-logs/`) |
| `models.Apertus-300M.name` | Local HF checkpoint dir or Hugging Face repo id |
| `models.Apertus-300M.extra_env.TOKENIZER` | Apertus 1.5 tokenizer path (shared infra path or your own copy) |
| `wandb_entity` / `wandb_project` | Your W&B destination |

The model checkpoint must be Hugging Face format with Apertus remote code. If you need to prepare/upload one, see `config_modifcation.py`, `model_modification.py`, and the `fixed_*_apertus.py` helpers in this folder.

## Run evals (automated)

```bash
cd evals
export WANDB_API_KEY=...
export HF_TOKEN=...

uv run scripts/automate.py --config-path=configs/eval_300m.json
```

This submits SLURM jobs for any missing benchmark tasks. Add `--sync` to push finished results to W&B automatically.

## Run evals (single job)

```bash
cd evals
export LOGS_ROOT=$SCRATCH/eval-logs/
export BACKEND=hf
export TRANSFORMERS_BRANCH=model/apertus+aimv2-fix
export TOKENIZER=/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-96000
export WANDB_API_KEY=...

sbatch --environment=./containers/env.toml \
  scripts/evaluate.sbatch \
  /path/to/apertus-300m-hf \
  1000 1048576 Apertus-300M
```

Useful optional env vars:

- `TASKS=hellaswag,winogrande` — run a subset instead of the full suite
- `LIMIT=10` — quick smoke test
- `FORCE_SINGLE_GPU=true` — debug on a single GPU

## After jobs finish

Logs go to `evals/logs/eval_Apertus-300M_*_<jobid>.out`. Harness results land under:

```
$LOGS_ROOT/Apertus-300M/iter_1000/harness/
```

Upload to W&B manually:

```bash
export WANDB_API_KEY=...
WANDB_ENTITY=<your-entity> WANDB_PROJECT=apertus-300m \
  python3 scripts/update_wandb.py $SCRATCH/eval-logs/ \
  --name Apertus-300M --it 1000
```

## Notes

- `BACKEND=hf` is intentional for this small custom architecture (not vLLM).
- The custom `model/apertus+aimv2-fix` transformers branch is installed automatically inside the SLURM job.
- SLURM reservation/account settings in `scripts/evaluate.sbatch` may need updating for your allocation.
