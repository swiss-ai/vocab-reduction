# vocab-reduction

Inference- and training-time optimization of **Apertus-1.5** via **vocabulary
reduction**. Apertus-1.5 is multimodal with a single ≈266k-token output
vocabulary (text + image + audio); the embedding table and LM-head projection
over that vocabulary dominate a small model's parameters and a large share of its
inference cost. This repo prunes the vocabulary down to the modality you actually
serve — yielding a much smaller embedding + LM head, faster inference, and a
cleaner base for modality-specialized training.

> **Semester project** (EPFL / SwissAI) — Faruk Zahiragić & Yusif Askari.
> A full narrative of the work — the pruning tool, conversion/packaging,
> modality-specific models, benchmarks, the MoE output head, and the eval
> pipeline — is in **[`JOURNAL.md`](JOURNAL.md)**.

## What's here

| Path | Purpose |
|------|---------|
| `prune_checkpoint.py` | Core tool: prune a full multimodal Apertus HF checkpoint to a modality-specific one (slice embedding + LM-head rows, rewrite tokenizer/config, emit `index_map.pt`). |
| `convert_apertus.sbatch` | Megatron-LM → HF conversion (feeds the pruner). |
| `latency_benchmark.py` | LM-head projection latency vs vocab size. |
| `memory_latency_benchmark.py` | End-to-end generation throughput (TPS ± std) and peak VRAM. |
| `model_checkpoint_gen.py`, `test_inference_apertus_300m.py` | HF Hub upload + tokenizer-config standardization. |
| `multimodal_moe_head/` | Origin of the two-expert modality-routed MoE output head (later productionized in Megatron-LM). |
| `evals/` | Apertus eval pipeline (customized `swiss-ai/evals`): HF remote-code fixes, training-loss viz, wandb push, benchmark configs. See `evals/README-300M.md`. |

## Quick start — prune a checkpoint

```bash
# text + vision only, including weights
python prune_checkpoint.py \
  --model-dir /path/to/apertus-full-hf \
  --out-dir   /path/to/apertus-text-vision \
  --modalities text vision \
  --prune-model-weights --device cuda:0 --dtype bfloat16
```

## Benchmarks

```bash
python latency_benchmark.py          # LM-head latency / max TPS per model
python memory_latency_benchmark.py   # weights, peak VRAM, TPS (avg ± std)
```

Both compare the full baseline (`Yusif-EPFL/apertus-1p5`) against the pruned
modality variants (`faruk-zahiragic/Apertus-1p5-96000-{image,audio}-only`).
