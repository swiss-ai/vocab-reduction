# Project Journal — Apertus-1.5 Vocabulary Reduction

A semester project (EPFL / SwissAI) on **inference- and training-time optimization
of Apertus-1.5 via vocabulary reduction**. Apertus-1.5 is multimodal: a single
output vocabulary of ≈266k tokens spanning text, image, and audio. The embedding
table and LM-head projection over that vocabulary dominate a small model's
parameter count and a large share of its inference cost. The thesis: **prune the
vocabulary to the modality you actually serve** → a much smaller embedding +
LM-head → faster, cheaper inference (and a clean base for modality-specialized
training).

- **Team:** Faruk Zahiragić & Yusif Askari. **Platform:** CSCS Clariden (GH200), HF + Megatron-LM (SwissAI fork).
- **Repo:** `github.com/swiss-ai/vocab-reduction`.

### Timeline (git history)
| Date | Author | Commit | What |
|------|--------|--------|------|
| 2026-03-14 | Faruk | `e272fc0` | Initial commit |
| 2026-03-21 | Faruk | `fa4e3ff` / `bd19485` | Memory & latency benchmarks (+ TPS std-dev) |
| 2026-03-21 | Yusif | `db6bdd7` | **`prune_checkpoint.py`** (the core pruning tool) |
| 2026-03-28 | Faruk | `3aa5d3c` | Embedding/LM-head latency script |
| 2026-04-01 | Faruk | `51a58d4` | HF model-generation/upload utility |
| 2026-04-02 | Yusif | `8b55c08` | Initial Llama-3 multimodal sbatch |
| 2026-04-08 | Faruk | `175e089` | Llama training script |
| 2026-04-18 | Faruk | `1514b1c` | Megatron→HF conversion script |
| 2026-04-23 | Faruk | `e1f1efe` | Apertus-300M test-inference script |
| 2026-04-24 | Faruk | `ccfa578` | HF repo creation on push |
| 2026-05-22 | Yusif | `311b77a` | **Multimodal MoE output head** (origin; later → Megatron-LM) |
| 2026-06-14 | Faruk | `05bc0a3` | Apertus-300M eval pipeline for collaborators |
| 2026-06-20 | Yusif | (this session) | 1B eval config + this journal |

---

## Part I — The vocabulary problem

Apertus-1.5's output vocabulary is laid out as contiguous modality blocks
(recoverable from `tokenizer_config.json` → `omnimodal_config`):

| Block | Range | Size |
|-------|-------|------|
| text (BPE) | `[0, 131072)` | 131072 |
| omni-special | at offset `131072` | small |
| vision (Emu3.5 codebook) | offset `131272`, size `131072` | 131072 |
| audio (WavTokenizer) | offset `262344`, size `4096` | 4096 |

So the input embedding and the (often untied) LM head are each `≈266k × hidden`
matrices. For a 300M-class model that is a *huge* fraction of the parameters and
the per-step output projection. If you only serve text — or only image, or only
audio — most of those rows are dead weight you still pay for every token. Vocab
reduction removes them.

---

## Part II — The pruning tool (`prune_checkpoint.py`)

The core contribution (≈1500 lines). It takes a full multimodal Apertus HF
checkpoint and emits a compact, modality-specific HF checkpoint. Structure:

- **`VocabularyLayout`** — reconstructs the text / omni-special / vision / audio
  `TokenSpan`s from `tokenizer_config.json`.
- **`ModalitySelection`** — which modalities to keep (`text`, `vision`, `audio`,
  any combination) and whether to keep the omni-special tokens.
- **`RemapPlan`** — the heart: builds a `keep_set` of old ids (selected modality
  spans **plus mandatory specials** bos/eos/pad/unk so they're never dropped),
  sorts it, and produces a contiguous `old_to_new = {old_id: new_id}` mapping
  plus the new compacted modality offsets.
- **`SourceArtifacts` / `PreparedArtifacts`** — load and re-emit the JSON sidecars.

End-to-end (`prune_and_save_checkpoint`, with `--prune-model-weights`):
1. Load the model (`AutoModelForCausalLM`, bf16, `trust_remote_code`).
2. Grab input embedding (`nn.Embedding`) and output head (`nn.Linear`).
3. Build `keep_index = tensor(keep_ids)` and **slice both matrices' rows**:
   `weight.index_select(0, keep_index)` (and the head bias if present).
4. Rebuild `nn.Embedding(len(keep_ids), …)` and `nn.Linear(…, len(keep_ids))`,
   copy the sliced weights, `set_input/output_embeddings`, and `tie_weights()` if
   `tie_word_embeddings` is set (tied and untied both handled).
5. Rewrite the sidecars:
   - **`tokenizer.json`** — prune the BPE `vocab`; **filter merges** (keep a rule
     only if left, right, and merged token all survive); remap `added_tokens`,
     padding id, and post-processor special-token ids.
   - **`tokenizer_config.json`** — set `base_vocab_size`/`vocab_size` to the kept
     text count, rewrite `omnimodal_config` with the new offsets, drop the
     `vision_tokenizer`/`audio_tokenizer` blocks for modalities you removed.
   - **`config.json` / `generation_config.json`** — new `vocab_size`, remapped
     `bos/eos/pad_token_id`.
6. Save the model + sidecars + **`index_map.pt`** (`new_id → old_id`, so external
   tools can slice matching weights later).

There's also an **artifacts-only mode** (rewrite the JSON layout without touching
weights). CLI: `--modalities text vision …`, `--prune-model-weights`,
`--keep/drop-omni-special-tokens`, `--device`, `--dtype`, `--trust-remote-code`.

---

## Part III — Conversion & HF packaging

Getting a trained Megatron checkpoint into a *prunable, loadable* HF model took a
small toolchain:

- **`convert_apertus.sbatch`** — Megatron→HF: `torchdist_2_torch.py` (de-shard) →
  `convert.py --loader core --saver swissai_hf --hf-tokenizer <apertus_emu3.5_wavtok>`.
  Its output is the input to `prune_checkpoint.py`.
- **`evals/model_modification.py`** — downloads the repo's `configuration_apertus.py`
  / `modeling_apertus.py` and regex-fixes them for **remote-code** use: relative
  imports (`from ...` / `from ..`) → `from transformers.…`, and the
  `PreTrainedConfig` → `PretrainedConfig` typo. Re-uploads as
  **`fixed_configuration_apertus.py`** / **`fixed_modeling_apertus.py`** (the clean,
  embeddable Apertus HF implementation: RMSNorm, RoPE, GQA, optional QK-norm,
  xIELU MLP, `ApertusForCausalLM` whose `lm_head = Linear(hidden, config.vocab_size)`
  — so vocab reduction is just a smaller `config.vocab_size`).
- **`evals/config_modifcation.py`** *(sic — filename typo)* — injects
  `model_type="apertus"`, the `auto_map`, and `trust_remote_code=True` into
  `config.json`, and replaces the bloated `tokenizer_config.json` with a minimal
  `LlamaTokenizerFast` one.
- **`model_checkpoint_gen.py`** — `create_repo` + `upload_folder` to push a pruned
  checkpoint to the HF Hub (private).
- **`test_inference_apertus_300m.py`** — standardizes the 300M baseline's tokenizer
  config (drops the custom `auto_map` so HF uses stock `LlamaTokenizerFast`).

---

## Part IV — The modality-specific models

From the Apertus-1.5 **iter-96000** checkpoint, pruned modality variants were
produced and published to the HF Hub:

- `Yusif-EPFL/apertus-1p5` — full multimodal baseline.
- `faruk-zahiragic/Apertus-1p5-96000-image-only` — vision-pruned.
- `faruk-zahiragic/Apertus-1p5-96000-audio-only` — audio-pruned.

These three are exactly the targets the benchmark suite compares.

---

## Part V — Benchmarking (does it actually pay off?)

Two harnesses, comparing the full baseline against the pruned variants:

- **`latency_benchmark.py`** — isolates the **LM-head projection**. Loads each
  model (bf16, `cuda:0`), takes its `lm_head`, feeds a `(1,1,hidden)` dummy,
  10-iter warmup, then **1000 timed forward passes via `cuda.Event`**. Reports
  `vocab`, head latency (ms/pass), and max head TPS (`1000/avg_ms`). This directly
  shows the head cost scaling with vocab size.
- **`memory_latency_benchmark.py`** — **end-to-end generation**: 5-iter warmup,
  then **100 runs of 50-token greedy generation** (`do_sample=False`,
  `use_cache=True`) on the prompt *"The future of artificial intelligence in
  Switzerland is"*. Reports model weight footprint (GB), **peak VRAM** (GB,
  `max_memory_allocated`), and **TPS as avg ± std** (`statistics.stdev`, added in
  `bd19485`).

Together they isolate the two wins of vocab reduction: a cheaper LM-head forward
(latency harness) and a smaller weight + memory footprint with steadier
throughput (memory harness).

---

## Part VI — The MoE output head (born here)

`multimodal_moe_head/` (committed `311b77a`) is the **origin** of the two-expert
modality-routed output head — a different answer to the same "the multimodal head
is expensive" problem: instead of statically pruning to one modality, route each
token to a text expert or a vision/audio expert at the output layer. This module
(head, router, experts, config, integration, tests) was later **productionized in
Megatron-LM** with EP=1/EP=2, real-data interleaving, training at 300M/1B/8B, and
HF export. That work has its own detailed journal:
`Megatron-LM/multimodal_moe_head/JOURNAL.md`.

---

## Part VII — Evaluation pipeline

A customization of `swiss-ai/evals` for these small custom Apertus checkpoints
(documented in `evals/README-300M.md`, added by Faruk in `05bc0a3`):

- **Training-loss observability** — `loss_viz.py` parses Megatron logs for
  `lm/text/vision/audio_token_loss` and plots them (`apertus_300m_baseline_loss.png`:
  text → ~3.2, audio → ~5.8, vision → ~8.0 over 8000 steps — vision is the hardest
  modality); `wandb_push.py` pushes the same series to wandb
  (`faruk-zahiragic-epfl/apertus-300m`).
- **Benchmark eval** — `scripts/automate.py` reads a config and submits SLURM jobs
  for the missing tasks; `scripts/evaluate.sbatch` converts (if Megatron) and runs
  lm-evaluation-harness; `scripts/update_wandb.py` aggregates and pushes. Configs:
  - `configs/eval_300m.json` — `apertus-300m-moe-ep2` and `apertus-300m-dense` at
    iter 8000, `BACKEND=hf`, `TRANSFORMERS_BRANCH=model/apertus+aimv2-fix`,
    `FORCE_SINGLE_GPU=true`, wandb `yusif-askari-epfl/apertus-1p5-evals`.
  - `configs/eval_1b.json` *(this session)* — `apertus-1b-climbmix-dense` and
    `apertus-1b-climbmix-moe` at iter 62000; drops `FORCE_SINGLE_GPU` (1B → 4-GPU
    data-parallel) and pins `EXTRA_PIPS=datasets==3.6.0` (the container's
    `datasets` 4.x removed `trust_remote_code`; 2.21.0 is too old; 3.6.0 works).

The 1B ClimbMix dense/MoE checkpoints these eval configs point at were trained in
the Megatron-LM repo (see that journal); the dense-1B eval scores (PIQA 0.701,
ARC-easy 0.646, HellaSwag-norm 0.481, WinoGrande 0.529, ARC-c-norm 0.334, MMLU
0.287) live in wandb `yusif-askari-epfl/apertus-1p5-evals`.

---

## Key artifacts

| Thing | Where |
|-------|-------|
| Pruning tool | `prune_checkpoint.py` (modality slicing + sidecar rewrite + `index_map.pt`) |
| Megatron→HF conversion | `convert_apertus.sbatch` |
| HF remote-code fixes | `evals/{model_modification,config_modifcation}.py`, `evals/fixed_*_apertus.py` |
| Full baseline model | HF `Yusif-EPFL/apertus-1p5` |
| Pruned models | HF `faruk-zahiragic/Apertus-1p5-96000-{image,audio}-only` |
| Benchmarks | `latency_benchmark.py` (LM-head), `memory_latency_benchmark.py` (end-to-end) |
| MoE head (origin) | `multimodal_moe_head/` → productionized in `Megatron-LM/multimodal_moe_head/` |
| Eval pipeline | `evals/` (`README-300M.md`, `configs/eval_{300m,1b}.json`, `loss_viz.py`) |
