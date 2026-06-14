import re
import wandb

# Config
log_file = "/iopsstor/scratch/cscs/faruk_zahiragic/main_run_megatron/Megatron-LM/logs/slurm/training/apertus-300m-mm-ablation-1916759.out"
entity = "faruk-zahiragic-epfl"
project = "apertus-300m"
run_name = "apertus-300m-vanilla-baseline-recovery"

# Initialize WandB
wandb.init(entity=entity, project=project, name=run_name, job_type="log_recovery")

# Regex to capture all modality losses
pattern = re.compile(
    r"iteration\s+(\d+)/\s+\d+.*lm loss:\s+([\d.E+-]+).*text_token_loss:\s+([\d.E+-]+).*vision_token_loss:\s+([\d.E+-]+).*audio_token_loss:\s+([\d.E+-]+)"
)

print(f"Parsing {log_file} and uploading to WandB...")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            it = int(match.group(1))
            wandb.log({
                "iteration": it,
                "train/lm_loss": float(match.group(2)),
                "train/text_loss": float(match.group(3)),
                "train/vision_loss": float(match.group(4)),
                "train/audio_loss": float(match.group(5)),
            }, step=it)

wandb.finish()
print("Done! Check your dashboard at wandb.ai")