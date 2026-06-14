import re
import matplotlib.pyplot as plt
import pandas as pd

log_file = "/iopsstor/scratch/cscs/faruk_zahiragic/main_run_megatron/Megatron-LM/logs/slurm/training/apertus-300m-mm-ablation-1916759.out"

data = []
pattern = re.compile(
    r"iteration\s+(\d+)/\s+\d+.*lm loss:\s+([\d.E+-]+).*text_token_loss:\s+([\d.E+-]+).*vision_token_loss:\s+([\d.E+-]+).*audio_token_loss:\s+([\d.E+-]+)"
)

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data.append({
                "iteration": int(match.group(1)),
                "lm_loss": float(match.group(2)),
                "text_loss": float(match.group(3)),
                "vision_loss": float(match.group(4)),
                "audio_loss": float(match.group(5))
            })

df = pd.DataFrame(data).drop_duplicates("iteration").sort_values("iteration")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df["iteration"], df["text_loss"], label="Text Loss", alpha=0.8)
plt.plot(df["iteration"], df["audio_loss"], label="Audio Loss", alpha=0.8)
plt.plot(df["iteration"], df["vision_loss"], label="Vision Loss", alpha=0.8)
plt.title("Apertus 300M Vanilla Baseline - Multimodal Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig("apertus_300m_baseline_loss.png")
print("Plot saved to apertus_300m_baseline_loss.png")

# Print final stats
print("\nFinal Iteration Metrics:")
print(df.iloc[-1].to_string())