import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

import numpy as np
import matplotlib.pyplot as plt

d = np.load(RESULTS_DIR / "trained_results.npz")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel 1: training loss curves ----
ax = axes[0]
ax.plot(d["loss_attn"], label="Attention", color="#d62728", alpha=0.8, lw=1.2)
ax.plot(d["loss_mc"], label="Memory Caching (GRM)", color="#2ca02c", alpha=0.8, lw=1.2)
ax.axhline(np.log(32), color="gray", ls="--", lw=1, label="chance level (ln 32)")
ax.set_xlabel("training step")
ax.set_ylabel("batch cross-entropy loss")
ax.set_title("Training dynamics\n(both learned via real backprop, not hand-set weights)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 5)

# ---- Panel 2: recall accuracy vs L, trained weights ----
ax = axes[1]
lengths = d["eval_lengths"]
ax.plot(lengths, d["acc_attn"], "o-", label="Attention (trained)", color="#d62728", lw=2, markersize=7)
ax.plot(lengths, d["acc_mc"], "s-", label="Memory Caching (trained, GRM, seg=16)", color="#2ca02c", lw=2, markersize=7)
ax.axhline(1/32, color="gray", ls=":", lw=1, label="chance level (1/32 classes)")
ax.axvspan(16, 64, alpha=0.08, color="blue", label="training length range [16,64]")
ax.set_xscale("log", base=2)
ax.set_xlabel("Sequence length L (test-time, includes 4x beyond training!)")
ax.set_ylabel("Retrieval accuracy")
ax.set_title("Recall generalization: trained models\n(extrapolating past the training length range)")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "trained_comparison.png", dpi=150)
print("saved trained_comparison.png")
