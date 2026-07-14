import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

import numpy as np
import matplotlib.pyplot as plt

acc = np.load(RESULTS_DIR / "accuracy_results.npz")
tim = np.load(RESULTS_DIR / "timing_results.npz")

lengths_acc = acc["lengths"]
lengths_tim = tim["lengths"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# ---- Panel 1: recall accuracy vs L ----
ax = axes[0]
ax.plot(lengths_acc, acc["attention"], "o-", label="Transformer (attention)", color="#d62728", lw=2)
ax.plot(lengths_acc, acc["mc_seg16"], "s-", label="Memory Caching (segment=16)", color="#2ca02c", lw=2)
ax.plot(lengths_acc, acc["mc_seg64"], "^-", label="Memory Caching (segment=64)", color="#1f77b4", lw=2)
ax.plot(lengths_acc, acc["linear_rnn"], "d-", label="Plain linear RNN (fixed memory)", color="#7f7f7f", lw=2)
ax.set_xscale("log", base=2)
ax.set_xlabel("Sequence length L (# stored key-value pairs)")
ax.set_ylabel("Retrieval accuracy (cosine sim to correct value)")
ax.set_title("Recall vs. memory capacity\n(fixed memory forgets as L grows)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Panel 2: analytical FLOPs (log-log, shows true asymptotic slope) ----
ax = axes[1]
ax.plot(lengths_tim, tim["flops_attention"], "o-", label="Transformer: O(L²)", color="#d62728", lw=2)
ax.plot(lengths_tim, tim["flops_mc"], "s-", label="Memory Caching: O(N·L)", color="#2ca02c", lw=2)
ax.plot(lengths_tim, tim["flops_linear_rnn"], "d-", label="Linear RNN: O(L)", color="#7f7f7f", lw=2)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Sequence length L")
ax.set_ylabel("Multiply-add operations (analytical)")
ax.set_title("Compute cost scaling\n(closed-form op counts from the same code)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

# ---- Panel 3: measured wall-clock (honest, with overhead caveat) ----
ax = axes[2]
ax.plot(lengths_tim, tim["attention"], "o-", label="Transformer (measured)", color="#d62728", lw=2)
ax.plot(lengths_tim, tim["mc_seg64"], "s-", label="Memory Caching (measured)", color="#2ca02c", lw=2)
ax.plot(lengths_tim, tim["linear_rnn"], "d-", label="Linear RNN (measured)", color="#7f7f7f", lw=2)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Sequence length L")
ax.set_ylabel("Wall-clock seconds")
ax.set_title("Measured wall-clock\n(Python-loop overhead visible at small L)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison.png", dpi=150)
print("saved comparison.png")
