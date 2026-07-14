"""
Experiment 1 — Recall accuracy vs. number of stored key-value pairs (L).

This reproduces, in miniature, the paper's Needle-in-a-Haystack / in-context
recall story: as you cram more pairs into a FIXED-size memory, interference
grows and retrieval degrades. Attention never forgets (each token individually
cached). Memory Caching sits in between, and gets closer to attention as you
shrink the segment size (more, smaller caches = less interference per cache).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import numpy as np
from mechanisms import make_kv_pairs, attention_retrieve, linear_rnn_memory, \
    linear_rnn_retrieve, build_segment_memories, mc_retrieve

d_k, d_v = 64, 64
lengths = [8, 16, 32, 64, 128, 256, 512, 1024]
seg_sizes = [16, 64]           # two MC configurations: fine-grained vs coarse
n_trials = 30                  # average over random query pairs / seeds


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def run():
    results = {
        "attention": [],
        "linear_rnn": [],
        **{f"mc_seg{c}": [] for c in seg_sizes},
    }

    for L in lengths:
        accs = {k: [] for k in results}
        for trial in range(n_trials):
            K, V = make_kv_pairs(L, d_k, d_v, seed=trial * 1000 + L)
            j = np.random.default_rng(trial + L).integers(0, L)   # which pair we query
            q = K[j]                                              # exact key query

            # 1) Transformer: exact attention over all L pairs
            y_attn = attention_retrieve(K, V, q)
            accs["attention"].append(cosine(y_attn, V[j]))

            # 2) Plain linear RNN: single fixed-size matrix, all L pairs crammed in
            M = linear_rnn_memory(K, V)
            y_rnn = linear_rnn_retrieve(M, q)
            accs["linear_rnn"].append(cosine(y_rnn, V[j]))

            # 3) Memory Caching: segmented + gated
            for c in seg_sizes:
                mems, mean_keys = build_segment_memories(K, V, c)
                y_mc = mc_retrieve(mems, mean_keys, q)
                accs[f"mc_seg{c}"].append(cosine(y_mc, V[j]))

        for k in results:
            results[k].append(np.mean(accs[k]))

        print(f"L={L:5d}  " + "  ".join(f"{k}={np.mean(accs[k]):.3f}" for k in results))

    return lengths, results


if __name__ == "__main__":
    lengths, results = run()
    np.savez(RESULTS_DIR / "accuracy_results.npz", lengths=lengths, **results)
