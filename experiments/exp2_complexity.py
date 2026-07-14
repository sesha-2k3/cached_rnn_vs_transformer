"""
Experiment 2 — Compute cost vs sequence length L.

For each mechanism we run a FULL sequence pass: build memory/state over all L
tokens, then retrieve for all L query positions (the causal, "generate every
token" workload a real model actually does). We time it directly to show the
O(L^2) / O(N*L) / O(L) scaling predicted by the math.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import time
import numpy as np
from mechanisms import make_kv_pairs, attention_retrieve, linear_rnn_memory, \
    linear_rnn_retrieve, build_segment_memories, mc_retrieve

d_k, d_v = 64, 64
lengths = [32, 64, 128, 256, 512, 1024, 2048]
seg_size = 64


def time_attention(K, V):
    t0 = time.perf_counter()
    for i in range(len(K)):
        attention_retrieve(K[:i + 1], V[:i + 1], K[i])   # causal: only past tokens
    return time.perf_counter() - t0


def time_linear_rnn(K, V):
    t0 = time.perf_counter()
    M = np.zeros((d_v, d_k))
    for i in range(len(K)):
        M = M + np.outer(V[i], K[i])          # O(d^2) update per token
        _ = linear_rnn_retrieve(M, K[i])       # O(d^2) retrieval per token
    return time.perf_counter() - t0


def time_mc(K, V, seg_size):
    t0 = time.perf_counter()
    L = len(K)
    mems, mean_keys = [], []
    cur_K, cur_V = [], []
    for i in range(L):
        cur_K.append(K[i]); cur_V.append(V[i])
        # online (in-progress) segment always counts as an active memory too
        Ks, Vs = np.array(cur_K), np.array(cur_V)
        active_mems = mems + [Vs.T @ Ks]
        active_mean_keys = mean_keys + [Ks.mean(axis=0)]
        _ = mc_retrieve(active_mems, active_mean_keys, K[i])   # O(N) retrieval per token
        if len(cur_K) == seg_size:
            mems.append(Vs.T @ Ks)
            mean_keys.append(Ks.mean(axis=0))
            cur_K, cur_V = [], []
    return time.perf_counter() - t0


def analytical_flops(lengths, d_k, d_v, seg_size):
    """
    Closed-form op counts matching the code's own operations, so we see the
    TRUE asymptotic trend (O(L^2), O(L), O(N*L)) without Python-interpreter
    overhead drowning it out at small L.
      attention: at position i, scoring costs ~i*d_k, weighted sum ~i*d_v
      linear_rnn: O(d_k*d_v) update + O(d_k*d_v) retrieval, per token
      mc: per token, ~ (i // seg_size) active memories, each a d_k*d_v matvec,
          plus rebuilding one segment matrix every seg_size tokens
    """
    attn, rnn, mc = [], [], []
    for L in lengths:
        attn.append(sum(i * (d_k + d_v) for i in range(1, L + 1)))
        rnn.append(L * 2 * d_k * d_v)
        n_segs = lambda i: i // seg_size
        mc_cost = sum(n_segs(i) * d_k * d_v for i in range(1, L + 1))
        mc_cost += (L // seg_size) * seg_size * d_k * d_v   # segment-build cost
        mc.append(mc_cost)
    return np.array(attn), np.array(rnn), np.array(mc)


def run():
    results = {"attention": [], "linear_rnn": [], f"mc_seg{seg_size}": []}
    for L in lengths:
        K, V = make_kv_pairs(L, d_k, d_v, seed=L)
        t_attn = time_attention(K, V)
        t_rnn = time_linear_rnn(K, V)
        t_mc = time_mc(K, V, seg_size)
        results["attention"].append(t_attn)
        results["linear_rnn"].append(t_rnn)
        results[f"mc_seg{seg_size}"].append(t_mc)
        print(f"L={L:5d}  attention={t_attn:.4f}s  linear_rnn={t_rnn:.4f}s  mc={t_mc:.4f}s")

    flops_attn, flops_rnn, flops_mc = analytical_flops(lengths, d_k, d_v, seg_size)
    return lengths, results, (flops_attn, flops_rnn, flops_mc)


if __name__ == "__main__":
    lengths, results, flops = run()
    np.savez(RESULTS_DIR / "timing_results.npz", lengths=lengths,
             flops_attention=flops[0], flops_linear_rnn=flops[1], flops_mc=flops[2],
             **results)
