"""
Two trainable, single-layer models on the same associative-recall task:

  Attention model:  logits = W_out · softmax(q·K_i) over ALL i, weighted V_i
  MC (GRM) model:   logits = W_out · sum_s gamma_s * (M^(s) @ q)
                    where M^(s) = sum_{i in segment s} outer(v_i, k_i)  (cached)
                    and   gamma = softmax_s <q, mean_key_s>            (u = q)

Both use the SAME embedding sizes so the comparison is apples-to-apples;
the only difference is the retrieval mechanism (full attention vs. cached
segment memories), exactly mirroring the architectural distinction discussed
in the paper.
"""
import numpy as np
from autograd import Tensor, embed, stack, stack_vecmat


def init_params(Nk_vocab, Nv, d_k, d_v, seed=0):
    r = np.random.default_rng(seed)
    def emb(n, d):
        return Tensor(r.normal(scale=1.0 / np.sqrt(d), size=(n, d)))
    return {
        # attention params
        "W_K": emb(Nk_vocab, d_k), "W_Q": emb(Nk_vocab, d_k), "W_V": emb(Nv, d_v),
        "W_out_attn": Tensor(r.normal(scale=1.0 / np.sqrt(d_v), size=(Nv, d_v))),
        "b_out_attn": Tensor(np.zeros(Nv)),
        # MC (GRM) params -- separate embedding tables, same sizes
        "W_k": emb(Nk_vocab, d_k), "W_v": emb(Nv, d_v), "W_q": emb(Nk_vocab, d_k),
        "W_out_mc": Tensor(r.normal(scale=1.0 / np.sqrt(d_v), size=(Nv, d_v))),
        "b_out_mc": Tensor(np.zeros(Nv)),
    }


def attention_forward(params, key_ids, value_ids, query_pos):
    Ks = [embed(params["W_K"], k) for k in key_ids]
    Vs = [embed(params["W_V"], v) for v in value_ids]
    q = embed(params["W_Q"], key_ids[query_pos])
    d_k = q.data.shape[0]
    inv_sqrt_dk = 1.0 / np.sqrt(d_k)
    scores = stack([q.dot(k).scale(inv_sqrt_dk) for k in Ks])
    attn = scores.softmax()
    y = stack_vecmat(attn, Vs)
    logits = params["W_out_attn"].matvec(y) + params["b_out_attn"]
    return logits


def mc_forward(params, key_ids, value_ids, query_pos, seg_size):
    Ks = [embed(params["W_k"], k) for k in key_ids]
    Vs = [embed(params["W_v"], v) for v in value_ids]
    q = embed(params["W_q"], key_ids[query_pos])
    d_k = q.data.shape[0]
    inv_sqrt_dk = 1.0 / np.sqrt(d_k)

    L = len(key_ids)
    seg_mems, seg_mean_keys = [], []
    for start in range(0, L, seg_size):
        end = min(start + seg_size, L)
        seg_k, seg_v = Ks[start:end], Vs[start:end]
        d_v, d_k_ = seg_v[0].data.shape[0], seg_k[0].data.shape[0]
        M = Tensor(np.zeros((d_v, d_k_)))
        for k_i, v_i in zip(seg_k, seg_v):
            M = M + v_i.outer(k_i)
        M = M.scale(1.0 / len(seg_k))          # normalize like a mean, not a raw sum
        mean_k = seg_k[0]
        for k_i in seg_k[1:]:
            mean_k = mean_k + k_i
        mean_k = mean_k.scale(1.0 / len(seg_k))
        seg_mems.append(M)
        seg_mean_keys.append(mean_k)

    gate_scores = stack([q.dot(mk).scale(inv_sqrt_dk) for mk in seg_mean_keys])
    gamma = gate_scores.softmax()
    retrieved = [M.matvec(q) for M in seg_mems]
    y = stack_vecmat(gamma, retrieved)
    logits = params["W_out_mc"].matvec(y) + params["b_out_mc"]
    return logits


def all_params(params):
    return list(params.values())
