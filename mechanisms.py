"""
Three memory mechanisms, implemented directly from their defining equations:

1. Causal softmax attention (Transformer) - O(L^2) total
2. Linear-attention RNN (fixed-size memory) - O(L) total
3. Memory Caching w/ GRM gating (segmented, cached) - O(N*L) total

We test them on an associative-recall task: store L random (key, value) pairs,
then query with one of the stored keys and see if the mechanism can still
retrieve the correct value. This directly probes the "fixed memory forces
forgetting" phenomenon the paper is about.
"""
import numpy as np

rng = np.random.default_rng(0)


def make_kv_pairs(L, d_k, d_v, seed=0):
    """L random unit-norm keys, L random one-hot-ish values (for easy scoring)."""
    r = np.random.default_rng(seed)
    K = r.normal(size=(L, d_k))
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    # values: random vectors, but we'll score retrieval via cosine similarity
    V = r.normal(size=(L, d_v))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return K, V


# 1. Transformer: exact causal/full softmax attention

def attention_retrieve(K, V, q, sharpness=20.0):
    """
    y = sum_i softmax(sharpness * q.k_i) v_i  -- attends to ALL L stored pairs
    individually. `sharpness` stands in for what trained W_Q/W_K projections
    do in a real model: push the true match's score well above the noise
    floor from unrelated keys. We calibrate it once from raw dot-product
    statistics rather than dividing by sqrt(d_k) on already unit-norm keys,
    which would UNDER-separate them (see README for the calibration numbers).
    """
    scores = sharpness * (K @ q)            # O(L * d_k)
    scores -= scores.max()
    w = np.exp(scores)
    w /= w.sum()
    return w @ V                            # O(L * d_v)


# 2. Plain linear-attention RNN: single fixed-size matrix memory

def linear_rnn_memory(K, V):
    """M = sum_i v_i k_i^T   (d_v x d_k), built once over all L pairs."""
    return V.T @ K                          # O(L * d_k * d_v) to build, O(1) states


def linear_rnn_retrieve(M, q):
    return M @ q                            # O(d_k * d_v) per query


# 3. Memory Caching (GRM variant): segmented + cached + gated

def build_segment_memories(K, V, seg_size):
    """Split into N segments of size C, one fixed-size memory matrix per segment."""
    L = K.shape[0]
    segs_K = [K[i:i + seg_size] for i in range(0, L, seg_size)]
    segs_V = [V[i:i + seg_size] for i in range(0, L, seg_size)]
    mems = [Vs.T @ Ks for Ks, Vs in zip(segs_K, segs_V)]          # list of (d_v x d_k)
    mean_keys = [Ks.mean(axis=0) for Ks in segs_K]                 # for gating, Eq. 10
    return mems, mean_keys


def mc_retrieve(mems, mean_keys, q, u=None, sharpness=40.0):
    """
    y = sum_s gamma_s * M^(s) q ,  gamma = softmax_s <u, mean_key_s>   (Eq. 9-10)
    Cost: O(N) memory forward passes instead of O(1) (RNN) or O(L) (attention).

    `sharpness` again stands in for what a *trained* gate (u = x_t W_u) would
    learn: the raw mean-pooled-key signal is real but weak (it's an average
    over C keys), so a trained projection would rescale it to be decisive.
    """
    u = q if u is None else u
    scores = sharpness * np.array([u @ mk for mk in mean_keys])
    scores -= scores.max()
    gamma = np.exp(scores)
    gamma /= gamma.sum()
    out = np.zeros(mems[0].shape[0])
    for g, M in zip(gamma, mems):
        out += g * (M @ q)
    return out
