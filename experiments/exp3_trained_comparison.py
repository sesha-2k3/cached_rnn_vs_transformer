"""
Train both models on the same associative-recall episodes with real
backprop (replacing the hand-set "sharpness" scalar from before with
weights the model actually learns), then compare recall accuracy across
sequence lengths -- including lengths LONGER than anything seen in training,
to test generalization/extrapolation of the retrieval mechanism itself.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import time
import numpy as np
from models import init_params, attention_forward, mc_forward, all_params
from optim import Adam

rng = np.random.default_rng(0)

Nk_vocab = 300   # pool of possible key identities (must cover max L we test)
Nv = 32          # number of distinct value classes
d_k, d_v = 32, 32
SEG_SIZE = 16
TRAIN_LENGTHS = [16, 32, 64]     # curriculum: train on these lengths only
EVAL_LENGTHS = [8, 16, 32, 64, 128, 256]   # includes 2x-4x beyond training!


def make_episode(L, rng):
    key_ids = rng.choice(Nk_vocab, size=L, replace=False)
    value_ids = rng.integers(0, Nv, size=L)
    query_pos = int(rng.integers(0, L))
    return key_ids, value_ids, query_pos


def run_episode_attn(params, ep):
    key_ids, value_ids, query_pos = ep
    logits = attention_forward(params, key_ids, value_ids, query_pos)
    target = int(value_ids[query_pos])
    loss = logits.cross_entropy(target)
    pred = int(np.argmax(logits.data))
    return loss, pred == target


def run_episode_mc(params, ep, seg_size=SEG_SIZE):
    key_ids, value_ids, query_pos = ep
    logits = mc_forward(params, key_ids, value_ids, query_pos, seg_size)
    target = int(value_ids[query_pos])
    loss = logits.cross_entropy(target)
    pred = int(np.argmax(logits.data))
    return loss, pred == target


def clip_grad_norm(params, max_norm=5.0):
    total = np.sqrt(sum(np.sum(p.grad ** 2) for p in params))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for p in params:
            p.grad *= scale
    return total


def train(params, run_fn, param_list, steps=400, batch_size=8, lr=3e-2, log_every=25, tag="", clip=5.0):
    opt = Adam(param_list, lr=lr)
    history = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        total_loss, correct = 0.0, 0
        for _ in range(batch_size):
            L = int(rng.choice(TRAIN_LENGTHS))
            ep = make_episode(L, rng)
            loss, ok = run_fn(params, ep)
            loss.backward(seed=1.0 / batch_size)
            total_loss += loss.data / batch_size
            correct += ok
        if clip is not None:
            clip_grad_norm(param_list, clip)
        opt.step()
        history.append((total_loss, correct / batch_size))
        if (step + 1) % log_every == 0:
            print(f"  [{tag}] step {step+1:4d}/{steps}  loss={total_loss:.3f}  "
                  f"batch_acc={correct/batch_size:.2f}  ({time.time()-t0:.1f}s elapsed)")
    return history


def evaluate(params, run_fn, lengths, n_trials=60):
    accs = []
    for L in lengths:
        correct = 0
        for _ in range(n_trials):
            ep = make_episode(L, rng)
            _, ok = run_fn(params, ep)
            correct += ok
        accs.append(correct / n_trials)
        print(f"    L={L:4d}  acc={correct/n_trials:.3f}")
    return accs


if __name__ == "__main__":
    print("=== Training single-layer Attention model ===")
    params = init_params(Nk_vocab, Nv, d_k, d_v, seed=1)
    attn_param_list = [params[k] for k in
                        ["W_K", "W_Q", "W_V", "W_out_attn", "b_out_attn"]]
    hist_attn = train(params, run_episode_attn, attn_param_list,
                       steps=800, batch_size=8, lr=0.05, clip=1.0, tag="attn")

    print("\n=== Training single-layer Memory-Caching (GRM) model ===")
    mc_param_list = [params[k] for k in
                      ["W_k", "W_v", "W_q", "W_out_mc", "b_out_mc"]]
    hist_mc = train(params, run_episode_mc, mc_param_list,
                     steps=1000, batch_size=8, lr=0.02, clip=1.0, tag="mc")

    print("\n=== Evaluating Attention across lengths (incl. beyond training!) ===")
    acc_attn = evaluate(params, run_episode_attn, EVAL_LENGTHS)

    print("\n=== Evaluating Memory-Caching across lengths ===")
    acc_mc = evaluate(params, run_episode_mc, EVAL_LENGTHS)

    np.savez(RESULTS_DIR / "trained_results.npz",
             eval_lengths=EVAL_LENGTHS,
             acc_attn=acc_attn, acc_mc=acc_mc,
             loss_attn=[h[0] for h in hist_attn], acc_hist_attn=[h[1] for h in hist_attn],
             loss_mc=[h[0] for h in hist_mc], acc_hist_mc=[h[1] for h in hist_mc])
    print("\nSaved trained_results.npz")
