import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from autograd import Tensor, embed, stack_vecmat

rng = np.random.default_rng(42)


def numgrad(f, t: Tensor, eps=1e-5):
    g = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = t.data[idx]
        t.data[idx] = orig + eps
        f1 = f()
        t.data[idx] = orig - eps
        f2 = f()
        t.data[idx] = orig
        g[idx] = (f1 - f2) / (2 * eps)
    return g


def check(name, build_fn, tensors):
    out = build_fn()
    out.backward()
    analytic = {n: t.grad.copy() for n, t in tensors}   # snapshot BEFORE numgrad runs
    ok = True
    for name_t, t in tensors:
        def scalar_f():
            o = build_fn()
            return o.data if o.data.ndim == 0 else o.data.sum()
        ng = numgrad(scalar_f, t)
        err = np.abs(ng - analytic[name_t]).max()
        status = "OK" if err < 1e-4 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{name}] grad wrt {name_t}: max_err={err:.2e}  {status}")
    return ok


all_ok = True

# matvec
W = Tensor(rng.normal(size=(3, 4)))
x = Tensor(rng.normal(size=4))
def f1():
    W.zero_grad(); x.zero_grad()
    return W.matvec(x)
all_ok &= check("matvec", f1, [("W", W), ("x", x)])

# vecmat
w = Tensor(rng.normal(size=5))
M = Tensor(rng.normal(size=(5, 3)))
def f2():
    w.zero_grad(); M.zero_grad()
    return w.vecmat(M)
all_ok &= check("vecmat", f2, [("w", w), ("M", M)])

# outer
a = Tensor(rng.normal(size=3))
b = Tensor(rng.normal(size=4))
def f3():
    a.zero_grad(); b.zero_grad()
    return a.outer(b)
all_ok &= check("outer", f3, [("a", a), ("b", b)])

# dot
p = Tensor(rng.normal(size=6))
q = Tensor(rng.normal(size=6))
def f4():
    p.zero_grad(); q.zero_grad()
    return p.dot(q)
all_ok &= check("dot", f4, [("p", p), ("q", q)])

# softmax (sum of outputs after a nonlinear reweighting so grad isn't trivially 0)
s = Tensor(rng.normal(size=5))
target_vec = Tensor(rng.normal(size=5))
def f5():
    s.zero_grad()
    probs = s.softmax()
    return probs.dot(target_vec)
all_ok &= check("softmax(dot)", f5, [("s", s)])

# cross_entropy
logits = Tensor(rng.normal(size=7))
def f6():
    logits.zero_grad()
    return logits.cross_entropy(3)
all_ok &= check("cross_entropy", f6, [("logits", logits)])

# embed
table = Tensor(rng.normal(size=(10, 4)))
def f7():
    table.zero_grad()
    e = embed(table, 5)
    return e.dot(Tensor(np.ones(4)))
all_ok &= check("embed", f7, [("table", table)])

# stack_vecmat (GRM gated aggregation)
weights = Tensor(rng.normal(size=4))
vecs = [Tensor(rng.normal(size=3)) for _ in range(4)]
def f8():
    weights.zero_grad()
    for v in vecs:
        v.zero_grad()
    y = stack_vecmat(weights, vecs)
    return y.dot(Tensor(np.ones(3)))
all_ok &= check("stack_vecmat", f8, [("weights", weights)] + [(f"vec{i}", v) for i, v in enumerate(vecs)])

print("\nALL GRADIENT CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
