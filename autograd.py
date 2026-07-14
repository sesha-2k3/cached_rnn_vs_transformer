"""
A minimal reverse-mode autograd engine, built from scratch.

Design choice: tensors here are UN-batched (plain vectors/matrices per
example). We loop over a batch in Python and accumulate gradients. This
trades raw speed for simplicity/correctness, which matters more since we
have no reference implementation to check against.
"""
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "name")

    def __init__(self, data, _prev=(), name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.name = name

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def backward(self, seed=None):
        topo, visited = [], set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    build(p)
                topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data) if seed is None else np.asarray(seed, dtype=np.float64)
        for t in reversed(topo):
            t._backward()

    # operations
    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), "add")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def matvec(self, x):
        """ self: (out,in) matrix,  x: (in,) vector  ->  (out,) vector """
        out = Tensor(self.data @ x.data, (self, x), "matvec")
        def _backward():
            self.grad += np.outer(out.grad, x.data)
            x.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def vecmat(self, M):
        """ self: (N,) vector,  M: (N,d) matrix  ->  (d,) vector : self @ M """
        out = Tensor(self.data @ M.data, (self, M), "vecmat")
        def _backward():
            self.grad += M.data @ out.grad
            M.grad += np.outer(self.data, out.grad)
        out._backward = _backward
        return out

    def outer(self, other):
        """ self: (m,) , other: (n,)  ->  (m,n) = outer(self, other) """
        out = Tensor(np.outer(self.data, other.data), (self, other), "outer")
        def _backward():
            self.grad += out.grad @ other.data
            other.grad += out.grad.T @ self.data
        out._backward = _backward
        return out

    def dot(self, other):
        out = Tensor(np.dot(self.data, other.data), (self, other), "dot")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def scale(self, c):
        """ multiply by a python float constant (no grad wrt c) """
        out = Tensor(self.data * c, (self,), "scale")
        def _backward():
            self.grad += out.grad * c
        out._backward = _backward
        return out

    def softmax(self):
        x = self.data - self.data.max()
        e = np.exp(x)
        p = e / e.sum()
        out = Tensor(p, (self,), "softmax")
        def _backward():
            g = out.grad
            self.grad += p * (g - np.dot(p, g))
        out._backward = _backward
        return out

    def cross_entropy(self, target_idx):
        """ self: logits (C,), target_idx: int  ->  scalar loss """
        x = self.data - self.data.max()
        e = np.exp(x)
        p = e / e.sum()
        loss = -np.log(p[target_idx] + 1e-12)
        out = Tensor(loss, (self,), "xent")
        def _backward():
            g = p.copy()
            g[target_idx] -= 1.0
            self.grad += g * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor({self.data.shape}, name={self.name})"


def embed(table: Tensor, idx: int):
    """ Row lookup with correct scatter-add gradient into the table. """
    out = Tensor(table.data[idx].copy(), (table,), "embed")
    def _backward():
        table.grad[idx] += out.grad
    out._backward = _backward
    return out


def stack(scalars):
    """ list of N scalar Tensors -> one (N,) Tensor, gradient scattered back """
    data = np.array([s.data for s in scalars])
    out = Tensor(data, tuple(scalars), "stack")
    def _backward():
        for i, s in enumerate(scalars):
            s.grad += out.grad[i]
    out._backward = _backward
    return out


def stack_vecmat(weights: Tensor, vectors):
    """
    weights: (N,) tensor of scalars,  vectors: list of N Tensors each (d,)
    returns y = sum_i weights[i] * vectors[i]     (the GRM aggregation, Eq. 9)
    """
    d = vectors[0].data.shape[0]
    N = len(vectors)
    y_data = np.zeros(d)
    for i in range(N):
        y_data += weights.data[i] * vectors[i].data
    out = Tensor(y_data, (weights, *vectors), "gated_sum")
    def _backward():
        for i in range(N):
            weights.grad[i] += np.dot(out.grad, vectors[i].data)
            vectors[i].grad += weights.data[i] * out.grad
    out._backward = _backward
    return out
