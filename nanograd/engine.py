"""
nanograd/engine.py

A tensor-valued autograd engine, in the spirit of Karpathy's micrograd
but operating on NumPy arrays instead of scalars.

Sri Mathe Ramanujaya Namaha
"""

import numpy as np


def unbroadcast(grad, shape):
    """
    Sum out the axes that were introduced by NumPy broadcasting so that
    `grad` matches `shape` exactly.

    Why this is correct: if a tensor of shape `shape` was broadcast to a
    larger shape during the forward pass, that means the *same* value was
    reused across multiple positions. By the multivariate chain rule, when
    a value is reused in several places, the gradients from every place it
    was used simply add up. Broadcasting is implicit reuse, so "undoing"
    a broadcast in the backward pass is exactly a sum over the axes that
    were stretched.
    """
    # Step 1: remove any extra leading dimensions NumPy added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Step 2: for dims that were size-1 and got stretched, sum them back down
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """
    Stores a NumPy array and its gradient. Tracks the small DAG of
    Tensors that produced it so `.backward()` can walk it in reverse.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

        # internal bookkeeping for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for debugging / visualization

    # ------------------------------------------------------------------
    # Core ops. Each op does two things:
    #   1. Forward: compute out.data
    #   2. Define out._backward: given out.grad, accumulate into the
    #      .grad of every input tensor via the local derivative and the
    #      chain rule.
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1 -> just route grad through,
            # unbroadcasting back to original shapes if broadcasting happened.
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # d(out)/d(self) = other, d(out)/d(other) = self  (elementwise)
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        assert isinstance(other, Tensor)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # Y = X @ W  =>  dL/dX = dL/dY @ W.T ,  dL/dW = X.T @ dL/dY
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    __matmul__ = matmul

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Tensor(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            # broadcasting an all-ones array of self's shape routes the
            # gradient back to every element that contributed to the sum
            self.grad += grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------
    # Convenience operators built from the primitives above
    # ------------------------------------------------------------------

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    # ------------------------------------------------------------------
    # The actual backprop: topological sort, then apply chain rule
    # in reverse order.
    # ------------------------------------------------------------------

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # seed gradient: dL/dL = 1
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
