"""
nanograd/nn.py

Minimal neural net layers built on top of the Tensor autograd engine.

Sri Mathe Ramanujaya Namaha
"""

import numpy as np
from .engine import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """y = x @ W + b"""

    def __init__(self, in_features, out_features):
        # Kaiming-ish init, nothing fancy
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor) -> Tensor:
        return x.matmul(self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP(Module):
    """Simple feed-forward network: Linear -> ReLU -> ... -> Linear"""

    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()
