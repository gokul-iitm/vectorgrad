"""
tests/test_gradcheck.py

Validates nanograd's analytical gradients two ways:
  1. Against PyTorch autograd (ground truth, if torch is installed)
  2. Against numerical (finite-difference) gradients (always works, no deps)

Sri Mathe Ramanujaya Namaha
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanograd.engine import Tensor


def numerical_grad(f, x: np.ndarray, eps=1e-5):
    """Central-difference numerical gradient of scalar-valued f w.r.t. array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        f_plus = f(x)

        x[idx] = orig - eps
        f_minus = f(x)

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_matmul_gradcheck():
    np.random.seed(0)
    X_np = np.random.randn(4, 3)
    W_np = np.random.randn(3, 2)

    def f(X_val):
        return (X_val @ W_np).sum()

    numeric = numerical_grad(f, X_np.copy())

    X = Tensor(X_np.copy())
    W = Tensor(W_np.copy())
    out = X.matmul(W).sum()
    out.backward()

    assert np.allclose(X.grad, numeric, atol=1e-4), \
        f"matmul grad mismatch:\nanalytic={X.grad}\nnumeric={numeric}"
    print("test_matmul_gradcheck: PASSED")


def test_broadcasting_add_gradcheck():
    np.random.seed(1)
    X_np = np.random.randn(5, 4)
    b_np = np.random.randn(4)

    def f(b_val):
        return (X_np + b_val).sum()

    numeric = numerical_grad(f, b_np.copy())

    X = Tensor(X_np.copy())
    b = Tensor(b_np.copy())
    out = (X + b).sum()
    out.backward()

    assert np.allclose(b.grad, numeric, atol=1e-4), \
        f"broadcast-add grad mismatch:\nanalytic={b.grad}\nnumeric={numeric}"
    print("test_broadcasting_add_gradcheck: PASSED")


def test_relu_tanh_gradcheck():
    np.random.seed(2)
    x_np = np.random.randn(3, 3)

    def f(x_val):
        return np.tanh(np.maximum(0, x_val)).sum()

    numeric = numerical_grad(f, x_np.copy())

    x = Tensor(x_np.copy())
    out = x.relu().tanh().sum()
    out.backward()

    assert np.allclose(x.grad, numeric, atol=1e-4), \
        f"relu+tanh grad mismatch:\nanalytic={x.grad}\nnumeric={numeric}"
    print("test_relu_tanh_gradcheck: PASSED")


def test_mlp_forward_backward_shapes():
    from nanograd.nn import MLP, mse_loss

    np.random.seed(3)
    model = MLP([3, 8, 8, 1])
    x = Tensor(np.random.randn(5, 3))
    y = Tensor(np.random.randn(5, 1))

    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()

    for p in model.parameters():
        assert p.grad.shape == p.data.shape
        assert not np.allclose(p.grad, 0), "grad should not be all zero after backward"
    print("test_mlp_forward_backward_shapes: PASSED")


def test_against_pytorch():
    try:
        import torch
    except ImportError:
        print("test_against_pytorch: SKIPPED (torch not installed)")
        return

    np.random.seed(4)
    X_np = np.random.randn(6, 5)
    W_np = np.random.randn(5, 3)
    b_np = np.random.randn(3)

    # nanograd
    X = Tensor(X_np.copy())
    W = Tensor(W_np.copy())
    b = Tensor(b_np.copy())
    out = (X.matmul(W) + b).relu().sum()
    out.backward()

    # torch
    Xt = torch.tensor(X_np, requires_grad=True)
    Wt = torch.tensor(W_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    outt = torch.relu(Xt @ Wt + bt).sum()
    outt.backward()

    assert np.allclose(X.grad, Xt.grad.numpy(), atol=1e-6)
    assert np.allclose(W.grad, Wt.grad.numpy(), atol=1e-6)
    assert np.allclose(b.grad, bt.grad.numpy(), atol=1e-6)
    print("test_against_pytorch: PASSED")


if __name__ == "__main__":
    test_matmul_gradcheck()
    test_broadcasting_add_gradcheck()
    test_relu_tanh_gradcheck()
    test_mlp_forward_backward_shapes()
    test_against_pytorch()
    print("\nAll tests passed.")
