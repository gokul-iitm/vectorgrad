# nanograd

A tensor-valued automatic differentiation engine, extending Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) from scalars to NumPy
arrays — with matrix multiplication, broadcasting-aware backprop, and a small
neural net library (`Linear`, `MLP`, `SGD`, `Adam`) built on top.

Every gradient rule is derived by hand in the comments before it's coded, and
verified two ways: finite-difference numerical gradient checking, and
(optionally) cross-checked against PyTorch autograd.

## Why this exists

micrograd is scalar-only: every number in the network is its own graph node.
That's great for teaching backprop, but it means matrix operations don't
exist — you can't express `Y = XW` directly, and broadcasting (e.g. adding a
bias vector to every row of a batch) has no defined gradient rule at all.

nanograd fills that gap: same DAG-based reverse-mode autodiff design as
micrograd, but every `Tensor` wraps a NumPy array, and the two ops that
matter most for real networks — `matmul` and broadcasting — are implemented
with their backward passes derived from first principles below.

## The core math

### Matrix multiply

For `Y = X @ W` with `X: (m,n)`, `W: (n,p)`, `Y: (m,p)`:

```
dL/dX = dL/dY @ W.T      shape: (m,p)(p,n) = (m,n)  ✓ matches X
dL/dW = X.T @ dL/dY       shape: (n,m)(m,p) = (n,p)  ✓ matches W
```

Derivation: `Y_ij = sum_k X_ik * W_kj`, so `dY_ij/dX_ik = W_kj`. Chain rule
gives `dL/dX_ik = sum_j dL/dY_ij * W_kj`, which is exactly `(dL/dY) @ W.T`
written in matrix form. Same argument for `W`.

### Broadcasting

When a smaller tensor (e.g. a bias vector) is broadcast across a larger one
during the forward pass, that means the *same* value was reused at every
broadcast position. By the multivariate chain rule, gradients from every
place a value was reused simply **sum**. So the backward pass for
broadcasting is: sum the incoming gradient over exactly the axes that were
stretched, then reshape back to the original tensor's shape. Implemented in
`unbroadcast()` in `engine.py`.

This is the part of tensor-autograd that's most commonly implemented wrong
(or skipped by copying an existing framework without understanding it) —
here it's derived and unit tested explicitly against finite-difference
gradients.

## Structure

```
nanograd/
  engine.py     Tensor class: forward ops + hand-derived backward passes
  nn.py         Linear, MLP, mse_loss
  optim.py      SGD (with momentum), Adam
tests/
  test_gradcheck.py   Numerical + PyTorch gradient checking
examples/
  train_regression.py  End-to-end training demo on a nonlinear function
```

## Usage

```python
from nanograd import Tensor
from nanograd.nn import MLP, mse_loss
from nanograd.optim import Adam

model = MLP([3, 32, 32, 1])
opt = Adam(model.parameters(), lr=0.01)

x = Tensor(X_train)   # (N, 3)
y = Tensor(y_train)   # (N, 1)

for epoch in range(300):
    opt.zero_grad()
    loss = mse_loss(model(x), y)
    loss.backward()
    opt.step()
```

Run the demo:

```bash
python examples/train_regression.py
```

Run gradient checks:

```bash
python tests/test_gradcheck.py
```

## What's verified

- `matmul` backward checked against finite-difference gradients
- Broadcasting backward (`X + b` with `b` broadcast across batch) checked
  against finite-difference gradients
- `relu` + `tanh` composition checked against finite-difference gradients
- Full MLP forward/backward produces correctly-shaped, non-zero gradients
  for every parameter
- End-to-end training on a nonlinear regression task converges (loss drops
  ~600x over 300 epochs)
- Optional: cross-checked against PyTorch autograd if `torch` is installed

## What this is not

This is not a fast or production framework — no GPU support, no autograd
graph pruning, no lazy evaluation. The point is correctness and clarity of
the underlying math, not performance. Every op is written so the backward
pass is a direct, readable transcription of a hand-derived chain rule.
