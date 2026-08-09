"""
examples/train_regression.py

Trains a nanograd MLP to fit y = sin(x1) + x2^2 - x3, a nonlinear
function, purely to prove the whole stack (Tensor -> MLP -> Adam)
actually learns something.

Sri Mathe Ramanujaya Namaha
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nanograd.engine import Tensor
from nanograd.nn import MLP, mse_loss
from nanograd.optim import Adam


def make_data(n=500, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - X[:, 2]
    return X, y.reshape(-1, 1)


def main():
    X_train, y_train = make_data(n=400, seed=0)
    X_val, y_val = make_data(n=100, seed=1)

    model = MLP([3, 32, 32, 1])
    opt = Adam(model.parameters(), lr=0.01)

    X_t = Tensor(X_train)
    y_t = Tensor(y_train)
    X_v = Tensor(X_val)
    y_v = Tensor(y_val)

    print(f"{'epoch':>6} | {'train_loss':>10} | {'val_loss':>10}")
    print("-" * 34)

    for epoch in range(1, 301):
        opt.zero_grad()
        pred = model(X_t)
        loss = mse_loss(pred, y_t)
        loss.backward()
        opt.step()

        if epoch % 30 == 0 or epoch == 1:
            val_pred = model(X_v)
            val_loss = mse_loss(val_pred, y_v)
            print(f"{epoch:>6} | {loss.data:>10.5f} | {val_loss.data:>10.5f}")

    print("\nDone. Loss should have dropped substantially from epoch 1 -> 300,")
    print("showing that Tensor autograd + Linear/ReLU layers + Adam all wired")
    print("correctly end-to-end.")


if __name__ == "__main__":
    main()
