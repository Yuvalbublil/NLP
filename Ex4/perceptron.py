from __future__ import annotations

import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron:
    def __init__(self, n_features: int = -1, num_iter: int = 2):
        super().__init__()
        self.num_iter = num_iter
        self.n_features = n_features
        self.coefs_ = None

    def fit(self, X, y):
        weights = np.zeros(self.n_features)
        b = 0

        for t in range(self.num_iter):
            for i, x in enumerate(X):
                if y[i] * x.dot(weights)[0] + b <= 0:
                    weights += (y[i] * x.toarray()).reshape(self.n_features)
                    b += y[i]
                else:
                    self.coefs_ = weights
                    return

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.coefs_)
