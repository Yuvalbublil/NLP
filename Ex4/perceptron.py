from __future__ import annotations

from typing import Callable
from typing import NoReturn

import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron:
    def __init__(self, n_features: int = -1, num_iter: int = 2):
        super().__init__()
        self.num_iter = num_iter
        self.n_features = n_features
        self.coefs_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        weights = np.zeros(self.n_features)
        b = 0

        for t in range(self.max_iter_):
            for i, x in enumerate(X):
                response = y[i] * np.matmul(training_loss_, x)
                self.callback_(self, x, response)
                if response <= 0:
                    training_loss_ += y[i] * x
                else:
                    self.coefs_ = training_loss_
                    return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = self.__intercepted_X(X)
        return np.matmul(X, self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)

    def __intercepted_X(self, X: np.ndarray) -> np.ndarray:
        return np.append(np.zeros((X.shape[0], 1)), X, axis=1)


if __name__ == '__main__':
    p = Perceptron()
    p.fit(np.random.rand(10, 4), np.random.rand(10))
    print(p.coefs_)
