import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    """
    Generates polynomial features up to a given degree.
    Example:
        X = [[2, 3]]
        degree=2 → [[1, 2, 3, 4, 6, 9]]
    """

    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.powers_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[1]
        self.n_input_features_ = n_features

        # Generate all combinations of feature indices with replacement
        combs = [
            c
            for d in range(0 if self.include_bias else 1, self.degree + 1)
            for c in combinations_with_replacement(range(n_features), d)
        ]
        self.powers_ = combs  # list, not np.array
        self.n_output_features_ = len(combs)
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, self.n_output_features_))
        for i, comb in enumerate(self.powers_):
            if len(comb) > 0:
                X_poly[:, i] = np.prod(X[:, comb], axis=1)
        return X_poly

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

