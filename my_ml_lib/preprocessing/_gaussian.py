import numpy as np

class GaussianBasisFeatures:
    """
    Expands input features using Gaussian (RBF) basis functions.

    Example:
        gbf = GaussianBasisFeatures(n_centers=3, sigma=1.0)
        gbf.fit(X_train)
        X_gbf = gbf.transform(X_test)
    """

    def __init__(self, n_centers=10, sigma=1.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.random_state = random_state
        self.centers_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)

        # Randomly choose n_centers samples from X as RBF centers
        idx = rng.choice(X.shape[0], self.n_centers, replace=False)
        self.centers_ = X[idx]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.centers_ is None:
            raise ValueError("You must call fit() before transform().")

        # Compute squared Euclidean distance from each point to each center
        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        return np.exp(-0.5 * (dists / self.sigma) ** 2)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
