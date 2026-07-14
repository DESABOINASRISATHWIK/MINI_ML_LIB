import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute mean and std for each feature."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Apply standardization."""
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
