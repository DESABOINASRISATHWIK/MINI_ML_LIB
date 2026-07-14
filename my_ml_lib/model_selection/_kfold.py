    import numpy as np

class KFold:
    """
    Simple implementation of K-Fold cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int or None, default=None
        Random seed for reproducible shuffling.

    Example
    -------
    >>> X = np.arange(10)
    >>> kf = KFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, val_idx in kf.split(X):
    ...     print("Train:", train_idx, "Val:", val_idx)
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        X = np.asarray(X)
        n_samples = len(X)

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop
