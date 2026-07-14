import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the dataset into training and test sets.

    Parameters
    ----------
    X : np.ndarray
        Features array of shape (n_samples, n_features)
    y : np.ndarray
        Labels array of shape (n_samples,)
    test_size : float, default=0.2
        Proportion of data to include in the test split.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_val_split(X, y, train_size=0.7, val_size=0.1, test_size=0.2, shuffle=True, random_state=None):
    """
    Splits the dataset into train, validation, and test sets.

    Parameters
    ----------
    X : np.ndarray
        Features array of shape (n_samples, n_features)
    y : np.ndarray
        Labels array of shape (n_samples,)
    train_size, val_size, test_size : float
        Proportions that must sum to 1.0
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx]
    )
