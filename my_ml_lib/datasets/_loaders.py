import numpy as np
import pandas as pd
import os
from my_ml_lib.model_selection._split import train_test_split
# ---------------------------------------------------------------------
# 1. Spambase Dataset Loader
# ---------------------------------------------------------------------
def load_spambase(path="data/spambase.data"):
    """
    Loads the UCI Spambase dataset.
    Expects a local CSV or .data file with 57 feature columns + 1 label column.
    Returns (X, y) as NumPy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download 'spambase.data' from UCI ML Repository "
            "and place it under a 'data/' folder."
        )

    data = pd.read_csv(path, header=None)
    X = data.iloc[:, :-1].values  # 57 features
    y = data.iloc[:, -1].values   # labels 0/1
    return X, y


# ---------------------------------------------------------------------
# 2. Fashion-MNIST Dataset Loader
# ---------------------------------------------------------------------

def load_fashion_mnist_three_way(
    train_path="../data/fashion-mnist_train.csv",
    test_path="../data/fashion-mnist_test.csv",
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):
    """
    Load the Fashion-MNIST dataset from CSV files, normalize it to [0,1],
    and split into training, validation, and test sets.

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the testing CSV file.
    val_ratio : float
        Fraction of total data to allocate for validation. Default is 0.15.
    test_ratio : float
        Fraction of total data to allocate for testing. Default is 0.15.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        NumPy arrays containing normalized image data and integer labels.
    """

    # ✅ Check file existence
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"CSV files not found at:\n{train_path}\n{test_path}\n"
            "Please download them from Kaggle or OpenML (Zalando Research)."
        )

    # ✅ Load CSVs
    print("📥 Reading Fashion-MNIST CSV files...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ✅ Combine both datasets (total 70,000 samples)
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"✅ Combined dataset shape: {full_df.shape}")

    # ✅ Separate features and labels
    y_full = full_df.iloc[:, 0].to_numpy(dtype=np.int64)
    X_full = full_df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # ✅ First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True
    )

    # ✅ Second split: validation set (adjusted ratio)
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        shuffle=True
    )

    # ✅ Print summary
    print("✅ Dataset split completed:")
    print(f"  Training set:     {X_train.shape}")
    print(f"  Validation set:   {X_val.shape}")
    print(f"  Test set:         {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

