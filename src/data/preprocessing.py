import os
from typing import Tuple, Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


# ==========================
#  L2 NORMALIZATION FUNCTIONS
# ==========================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 1D vector.
    If the norm is zero, return the vector unchanged.
    """
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a 2D array.
    If a row's norm is zero, that row is left unchanged.
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0  # Avoid division by zero
    return X / norm


# ==========================
#  DATA LOADING & PREPARATION
# ==========================

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the CSV is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    if df.empty:
        raise ValueError("CSV file is empty")
    return df


def prepare_features_and_labels(
    df: pd.DataFrame,
    label_column: str = "diagnosis",
    positive_label: str = "M",
    negative_label: str = "B",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features (X) and labels (y) from the DataFrame.

    - Maps label_column to binary:
        positive_label -> 1
        negative_label -> 0
    - Drops label column from features.
    - Drops 'id' column if present.
    - Keeps only numeric feature columns.
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    # Raw labels
    y_raw = df[label_column]

    # Map labels to 0/1
    y = y_raw.map({positive_label: 1, negative_label: 0}).values
    if np.any(np.isnan(y)):
        raise ValueError("Label column contains NaN values after mapping")

    # Drop label column from features
    feature_df = df.drop(columns=[label_column])

    # Drop 'id' column if it exists
    if "id" in feature_df.columns:
        feature_df = feature_df.drop(columns=["id"])

    # Keep only numeric columns
    feature_df = feature_df.select_dtypes(include=[np.number])

    X = feature_df.values
    return X, y


# ==========================
#  PREPROCESSOR CLASS
# ==========================

class BreastCancerPreprocessor:
    """
    Class for preprocessing the breast cancer dataset.

    - Applies MinMax scaling to [0, 1].
    - Optionally applies L2 normalization per row
      (useful for amplitude encoding in QSVM).
    """

    def __init__(self, apply_l2: bool = False):
        self.scaler: Optional[MinMaxScaler] = None
        self.apply_l2 = apply_l2

    def fit(self, X: np.ndarray) -> None:
        """Fit the scaler to the data."""
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted scaler.
        Applies L2 normalization per row if apply_l2 is True.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted. Call fit() before transform()."
            )

        X_scaled = self.scaler.transform(X)
        if self.apply_l2:
            X_scaled = l2_normalize_rows(X_scaled)
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and transform the data in one step."""
        self.fit(X)
        return self.transform(X)


# ==========================
#  TRAIN/TEST SPLIT & K-FOLD
# ==========================

def get_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    If stratify=True, stratified split is performed based on y.
    """
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test


def get_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate stratified K-Fold splits for cross-validation.

    Yields:
        (X_train, X_test, y_train, y_test) for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test

