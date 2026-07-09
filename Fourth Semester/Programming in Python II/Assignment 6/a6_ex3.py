# a6_ex3.py
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler   # Convenience for normalization


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def _split_indices(n: int) -> Tuple[slice, slice, slice]:
    """
    Return three slices that split n into an 80/10/10 ratio.
    Use a deterministic cut without shuffling to preserve temporal order;
    shuffling only happens in the training DataLoader.
    """
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


# -----------------------------------------------------------------------------
# Main function required by the exercise
# -----------------------------------------------------------------------------
def get_data_loaders(
    df: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convert the preprocessed DataFrame into train/validation/test DataLoaders.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from exercise 1. It must contain the 'PM2.5' column,
        which will be used as the target.
    batch_size : int, default 32
        Mini-batch size.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        The three loaders in the order (train_loader, val_loader, test_loader).
    """
    # ---------------------------------------------------------------------
    # 1) Split X (numeric features) and y (target).
    # ---------------------------------------------------------------------
    y = df["PM2.5"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Take all numeric columns except the target.
    feature_cols = (
        df.drop(columns=["PM2.5"])
          .select_dtypes(include=["number"])
          .columns
    )
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------------------
    # 2) Split 80/10/10 without shuffling to preserve temporal order.
    # ---------------------------------------------------------------------
    train_sl, val_sl, test_sl = _split_indices(len(df))

    X_train, y_train = X[train_sl], y[train_sl]
    X_val,   y_val   = X[val_sl],   y[val_sl]
    X_test,  y_test  = X[test_sl],  y[test_sl]

    # ---------------------------------------------------------------------
    # 3) Normalize features, fitting only on the training split.
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save to disk for reuse in exercises 4, 5, and 7.
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ---------------------------------------------------------------------
    # 4) torch TensorDataset + DataLoader
    # ---------------------------------------------------------------------
    def to_loader(X_np: np.ndarray, y_np: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)   # Shuffle training only
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_test,  shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Quick usage example, useful during development
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Assume the CSV created in exercise 1 is in the current working directory.
    df = (
        pd.read_csv("air_quality_cleaned.csv", parse_dates=["datetime"])
          .set_index("datetime")
    )
    tl, vl, tsl = get_data_loaders(df)
    print(
        f"Train batches: {len(tl)} | Val batches: {len(vl)} | "
        f"Test batches: {len(tsl)}"
    )
