# a6_ex3.py
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler   # ← comodità per il normalising


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def _split_indices(n: int) -> Tuple[slice, slice, slice]:
    """
    Restituisce tre slice che partizionano  n  in ratio 80/10/10.
    Usa un cut “deterministico” (no shuffle qui) per mantenere sequenzialità
    – lo shuffle avverrà soltanto nel DataLoader di training.
    """
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


# -----------------------------------------------------------------------------
# Funzione principale richiesta dall’esercizio
# -----------------------------------------------------------------------------
def get_data_loaders(
    df: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Converte il DataFrame preprocessato in tre DataLoader (train/val/test).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset già pulito (output exercise 1).  Deve contenere la colonna
        'PM2.5', che sarà usata come target.
    batch_size : int, default 32
        Dimensione dei mini-batch.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        I tre loader nell’ordine (train_loader, val_loader, test_loader).
    """
    # ---------------------------------------------------------------------
    # 1) separa X (feature numeriche) e y (target)
    # ---------------------------------------------------------------------
    y = df["PM2.5"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # prendi tutte le colonne numeriche tranne il target
    feature_cols = (
        df.drop(columns=["PM2.5"])
          .select_dtypes(include=["number"])
          .columns
    )
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------------------
    # 2) split 80/10/10 senza shuffle per preservare ordine temporale
    # ---------------------------------------------------------------------
    train_sl, val_sl, test_sl = _split_indices(len(df))

    X_train, y_train = X[train_sl], y[train_sl]
    X_val,   y_val   = X[val_sl],   y[val_sl]
    X_test,  y_test  = X[test_sl],  y[test_sl]

    # ---------------------------------------------------------------------
    # 3) normalizzazione (fit SOLO su training)
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # salva a disco per ri-uso in Exercise 4-5-7
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

    train_loader = to_loader(X_train, y_train, shuffle=True)   # shuffle SOLO train
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_test,  shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Esempio rapido d’uso (non richiesto ma comodo in sviluppo)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # assume il CSV creato nell’esercizio 1 si trovi nella cwd
    df = (
        pd.read_csv("air_quality_cleaned.csv", parse_dates=["datetime"])
          .set_index("datetime")
    )
    tl, vl, tsl = get_data_loaders(df)
    print(
        f"Train batches: {len(tl)} | Val batches: {len(vl)} | "
        f"Test batches: {len(tsl)}"
    )
