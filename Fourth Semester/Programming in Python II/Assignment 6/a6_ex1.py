# a6_ex1.py
from __future__ import annotations
import io
import os
import zipfile
from typing import List

import pandas as pd
import numpy as np
import urllib.request


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
DATA_URL = (
    # link diretto al file .zip indicato nella pagina UCI
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)


def _download_if_needed(zip_path: str, url: str = DATA_URL) -> None:
    """Scarica la risorsa remota solo se non esiste già in locale."""
    if os.path.exists(zip_path):
        return
    print(f"Downloading dataset to «{zip_path}» …")
    urllib.request.urlretrieve(url, zip_path)


# --------------------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------------------
def preprocess_data(zip_path: str, station: str) -> pd.DataFrame:
    """
    Estrae, pulisce e arricchisce i dati della stazione indicata.

    Parameters
    ----------
    zip_path : str
        Percorso locale del file «beijing+multi+site+air+quality+data.zip».
    station : str
        Nome della stazione – es. 'Aotizhongxin'.

    Returns
    -------
    pd.DataFrame
        DataFrame con indice datetime, pronto per l’EDA / modellazione.
    """
    # 1) download se serve
    _download_if_needed(zip_path)

    # 2) apri lo zip esterno
    with zipfile.ZipFile(zip_path) as outer_zip:
        # all’interno c’è esattamente un altro zip
        inner_name: str = next(
            name for name in outer_zip.namelist() if name.endswith(".zip")
        )
        inner_bytes = outer_zip.read(inner_name)

    # 3) apri lo zip interno direttamente da memoria
        # ---------------------------------------------------------------------------
    # 3) apri lo zip interno, individua il CSV della stazione e caricalo          #
    # ---------------------------------------------------------------------------
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        core_name = f"PRSA_Data_{station}_20130301-20170228.csv"
        matches = [name for name in inner_zip.namelist() if name.endswith(core_name)]
        if not matches:
            raise ValueError(
                f"Stazione «{station}» non trovata. "
                f"File disponibili: {inner_zip.namelist()}"
            )
        csv_path = matches[0]

        # ---------- LEGGI CSV (senza parse_dates) ----------
        with inner_zip.open(csv_path) as f:
            df = pd.read_csv(f, na_values=["NA"])

    # ---------------------------------------------------------------------------
    # 4) costruisci la colonna datetime e imposta l’indice                        #
    # ---------------------------------------------------------------------------
    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce",
    )
    df = df.sort_values("datetime").set_index("datetime")

    # rimuovi colonne ridondanti
    df.drop(columns=["No", "year", "month", "day", "hour"], inplace=True, errors="ignore")

    # colonne numeriche (escludo 'wd' – stringa direzione vento – e 'station')
    numeric_cols: List[str] = (
        df.select_dtypes(include=["number"]).columns.tolist()
    )

    # valori negativi → NaN (trattano out-of-range come missing)
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0.0)

    # interpolazione temporale lineare
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit_direction="both"
    )

    # 5) feature engineering ---------------------------------------------------------
    df["hour"] = df.index.hour
    # df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    # df["is_weekend"] = (df.index.dayofweek >= 5).astype(np.uint8)
    # df["pm25_rolling24h"] = df["PM2.5"].rolling(24, min_periods=1).mean()

    # 6) salvataggio -----------------------------------------------------------------
    df.reset_index().to_csv("air_quality_cleaned.csv", index=False)

    return df


# --------------------------------------------------------------------------------------
# quick-&-dirty manual run -------------------------------------------------------------
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    cleaned = preprocess_data(
        zip_path="beijing+multi+site+air+quality+data.zip",
        station="Aotizhongxin",
    )
    print(cleaned.head())
