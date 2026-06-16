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
    # Direct link to the .zip file listed on the UCI page.
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)


def _download_if_needed(zip_path: str, url: str = DATA_URL) -> None:
    """Download the remote resource only if it does not already exist locally."""
    if os.path.exists(zip_path):
        return
    print(f"Downloading dataset to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)


# --------------------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------------------
def preprocess_data(zip_path: str, station: str) -> pd.DataFrame:
    """
    Extract, clean, and enrich the data for the selected station.

    Parameters
    ----------
    zip_path : str
        Local path to the beijing+multi+site+air+quality+data.zip file.
    station : str
        Station name, for example 'Aotizhongxin'.

    Returns
    -------
    pd.DataFrame
        DataFrame with a datetime index, ready for EDA and modelling.
    """
    # 1) Download if needed.
    _download_if_needed(zip_path)

    # 2) Open the outer zip file.
    with zipfile.ZipFile(zip_path) as outer_zip:
        # The archive contains exactly one nested zip file.
        inner_name: str = next(
            name for name in outer_zip.namelist() if name.endswith(".zip")
        )
        inner_bytes = outer_zip.read(inner_name)

    # 3) Open the inner zip directly from memory.
        # ---------------------------------------------------------------------------
    # 3) Open the inner zip, locate the station CSV, and load it.                 #
    # ---------------------------------------------------------------------------
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        core_name = f"PRSA_Data_{station}_20130301-20170228.csv"
        matches = [name for name in inner_zip.namelist() if name.endswith(core_name)]
        if not matches:
            raise ValueError(
                f"Station {station} not found. "
                f"Available files: {inner_zip.namelist()}"
            )
        csv_path = matches[0]

        # ---------- READ CSV without parse_dates ----------
        with inner_zip.open(csv_path) as f:
            df = pd.read_csv(f, na_values=["NA"])

    # ---------------------------------------------------------------------------
    # 4) Build the datetime column and set it as the index.                       #
    # ---------------------------------------------------------------------------
    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce",
    )
    df = df.sort_values("datetime").set_index("datetime")

    # Remove redundant columns.
    df.drop(columns=["No", "year", "month", "day", "hour"], inplace=True, errors="ignore")

    # Numeric columns, excluding 'wd' as wind-direction text and 'station'.
    numeric_cols: List[str] = (
        df.select_dtypes(include=["number"]).columns.tolist()
    )

    # Convert negative values to NaN, treating out-of-range values as missing.
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0.0)

    # Linear time interpolation.
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit_direction="both"
    )

    # 5) feature engineering ---------------------------------------------------------
    df["hour"] = df.index.hour
    # df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    # df["is_weekend"] = (df.index.dayofweek >= 5).astype(np.uint8)
    # df["pm25_rolling24h"] = df["PM2.5"].rolling(24, min_periods=1).mean()

    # 6) Save output. ---------------------------------------------------------------
    df.reset_index().to_csv("air_quality_cleaned.csv", index=False)

    return df


# --------------------------------------------------------------------------------------
# Quick manual run ---------------------------------------------------------------------
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    cleaned = preprocess_data(
        zip_path="beijing+multi+site+air+quality+data.zip",
        station="Aotizhongxin",
    )
    print(cleaned.head())
