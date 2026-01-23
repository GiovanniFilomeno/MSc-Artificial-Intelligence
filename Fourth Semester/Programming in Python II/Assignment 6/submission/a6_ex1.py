from __future__ import annotations
import io
import os
import zipfile
from typing import List

import pandas as pd
import numpy as np
import urllib.request


DATA_URL = (
    # Link for zip download
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
)


def _download_if_needed(zip_path: str, url: str = DATA_URL) -> None:
    """Scarica la risorsa remota solo se non esiste già in locale."""
    if os.path.exists(zip_path):
        return
    print(f"Downloading dataset to «{zip_path}» …")
    urllib.request.urlretrieve(url, zip_path)


# Preprocessing
def preprocess_data(zip_path: str, station: str) -> pd.DataFrame:
    # if needed, download
    _download_if_needed(zip_path)

    # Open external zip
    with zipfile.ZipFile(zip_path) as outer_zip:
        inner_name: str = next(
            name for name in outer_zip.namelist() if name.endswith(".zip")
        )
        inner_bytes = outer_zip.read(inner_name)

    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        core_name = f"PRSA_Data_{station}_20130301-20170228.csv"
        matches = [name for name in inner_zip.namelist() if name.endswith(core_name)]
        if not matches:
            raise ValueError(
                f"Stazione «{station}» non trovata. "
                f"File disponibili: {inner_zip.namelist()}"
            )
        csv_path = matches[0]

        # Read csv
        with inner_zip.open(csv_path) as f:
            df = pd.read_csv(f, na_values=["NA"])

    # Build datetime and index 
    df["datetime"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]),
        errors="coerce",
    )
    df = df.sort_values("datetime").set_index("datetime")

    # Remove redundant columns
    df.drop(columns=["No", "year", "month", "day", "hour"], inplace=True, errors="ignore")

    # Numerical columns
    numeric_cols: List[str] = (
        df.select_dtypes(include=["number"]).columns.tolist()
    )

    # Neg values
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0.0)

    # Linear interpolation
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time", limit_direction="both"
    )

    df["hour"] = df.index.hour
    df["month"] = df.index.month

    # Saving
    df.reset_index().to_csv("air_quality_cleaned.csv", index=False)

    return df


# Run
if __name__ == "__main__":
    cleaned = preprocess_data(
        zip_path="beijing+multi+site+air+quality+data.zip",
        station="Aotizhongxin",
    )
    print(cleaned.head())
