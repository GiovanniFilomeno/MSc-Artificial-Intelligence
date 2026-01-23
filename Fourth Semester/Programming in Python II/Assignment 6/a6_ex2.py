# a6_ex2.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns     # ← se preferite seaborn per l’heatmap

# ----------------------------------------------------------------------
# Figure 1 – trend giornaliero PM2.5
# ----------------------------------------------------------------------
def plot_pm25_trend(df: pd.DataFrame) -> None:
    """
    Plotta la media giornaliera di PM2.5 e salva come «eda_pm25_trend.pdf».
    Il DataFrame deve contenere la colonna 'PM2.5' e un DatetimeIndex
    (come restituito dal preprocess dell’esercizio 1).
    """
    daily = df["PM2.5"].resample("D").mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, linewidth=0.8)
    ax.set_title("Daily Average PM2.5")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    fig.tight_layout()
    fig.savefig("eda_pm25_trend.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2 – heatmap di correlazione
# ----------------------------------------------------------------------
def plot_correlation(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    # limiti dinamici
    vmin = corr.values.min()
    vmax = corr.values.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlation Heatmap")

    # annotazioni
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson r", rotation=-90, va="bottom")

    fig.tight_layout()
    fig.savefig("eda_correlation_heatmap.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 – istogramma PM2.5 medio per giorno
# ----------------------------------------------------------------------
def plot_histogram_pm25(df: pd.DataFrame) -> None:
    """
    Plotta la distribuzione della media giornaliera di PM2.5
    e salva come «eda_pm25_histogram.pdf».
    """
    daily = df["PM2.5"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(daily.values, bins=40, edgecolor="black")
    ax.set_title("PM2.5 Histogram")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("eda_pm25_histogram.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Facoltativo: esecuzione da riga di comando
# ----------------------------------------------------------------------
def _cli() -> None:
    """
    Esempio di uso:

        python a6_ex2.py --csv air_quality_cleaned.csv
    """
    parser = argparse.ArgumentParser(description="Generate EDA plots.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("air_quality_cleaned.csv"),
        help="Percorso al file CSV pulito (default: air_quality_cleaned.csv)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Il file {args.csv} non esiste. "
            "Assicurati di aver eseguito a6_ex1.py prima."
        )

    df = pd.read_csv(args.csv, parse_dates=["datetime"]).set_index("datetime")

    plot_pm25_trend(df)
    plot_correlation(df)
    plot_histogram_pm25(df)
    print("→ Grafici salvati in PDF nella cartella corrente.")


if __name__ == "__main__":
    # Esempio minimo: carica il CSV generato dall’esercizio 1
    df = (
        pd.read_csv("air_quality_cleaned.csv", parse_dates=["datetime"])
          .set_index("datetime")
    )
    plot_pm25_trend(df)
    plot_correlation(df)
    plot_histogram_pm25(df)
    print("Grafici salvati (PDF) nella cartella corrente.")