from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Daily trend PM2.5
def plot_pm25_trend(df: pd.DataFrame) -> None:
    daily = df["PM2.5"].resample("D").mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, linewidth=0.8)
    ax.set_title("Daily Average PM2.5")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    fig.tight_layout()
    fig.savefig("eda_pm25_trend.pdf")
    plt.close(fig)

# Heatmap correlation
def plot_correlation(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    # Limit given by range
    vmin = corr.values.min()
    vmax = corr.values.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlation Heatmap")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson r", rotation=-90, va="bottom")

    fig.tight_layout()
    fig.savefig("eda_correlation_heatmap.pdf")
    plt.close(fig)


# Histogram avg daily
def plot_histogram_pm25(df: pd.DataFrame) -> None:
    daily = df["PM2.5"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(daily.values, bins=40, edgecolor="black")
    ax.set_title("PM2.5 Histogram")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("eda_pm25_histogram.pdf")
    plt.close(fig)


if __name__ == "__main__":
    df = (
        pd.read_csv("air_quality_cleaned.csv", parse_dates=["datetime"])
          .set_index("datetime")
    )
    plot_pm25_trend(df)
    plot_correlation(df)
    plot_histogram_pm25(df)
    print("Graphs saved (PDF) in the current folder.")