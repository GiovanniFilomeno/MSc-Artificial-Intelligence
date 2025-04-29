from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def analyze_netflix(data_path: str | Path) -> pd.DataFrame:

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    # Read the data 
    df = pd.read_csv(data_path)

    # Data cleaning and conversion
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["country"].fillna("Unknown", inplace=True)

    # Monthly count
    monthly = (
        df.set_index("date_added")
        .resample("M")
        .size()
        .rename("monthly_releases")
        .to_frame()
    )
    monthly["smoothed"] = monthly["monthly_releases"].rolling(window=3).mean()

    # Plot
    ax = monthly.plot(figsize=(9, 4))
    ax.set_title("Netflix Releases over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number")
    plt.tight_layout()
    plt.savefig("netflix_releases_over_time.pdf")
    plt.close()

    # Get first country
    df["main_country"] = (
        df["country"]
        .str.split(",", n=1).str[0]  
        .str.strip()
    )

    # Map for continent as stated in the exercise
    continent_map = {
        "United States": "North America",
        "India": "Asia",
        "United Kingdom": "Europe",
        "Japan": "Asia",
        "Brazil": "South America",
        "Unknown": "Unknown",
        "Canada": "North America",
        "France": "Europe",
        "Spain": "Europe",
        "Mexico": "North America",
    }

    df["continent"] = df["main_country"].map(continent_map).fillna("Other")

    # Count based on continenta and county
    counts = (
        df.groupby(["continent", "main_country"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["continent", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Print as stated in the exercise
    print("-----------------------------------------")
    print("Print check:")
    print(counts)
    print("-----------------------------------------")

    # Keep top 3
    counts["rank"] = counts.groupby("continent")["count"].rank(
        method="first", ascending=False
    )
    top3 = (
        counts[counts["rank"] <= 3]
        .drop(columns="rank")
        .reset_index(drop=True)
    )

    return top3


# Exercise main
if __name__ == "__main__":
    data_path = "netflix_titles.csv"
    result = analyze_netflix(data_path)
    print(result)
