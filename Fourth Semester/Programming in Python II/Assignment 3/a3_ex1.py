from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _compute_position_score(row: pd.Series) -> float | np.nan:
    pos = str(row["club_position"]).strip().upper()

    if pos == "ST":  # Striker
        return (
            0.4 * row["pace"]
            + 0.4 * row["shooting"]
            + 0.2 * row["dribbling"]
        )

    if pos == "CM":  # Central Midfielder
        return (
            0.3 * row["passing"]
            + 0.3 * row["dribbling"]
            + 0.4 * row["power_stamina"]
        )

    if pos == "CB":  # Central Back
        return (
            0.4 * row["defending"]
            + 0.3 * row["physic"]
            + 0.3 * row["pace"]
        )

    if pos == "GK":  # Goalkeeper
        return 0.25 * (
            row["goalkeeping_diving"]
            + row["goalkeeping_reflexes"]
            + row["goalkeeping_handling"]
            + row["goalkeeping_positioning"]
        )

    return np.nan


def analyze_fifa(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path, low_memory=False) # Part 1.a
    df.fillna(0, inplace=True)  # Part 1.b

    # Part 1.c
    df["position_score"] = df.apply(_compute_position_score, axis=1)

    # Part 1.e
    plt.figure()
    df["position_score"].dropna().plot.hist(bins=50)
    plt.title("Position Score Distribution")
    plt.tight_layout()
    plt.grid()
    plt.ylabel('')
    plt.savefig("position_score_distribution.pdf")
    plt.close()

    # Part 1.d
    cols = ["short_name", "club_position", "position_score"]
    df_reduced = (
        df[cols]
        .dropna(subset=['position_score'])
        .sort_values(["club_position", "position_score"], ascending=[True, False])
        .groupby("club_position", as_index=False, sort=False)
        .head(3)  # Request by exercise
        .reset_index(drop=True)
    )

    df_reduced = df_reduced.sort_values("position_score", ascending=False).reset_index(drop=True)

    return df_reduced


# Example from exercise
if __name__ == "__main__":
    data_path = "players_22.csv"          
    top_players = analyze_fifa(data_path)
    print(top_players[0:12])
