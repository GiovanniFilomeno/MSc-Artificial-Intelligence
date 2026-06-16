from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def analyze_titanic(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    # Read the data
    df = pd.read_csv(data_path)

    # Percentage per class and sex 
    survival_rate = (
        df.groupby(["Pclass", "Sex"], sort=True)["Survived"]
        .mean()                    
        .unstack("Sex")           
        .sort_index()              
    )

    # # Print as stated in the exercise
    # print("-----------------------------------------")
    # print("Print check:")
    # print(survival_rate)
    # print("-----------------------------------------")

    # Plot
    ax = survival_rate.plot(
        kind="bar",
        figsize=(6, 4),
        edgecolor="white",
    )
    ax.set_title("Percentage of Titanic Survivors per Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Percentage of Survivors")
    ax.legend(title="Sex")

    # Convert y-axis to percentage format
    ax.set_ylim(0, 1)
    ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()])

    plt.tight_layout()
    plt.savefig("titanic_survival_rate.pdf")
    plt.close()

    # Return value
    survivors_count = (
        df.groupby(["Pclass", "Sex"], sort=True)["Survived"]
        .sum() # --> sum them 
        .reset_index(name="value")
    )
    survivors_count["variable"] = "Survived"

    # Re-order columns to match example
    survivors_count = survivors_count[["Pclass", "Sex", "variable", "value"]]

    return survivors_count


# Exercise main
if __name__ == "__main__":
    data_path = "titanic.csv"
    result = analyze_titanic(data_path)
    print(result)
