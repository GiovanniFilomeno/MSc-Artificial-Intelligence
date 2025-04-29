from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def analyze_superstore(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    orders  = pd.read_excel(data_path, sheet_name="Orders")
    returns = pd.read_excel(data_path, sheet_name="Returns")[['Order ID']].drop_duplicates()

    # Merge the data
    df = orders.merge(
        returns,
        on='Order ID',       # Merge based on the 'Order ID' column
        how='left',          # Keep all rows from 'orders'
        indicator=True,      
        suffixes=("", "_returned")
    )

    df["Returned"] = df["_merge"] == "both"

    cat_sales = (
        df.pivot_table(
            values="Sales",
            index="Category",
            columns="Returned",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    # Print as stated in the exercise
    print("-----------------------------------------")
    print("Print check:")
    print(cat_sales)
    print("-----------------------------------------")

    cat_sales.rename(columns={False: "No", True: "Yes"}, inplace=True) # renaming for the plot
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["YearMonth"]  = df["Order Date"].dt.to_period("M")

    monthly_region_sales = (
        df.groupby(["YearMonth", "Region"], sort=True)["Sales"]
        .sum()
        .unstack("Region")
        .sort_index()
    )

    # Visualization
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8.27, 11.69),
        constrained_layout=True,
    )

    # Top plot
    cat_sales.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        edgecolor="white",
    )
    axes[0].set_title("Total Sales by Category and Return Status")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Total Sales")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="center")

    # Bottom plot
    monthly_region_sales.plot(
        ax=axes[1],
        linewidth=1.2,
        ms=3,
    )
    axes[1].set_title("Monthly Sales by Region")
    axes[1].set_xlabel("Year-Month")
    axes[1].set_ylabel("Sales")
    plt.setp(axes[1].get_xticklabels(), rotation=0, ha="center")

    plt.savefig("superstore_plots.pdf")
    plt.close(fig)

    return monthly_region_sales


# Exercise main
if __name__ == "__main__":
    DATA_XLSX = "superstore.xlsx"
    analysis  = analyze_superstore(DATA_XLSX)
    print(analysis.head()) 