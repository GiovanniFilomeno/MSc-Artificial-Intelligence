from __future__ import annotations
import matplotlib.dates as mdates

from pathlib import Path
from typing import List, Tuple, Optional
import pickle

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from shiny.types import FileInfo

# UI
sidebar = ui.sidebar(
    ui.input_file("csv_file", "Upload data file", multiple=False, accept={"text/csv": ".csv"}),
    ui.hr(),
    ui.output_ui("sidebar_controls"),  # updated after upload
    width="300px",
)

app_ui = ui.page_sidebar(
    sidebar,
    ui.h2("Air Quality Dashboard"),
    ui.output_plot("pollution_plot", height="600px"),
)


def server(input, output, session):
    # Load csv
    @reactive.Calc
    def df_raw() -> Optional[pd.DataFrame]:
        uploaded: List[FileInfo] = input.csv_file()
        if not uploaded:
            return None
        tmp = Path(uploaded[0]["datapath"])
        df = pd.read_csv(tmp, parse_dates=["datetime"]).set_index("datetime")
        return df

    @output
    @render.ui
    def sidebar_controls():
        df = df_raw()
        if df is None:
            return ui.markdown("Upload a **csv** data file to continue")

        pollutant_cols = [
            c
            for c in df.columns
            if df[c].dtype != "O" and c not in {"hour", "month", "dayofweek", "is_weekend"}
        ]

        return ui.TagList(
            ui.input_selectize(
                "pollutants",
                "Select Pollutants",
                choices=pollutant_cols,
                selected=["PM2.5"] if "PM2.5" in pollutant_cols else pollutant_cols[:1],
                multiple=True,
            ),
            ui.input_slider("smooth_window", "Smoothing window (days)", min=1, max=30, value=1),
            ui.input_checkbox("show_pred", "Show PM2.5 Prediction", value=False),
            ui.panel_conditional(
                "input.show_pred == true",
                ui.markdown("Upload scaler and weight files to show predictions."),
                ui.input_file("scaler_file", "Upload scaler (.pkl)"),
                ui.input_file("weights_file", "Upload weights (.pt)"),
            ),
        )

    # Load model + scaler
    @reactive.Calc
    def model_and_scaler() -> Tuple[Optional[torch.nn.Module], Optional[object]]:
        if not input.show_pred():
            return None, None
        if not input.scaler_file() or not input.weights_file():
            return None, None

        # scaler 
        scaler_path = Path(input.scaler_file()[0]["datapath"])
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # model
        from a6_ex4 import PM_Model # load ex4

        state_dict_path = Path(input.weights_file()[0]["datapath"])
        state_dict = torch.load(state_dict_path, map_location="cpu")

        n_features = int(scaler.mean_.shape[0])
        lin_sizes = [w.shape[0] for name, w in state_dict.items() if name.endswith("weight")]
        hidden_sizes = tuple(lin_sizes[:-1]) or (64, 32)

        model = PM_Model(in_features=n_features, hidden_layers=hidden_sizes)
        model.load_state_dict(state_dict)
        model.eval()
        return model, scaler

    # Plot 
    @output
    @render.plot
    def pollution_plot():
        df = df_raw()
        if df is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Upload a CSV to display data", ha="center", va="center")
            ax.axis("off")
            return fig

        pollutants = input.pollutants() or []
        window = input.smooth_window()
        show_pred = input.show_pred()

        if not pollutants:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Select at least one pollutant", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, col in enumerate(pollutants):
            series = df[col].rolling(window * 24, min_periods=1).mean()
            ax.plot(series.index, series.values, label=col, color=colors[i % len(colors)])

        # Prediction overlay
        if show_pred and "PM2.5" in pollutants:
            model, scaler = model_and_scaler()
            if model is not None and scaler is not None:
                feat_cols = (
                    df.select_dtypes(include=[np.number])
                    .drop(columns=["PM2.5"], errors="ignore")
                    .columns
                )
                X = df[feat_cols].values.astype(np.float32)
                X_scaled = scaler.transform(X)
                with torch.no_grad():
                    y_pred = model(torch.from_numpy(X_scaled)).squeeze().numpy()
                pred = pd.Series(y_pred, index=df.index).rolling(window * 24, 1).mean()
                ax.plot(
                    pred.index,
                    pred.values,
                    label="PM2.5 (Predicted)",
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                )

        ax.set_title("Pollution Levels")
        ax.set_xlabel("datetime")
        ax.set_ylabel("Value")
        ax.legend()
        # fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))  

        fig.tight_layout()
        return fig

# App
app = App(app_ui, server)
