from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, reactive, render, ui
from shiny.types import FileInfo


sidebar = ui.sidebar(
    ui.input_file(
        "csv_file",
        "Upload an air-quality CSV",
        multiple=False,
        accept={"text/csv": ".csv"},
    ),
    ui.hr(),
    ui.output_ui("sidebar_controls"),
    width="300px",
)

app_ui = ui.page_sidebar(
    sidebar,
    ui.h2("Air-Quality Explorer"),
    ui.markdown(
        "Inspect pollutant measurements over time and apply a rolling mean. "
        "Model training and evaluation remain separate from this public dashboard."
    ),
    ui.output_plot("pollution_plot", height="600px"),
)


def server(input, output, session):
    @reactive.Calc
    def df_raw() -> Optional[pd.DataFrame]:
        uploaded: Optional[List[FileInfo]] = input.csv_file()
        if not uploaded:
            return None

        upload_path = Path(uploaded[0]["datapath"])
        frame = pd.read_csv(upload_path)
        if "datetime" not in frame.columns:
            raise ValueError("The CSV must contain a 'datetime' column.")

        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        return frame.set_index("datetime").sort_index()

    @output
    @render.ui
    def sidebar_controls():
        frame = df_raw()
        if frame is None:
            return ui.markdown("Upload a **CSV** file to begin.")

        measurement_columns = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in {"hour", "month", "dayofweek", "is_weekend"}
        ]
        if not measurement_columns:
            return ui.markdown("The uploaded file has no numeric measurement columns.")

        default = "PM2.5" if "PM2.5" in measurement_columns else measurement_columns[0]
        return ui.TagList(
            ui.input_selectize(
                "pollutants",
                "Select numeric series",
                choices=measurement_columns,
                selected=[default],
                multiple=True,
            ),
            ui.input_slider(
                "smooth_window",
                "Smoothing window (days)",
                min=1,
                max=30,
                value=1,
            ),
        )

    @output
    @render.plot
    def pollution_plot():
        frame = df_raw()
        if frame is None:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.text(0.5, 0.5, "Upload a CSV to display data", ha="center", va="center")
            axis.axis("off")
            return figure

        pollutants = input.pollutants() or []
        if not pollutants:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.text(0.5, 0.5, "Select at least one pollutant", ha="center", va="center")
            axis.axis("off")
            return figure

        window_days = input.smooth_window()
        figure, axis = plt.subplots(figsize=(10, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for index, column in enumerate(pollutants):
            series = frame[column].rolling(f"{window_days}D", min_periods=1).mean()
            axis.plot(
                series.index,
                series.values,
                label=column,
                color=colors[index % len(colors)],
            )

        axis.set_title("Air-quality measurements")
        axis.set_xlabel("Date")
        axis.set_ylabel("Measured value")
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        axis.legend()
        figure.tight_layout()
        return figure


app = App(app_ui, server)
