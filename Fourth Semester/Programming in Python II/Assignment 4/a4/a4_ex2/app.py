import pandas as pd
import plotly.express as px
from shiny import App, ui, reactive
from shinywidgets import output_widget, render_widget


DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"


def _load_and_preprocess() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df = df[["country", "iso_code", "year", "co2"]]
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    df.dropna(subset=["iso_code", "co2"], inplace=True)
    df = df[(df["co2"] > 0) & (df["iso_code"].str.len() == 3)]
    return df


def make_ui(initial_choices: dict):
    timeseries_sidebar = ui.sidebar(
        ui.input_selectize(
            "country_select",
            "Choose a country",
            choices=initial_choices,
            selected="Austria",
        ),
        ui.input_slider(
            "rolling_window",
            "Rolling mean (years)",
            min=1,
            max=20,
            value=5,
        ),
        width=250,
    )

    worldmap_sidebar = ui.sidebar(
        ui.input_slider(
            "year_select",
            "Year",
            min=1750,
            max=2023,
            value=2007,
            sep="",
        ),
        width=250,
    )

    return ui.page_navbar(
        ui.nav_panel(
            "Country – CO₂ Time Series",
            ui.layout_sidebar(
                timeseries_sidebar,
                output_widget("timeseries_plot"),
            ),
        ),
        ui.nav_panel(
            "World Map – CO₂ Emissions per Country",
            ui.layout_sidebar(
                worldmap_sidebar,
                output_widget("worldmap_plot"),
            ),
        ),
        title="CO₂ Dashboard",
    )


def server(input, output, session):
    @reactive.Calc
    def data() -> pd.DataFrame:
        return _load_and_preprocess()

    @reactive.effect
    def _update_country_choices():
        df = data()
        if df.empty:
            return

        country_choices = (
            df[["country"]]
            .drop_duplicates()
            .sort_values("country")
            .set_index("country")
            .index.to_series()
            .to_dict()
        )

        default_country = "Austria" if "Austria" in country_choices else next(iter(country_choices))

        session.send_input_message(
            "country_select",
            {
                "choices": country_choices,
                "value": default_country,
            },
        )

    @reactive.effect
    def _update_year_slider_limits():
        df = data()
        if df.empty:
            return

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())
        current_val = input.year_select() or 2007

        if not (min_year <= current_val <= max_year):
            current_val = 2007 if (min_year <= 2007 <= max_year) else min_year

        session.send_input_message(
            "year_select",
            {
                "min": min_year,
                "max": max_year,
                "value": current_val,
            },
        )

    @output
    @render_widget
    def timeseries_plot():
        df = data()
        country = input.country_select()

        if not country or df.empty:
            fig = px.line(title="Select a country")
            fig.update_layout(xaxis_title="Year", yaxis_title="CO₂ (million tons)")
            return fig

        country_df = df[df["country"] == country].sort_values("year")
        if country_df.empty:
            fig = px.line(title="No data available")
            fig.update_layout(xaxis_title="Year", yaxis_title="CO₂ (million tons)")
            return fig

        window = min(input.rolling_window(), len(country_df))

        country_df = country_df.copy()
        country_df["co2_smoothed"] = (
            country_df["co2"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

        fig = px.line(
            country_df,
            x="year",
            y="co2",
            title=f"CO₂ emissions – {country}",
            labels={"co2": "CO₂ (million tons)", "year": "Year"},
        )
        fig.add_scatter(
            x=country_df["year"],
            y=country_df["co2_smoothed"],
            mode="lines",
            name=f"{window}-year mean",
        )
        fig.update_layout(
            hovermode="x unified",
            legend_title="",
        )
        return fig

    @output
    @render_widget
    def worldmap_plot():
        df = data()
        year = input.year_select()

        if df.empty or year is None:
            fig = px.choropleth(title="Select a year or wait for data")
            fig.update_layout(margin={"t": 40, "l": 0, "r": 0, "b": 0})
            return fig

        year_df = df[df["year"] == year].drop_duplicates(subset="iso_code")

        if year_df.empty:
            fig = px.choropleth(title=f"No data for {year}")
            fig.update_layout(margin={"t": 40, "l": 0, "r": 0, "b": 0})
            return fig

        fig = px.choropleth(
            year_df,
            locations="iso_code",
            color="co2",
            hover_name="country",
            hover_data={"iso_code": True, "co2": True},
            color_continuous_scale="Reds",
            title=f"CO₂ emissions worldwide – {year}",
        )
        fig.update_layout(
            margin={"t": 40, "l": 0, "r": 0, "b": 0},
            coloraxis_colorbar_title="CO₂\n(million tons)",
        )
        return fig


_initial_df = _load_and_preprocess()
_initial_choices = (
    _initial_df[["country"]]
    .drop_duplicates()
    .sort_values("country")
    .set_index("country")
    .index.to_series()
    .to_dict()
)

app = App(ui=make_ui(_initial_choices), server=server)