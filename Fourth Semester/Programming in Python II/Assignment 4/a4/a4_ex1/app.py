from io import BytesIO

import pandas as pd
from shiny import App, ui, reactive, render
from shinywidgets import render_widget

rv_raw_data = reactive.Value(None)   
rv_clean_data = reactive.Value(None)  
rv_analysis  = reactive.Value(None)  

def make_ui():
    sidebar = ui.sidebar(
        ui.input_file("file", "Upload CSV file", accept=[".csv"], multiple=False),
        ui.input_action_button("analyze", "Analyze", class_="btn-primary"),
        ui.hr(),
        ui.tags.small("Tip: type a column name, press Enter or Tab to confirm."),
        ui.input_selectize(
            "remove_cols",
            "Remove Columns",
            choices=[],
            multiple=True,
            options={"create": True, "persist": True, "selectOnTab": True, "closeAfterSelect": False},
        ),
        ui.input_select(
            "na_handling",
            "With NaNs:",
            choices={
                "none": "No change",
                "zero": "Replace with 0",
                "mean": "Replace with mean",
                "median": "Replace with median",
                "drop": "Drop rows with missing values",
            },
            selected="none",
        ),
        ui.input_selectize(
            "transform_cols",
            "Columns to transform",
            choices=[],
            multiple=True,
            options={"create": True, "persist": True, "selectOnTab": True, "closeAfterSelect": False},
        ),
        ui.input_select(
            "transform_strategy",
            "Transform Strategy:",
            choices={
                "none": "No change",
                "normalize": "Normalize",
                "standardize": "Standardize",
            },
            selected="none",
        ),
        ui.input_action_button("clean", "Clean", class_="btn-success"),
        ui.download_button("download", "Download Cleaned Data"),
        ui.input_action_button("reset", "Reset", class_="btn-secondary mt-2"),
        ui.hr(),
        ui.input_dark_mode(id="page_mode"),
        width=260,
    )

    main_panel = ui.navset_pill(
        ui.nav_panel("Data", ui.output_data_frame("data_tbl")),
        ui.nav_panel("Analysis", ui.output_data_frame("analysis_tbl")),
    )

    return ui.page_sidebar(sidebar, main_panel, title="Data Cleaner")


app_ui = make_ui()

def server(input, output, session):

    @reactive.effect
    def _load_file():
        fileinfo = input.file()
        if not fileinfo:
            return
        try:
            df = pd.read_csv(fileinfo[0]["datapath"])
        except Exception as e:
            ui.notification_show(f"Error loading file: {e}", type="error", duration=5)
            return

        rv_raw_data.set(df)
        rv_clean_data.set(df.copy())

        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        session.send_input_message("remove_cols", {"choices": cols, "value": []})
        session.send_input_message("transform_cols", {"choices": num_cols, "value": []})


    @reactive.effect
    @reactive.event(input.analyze)
    def _analyze():
        df = rv_clean_data.get()
        if df is None:
            return
        info = (
            pd.DataFrame({
                "Column": df.columns,
                "Missing values": df.isna().sum().values,
                "Data type": df.dtypes.astype(str).values,
                "Nr. of unique values": df.nunique().values,
            })
            .sort_values("Missing values", ascending=False, ignore_index=True)
        )
        rv_analysis.set(info)

    @reactive.effect
    @reactive.event(input.clean)
    def _clean():
        df = rv_clean_data.get()
        if df is None:
            return

        drop_cols = input.remove_cols() or []
        drop_cols = [c.strip() for c in drop_cols if c.strip() in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        na_opt = input.na_handling()
        if na_opt == "zero":
            df = df.fillna(0)
        elif na_opt == "mean":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif na_opt == "median":
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif na_opt == "drop":
            df = df.dropna()

        t_cols = input.transform_cols() or []
        t_strategy = input.transform_strategy()
        if t_strategy != "none":
            for col in t_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if t_strategy == "normalize":
                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    elif t_strategy == "standardize":
                        df[col] = (df[col] - df[col].mean()) / df[col].std()

        rv_clean_data.set(df)

        updated_cols = df.columns.tolist()
        updated_num_cols = df.select_dtypes(include="number").columns.tolist()

        session.send_input_message(
            "remove_cols", 
            {"choices": updated_cols, "value": [col for col in drop_cols if col in updated_cols]}
        )

        session.send_input_message(
            "transform_cols", 
            {"choices": updated_num_cols, "value": [col for col in t_cols if col in updated_num_cols]}
        )


    @reactive.effect
    @reactive.event(input.reset)
    def _reset():
        raw = rv_raw_data.get()
        if raw is None:
            return
        rv_clean_data.set(raw.copy())
        rv_analysis.set(None)

        cols = raw.columns.tolist()
        num_cols = raw.select_dtypes(include="number").columns.tolist()

        session.send_input_message("remove_cols", {"choices": cols, "value": []})
        session.send_input_message("na_handling", {"value": "none"})
        session.send_input_message("transform_cols", {"choices": num_cols, "value": []})
        session.send_input_message("transform_strategy", {"value": "none"})


    @output
    @render.data_frame
    def data_tbl():
        df = rv_clean_data.get()
        return df if df is not None and not df.empty else pd.DataFrame()

    @output
    @render.data_frame
    @reactive.event(input.analyze, ignore_none=False)
    def analysis_tbl():
        df = rv_clean_data.get()
        if df is None:
            return pd.DataFrame()
        info = (
            pd.DataFrame({
                "Column": df.columns,
                "Missing values": df.isna().sum().values,
                "Data type": df.dtypes.astype(str).values,
                "Nr. of unique values": df.nunique().values,
            })
            .sort_values("Missing values", ascending=False, ignore_index=True)
        )
        return info


    @output
    @render.download
    def download():
        def _bytes():
            df = rv_clean_data.get() or pd.DataFrame()
            buf = BytesIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return buf.read()
        return {"filename": "cleaned_data.csv", "data": _bytes}

app = App(app_ui, server)
