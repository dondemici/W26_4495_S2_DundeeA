# ems_forecaster_app.py

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# 0. Fake historical data (replace with your real hourly agg from NEMSIS)
# -------------------------------------------------------------------
def load_fake_history():
    # 60 days of hourly data, with simple daily pattern
    dates = pd.date_range(end=pd.Timestamp.now().normalize(),
                          periods=60 * 24, freq="H")
    df = pd.DataFrame({"timestamp": dates})
    df["hour"] = df["timestamp"].dt.hour
    # simple pattern: busier late afternoon/evening
    base = 2 + 4 * np.sin((df["hour"] - 12) * np.pi / 12) + np.random.randn(len(df))
    df["call_volume"] = (base.clip(min=0) + 1).round().astype(int)
    return df

history_df = load_fake_history()

# -------------------------------------------------------------------
# 1. Simple forecasting "model"
#    Here: for each hour-of-day, use historical mean calls as forecast.
#    You will later replace this with your trained ML model.
# -------------------------------------------------------------------
def forecast_calls(start_time, horizon_hours, location="System-wide"):
    """
    Returns a DataFrame with one row per forecasted hour:
    columns: ["forecast_time", "predicted_calls", "high_strain"]
    """
    # build future hourly timestamps
    future_index = pd.date_range(start=start_time,
                                 periods=horizon_hours,
                                 freq="H")
    fut = pd.DataFrame({"forecast_time": future_index})
    fut["hour"] = fut["forecast_time"].dt.hour

    # hourly mean from history
    hourly_mean = (
        history_df.groupby("hour")["call_volume"]
        .mean()
        .rename("mean_calls")
    )

    fut = fut.merge(hourly_mean, on="hour", how="left")
    # add small random noise just so it looks less flat
    rng = np.random.default_rng(42)
    fut["predicted_calls"] = (fut["mean_calls"] +
                              rng.normal(scale=0.5, size=len(fut))).clip(min=0)
    fut["predicted_calls"] = fut["predicted_calls"].round(1)

    # define high-strain as top 25% of predicted calls in this window
    q75 = fut["predicted_calls"].quantile(0.75)
    fut["high_strain"] = (fut["predicted_calls"] >= q75).astype(int)

    return fut[["forecast_time", "predicted_calls", "high_strain"]]

# -------------------------------------------------------------------
# 2. Simple risk scorer based on workload (placeholder)
#    Here: map predicted calls to Low/Med/High risk.
# -------------------------------------------------------------------
def compute_risk(df, crews=None):
    """
    Add risk-related columns to forecast df.
    crews: optional number of crews to compute calls_per_crew.
    """
    out = df.copy()
    if crews and crews > 0:
        out["calls_per_crew"] = (out["predicted_calls"] / crews).round(2)
    else:
        out["calls_per_crew"] = np.nan

    # simple rule-based risk: you will later replace with a learned model
    def risk_level(row):
        if row["predicted_calls"] >= out["predicted_calls"].quantile(0.75):
            return "High"
        elif row["predicted_calls"] >= out["predicted_calls"].median():
            return "Medium"
        else:
            return "Low"

    out["risk_level"] = out.apply(risk_level, axis=1)
    return out

# -------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="EMS Workload Forecaster", layout="wide")

st.title("EMS Workload & Risk Forecaster (Prototype)")

st.markdown(
    "This prototype forecasts **hourly call volume** and flags "
    "**high‑strain hours** based on a simple historical pattern. "
    "You can later plug in your real Databricks model."
)

# ---- Input panel ---------------------------------------------------
st.sidebar.header("Forecast settings")

location = st.sidebar.selectbox(
    "Location",
    ["System-wide", "Hospital A", "Hospital B", "Zone West"],
    index=0,
)

start_time = st.sidebar.datetime_input(
    "Forecast start time",
    value=pd.Timestamp.now().replace(minute=0, second=0, microsecond=0),
)

horizon_label = st.sidebar.selectbox(
    "Forecast horizon",
    ["Next 4 hours", "Next 8 hours", "Next 12 hours", "Next 24 hours"],
    index=2,
)

horizon_map = {
    "Next 4 hours": 4,
    "Next 8 hours": 8,
    "Next 12 hours": 12,
    "Next 24 hours": 24,
}
horizon_hours = horizon_map[horizon_label]

crews = st.sidebar.number_input(
    "Planned active crews (optional)",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

show_risk = st.sidebar.checkbox("Show staff risk level", value=True)

run_btn = st.sidebar.button("Run forecast")

# ---- Main logic ----------------------------------------------------
if run_btn:
    st.subheader(f"Forecast for {location}")
    forecast_df = forecast_calls(start_time, horizon_hours, location)

    if show_risk:
        forecast_df = compute_risk(forecast_df, crews if crews > 0 else None)

    # Summary tiles
    col1, col2, col3 = st.columns(3)

    peak_row = forecast_df.loc[forecast_df["predicted_calls"].idxmax()]
    with col1:
        st.metric(
            "Peak predicted calls/hour",
            f"{peak_row['predicted_calls']}",
            help=f"At {peak_row['forecast_time']:%Y-%m-%d %H:%M}"
        )

    num_high = int(forecast_df["high_strain"].sum())
    with col2:
        st.metric(
            "High‑strain hours in window",
            f"{num_high}",
            help="Hours where predicted calls are in the top 25% for this window."
        )

    if crews > 0 and show_risk:
        max_cpc = forecast_df["calls_per_crew"].max()
        with col3:
            st.metric(
                "Max calls per crew",
                f"{max_cpc:.2f}",
                help="Highest predicted calls per crew in the window."
            )
    else:
        with col3:
            st.metric("Risk mode", "On" if show_risk else "Off")

    st.markdown("### Hour‑by‑hour forecast")
    # prettier table
    display_df = forecast_df.copy()
    display_df["forecast_time"] = display_df["forecast_time"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    st.dataframe(display_df, use_container_width=True)

    # Chart
    st.markdown("### Predicted workload (calls/hour)")

    chart_df = forecast_df.set_index("forecast_time")["predicted_calls"]
    st.bar_chart(chart_df)

else:
    st.info("Set your options in the left sidebar and click **Run forecast** to see results.")
