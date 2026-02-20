# forecaster_app.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- 1. Page config ----------
st.set_page_config(
    page_title="EMS Injury / Exposure Forecaster",
    layout="wide"
)

st.title("EMS Work‑Related Exposure Forecast")
st.caption("Prototype app – weekly exposures forecast with SARIMA / Prophet")

# ---------- 2. Fake data for now (replace with your real weekly series) ----------
def load_fake_weekly_exposures():
    # Weekly dates for 2024
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-MON")
    n = len(dates)
    # Simple pattern: baseline 10 + seasonal bump + noise
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 52)
    counts = 10 + seasonal + np.random.normal(0, 1.5, size=n)
    counts = np.clip(np.round(counts), 0, None).astype(int)
    df = pd.DataFrame({"week_start": dates, "exposures": counts})
    return df

history_df = load_fake_weekly_exposures()

# ---------- 3. Sidebar controls ----------
st.sidebar.header("Forecast settings")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Prophet (default)", "SARIMA"],
    index=0
)

horizon_weeks = st.sidebar.slider(
    "Forecast horizon (weeks)",
    min_value=4,
    max_value=24,
    value=12,
    step=1
)

use_events = st.sidebar.checkbox(
    "Include major events (concerts, rallies, holidays)",
    value=False
)

use_weather = st.sidebar.checkbox(
    "Include weather effects (rain/snow)",
    value=False
)

show_original_vs_updated = st.sidebar.checkbox(
    "Show original vs updated forecast",
    value=False
)

run_button = st.sidebar.button("Run forecast")

# ---------- 4. Simple placeholder forecast function ----------
def simple_naive_forecast(df, horizon):
    """
    Very simple baseline:
    - Take last 8 weeks average as level.
    - Add small random noise for illustration.
    """
    df = df.sort_values("week_start")
    last_date = df["week_start"].iloc[-1]
    last_8_mean = df["exposures"].tail(8).mean()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    point_forecast = last_8_mean + np.random.normal(0, 1, size=horizon)
    point_forecast = np.clip(np.round(point_forecast), 0, None)

    # Simple prediction interval: ±3 exposures
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    fcst = pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper
    })
    return fcst

# ---------- 5. Main logic ----------
if not run_button:
    st.info("Set options in the left sidebar, then click **Run forecast**.")
else:
    # In the future you will branch here for SARIMA vs Prophet
    forecast_df = simple_naive_forecast(history_df, horizon_weeks)

    # For “original vs updated” demo, pretend updated forecast
    # is based on slightly higher recent trend
    updated_forecast_df = None
    if show_original_vs_updated:
        updated_forecast_df = forecast_df.copy()
        updated_forecast_df["yhat"] = updated_forecast_df["yhat"] + 2
        updated_forecast_df["yhat_lower"] = updated_forecast_df["yhat_lower"] + 2
        updated_forecast_df["yhat_upper"] = updated_forecast_df["yhat_upper"] + 2

    # ---------- 5a. Summary metrics ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Last historical week",
            history_df["week_start"].max().date().isoformat()
        )
    with col2:
        st.metric(
            "Model selected",
            model_choice
        )
    with col3:
        st.metric(
            "Forecast horizon",
            f"{horizon_weeks} weeks"
        )

    # ---------- 5b. Plot historical + forecast ----------
    st.subheader("Historical weekly exposures + forecast")

    plot_df_hist = history_df.rename(columns={"exposures": "value"})
    plot_df_hist["type"] = "Historical"

    plot_df_fcst = forecast_df[["week_start", "yhat"]].rename(
        columns={"week_start": "week_start", "yhat": "value"}
    )
    plot_df_fcst["type"] = "Forecast (baseline)"

    plot_df = pd.concat([plot_df_hist, plot_df_fcst], ignore_index=True)

    # Line chart
    chart_data = plot_df.pivot(index="week_start", columns="type", values="value")
    st.line_chart(chart_data)

    # Show prediction interval table (for clarity in demo)
    st.markdown("**Forecast table (baseline)**")
    st.dataframe(
        forecast_df[["week_start", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={
            "week_start": "Week start",
            "yhat": "Forecast",
            "yhat_lower": "Lower",
            "yhat_upper": "Upper"
        })
    )

    # ---------- 5c. Original vs updated overlay ----------
    if show_original_vs_updated and updated_forecast_df is not None:
        st.subheader("Original vs updated forecast (illustration)")

        comp = pd.DataFrame({
            "week_start": forecast_df["week_start"],
            "Baseline forecast": forecast_df["yhat"],
            "Updated forecast": updated_forecast_df["yhat"]
        }).set_index("week_start")

        st.line_chart(comp)

    # ---------- 5d. Narrative explanation ----------
    st.subheader("Interpretation (for managers)")
    next4 = forecast_df.head(4)
    avg_next4 = next4["yhat"].mean()
    avg_lower = next4["yhat_lower"].mean()
    avg_upper = next4["yhat_upper"].mean()

    text = (
        f"Over the next 4 weeks, the model expects about **{avg_next4:.1f}** "
        f"work‑related exposures per week "
        f"(roughly {avg_lower:.0f} to {avg_upper:.0f}). "
    )

    if use_events:
        text += "In a future version, major events (concerts, rallies, holidays) will be added as extra predictors to adjust these estimates upward on high‑risk weeks. "
    if use_weather:
        text += "Weather (rain, snow, extreme heat) will also be added as regressors to capture environmental effects on injury risk. "

    st.write(text)
