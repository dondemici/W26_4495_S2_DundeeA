# forecaster_app.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

import sqlalchemy as sa
from sqlalchemy.engine import URL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- 1. Page config ----------
st.set_page_config(
    page_title="EMS Injury / Exposure Forecaster",
    layout="wide"
)

st.title("EMS Work‑Related Exposure Forecast")
st.caption("Prototype app – weekly exposures forecast with SARIMA / Prophet")

# ---------- 2. Fake data for now (replace with your real weekly series) ----------
#def load_fake_weekly_exposures():
    # Weekly dates for 2024
#   dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-MON")
#   n = len(dates)
#    # Simple pattern: baseline 10 + seasonal bump + noise
#    seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 52)
#    counts = 10 + seasonal + np.random.normal(0, 1.5, size=n)
#    counts = np.clip(np.round(counts), 0, None).astype(int)
#    df = pd.DataFrame({"week_start": dates, "exposures": counts})
#    return df

# history_df = load_fake_weekly_exposures()

# Build SQLAlchemy engine using st.secrets
@st.cache_resource
def get_engine():
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={st.secrets['server']};"
        f"DATABASE={st.secrets['database']};"
        "Trusted_Connection=yes;"
    )
    connection_url = URL.create(
        "mssql+pyodbc",
        query={"odbc_connect": connection_string},
    )
    engine = sa.create_engine(connection_url)
    return engine

@st.cache_data(ttl=3600)
def load_weekly_exposures_from_fact():
    engine = get_engine()

    chunks = pd.read_sql(
        """
        SELECT PcrKey, EventTime_raw, exposure_flag
        FROM dbo.PCR_Exposure_Minimal
        """,
        engine,
        chunksize=100_000,
    )

    daily_counts = {}

    for chunk in chunks:
        # 1) Parse datetime exactly as in notebook
        chunk["event_dt"] = pd.to_datetime(
            chunk["EventTime_raw"].astype(str).str.strip(),
            format="%d%b%Y:%H:%M:%S",
            errors="coerce",
        )

        # 2) Keep valid datetimes and exposure_flag == 1
        chunk = chunk[
            chunk["event_dt"].notna()
            & (chunk["exposure_flag"] == 1)
        ]

        # 3) Count per calendar day
        vc = chunk["event_dt"].dt.date.value_counts()
        for d, c in vc.items():
            daily_counts[d] = daily_counts.get(d, 0) + c

    # 4) Build daily series
    if not daily_counts:
        return pd.DataFrame(columns=["week_start", "exposures"])

    daily = pd.Series(daily_counts).sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D", fill_value=0)
    daily.name = "exposure_count"

    # 5) Collapse to weekly (Mon-based) for the app
    weekly = (
        daily
        .resample("W-MON")
        .sum()
        .rename("exposures")
        .reset_index()
        .rename(columns={"index": "week_start"})
    )

    return weekly

history_df = load_weekly_exposures_from_fact()


# ---------- 3. Sidebar controls ----------
st.sidebar.header("Forecast settings")

model_choice = st.sidebar.selectbox(
    "Model",
    [
        "Prophet (default)",
        "Naive",
        "Moving Average",
        "Exponential Smoothing",
        "Holt",
        "Holt-Winters",
        "SARIMA",
        "ML (Experimental)",
    ],
    index=0,
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
# Define extra forecast functions (Naive, Moving Avg, SES, SARIMA)
def naive_forecast(df, horizon):
    df = df.sort_values("week_start")
    last_date = df["week_start"].iloc[-1]
    last_8_mean = df["exposures"].tail(8).mean()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    point_forecast = np.full(horizon, last_8_mean)
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def moving_average_forecast(df, horizon, window=8):
    df = df.sort_values("week_start")
    last_date = df["week_start"].iloc[-1]
    ma_level = df["exposures"].rolling(window).mean().iloc[-1]

    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    point_forecast = np.full(horizon, ma_level)
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def ses_forecast(df, horizon):
    df = df.sort_values("week_start")
    y = df["exposures"].astype(float)
    model = SimpleExpSmoothing(y).fit(optimized=True)
    fcst_vals = model.forecast(horizon)

    last_date = df["week_start"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON"
    )

    point_forecast = fcst_vals.values
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def sarima_forecast(df, horizon):
    df = df.sort_values("week_start")
    y = df["exposures"].astype(float)
    y.index = pd.DatetimeIndex(df["week_start"])

    model = SARIMAX(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    fcst_obj = results.get_forecast(steps=horizon)
    mean_fcst = fcst_obj.predicted_mean
    conf_int = fcst_obj.conf_int(alpha=0.05)

    return pd.DataFrame({
        "week_start": mean_fcst.index,
        "yhat": mean_fcst.values,
        "yhat_lower": conf_int.iloc[:, 0].values,
        "yhat_upper": conf_int.iloc[:, 1].values,
    })


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

def holt_forecast(df, horizon):
    df = df.sort_values("week_start")
    y = df["exposures"].astype(float)

    model = ExponentialSmoothing(
        y,
        trend="add",              # additive trend
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    fcst_vals = model.forecast(horizon)

    last_date = df["week_start"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    point_forecast = fcst_vals.values
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def holt_winters_forecast(df, horizon, seasonal_periods=52):
    df = df.sort_values("week_start")
    y = df["exposures"].astype(float)
    
    if len(y) < 2 * seasonal_periods:
        raise ValueError(
            f"Need at least {2 * seasonal_periods} observations for Holt-Winters "
            f"(got {len(y)}). Use Holt (trend only) instead."
        )


    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)

    fcst_vals = model.forecast(horizon)

    last_date = df["week_start"].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    point_forecast = fcst_vals.values
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": future_dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def make_supervised(df, n_lags=4):
    df = df.sort_values("week_start").copy()
    for lag in range(1, n_lags+1):
        df[f"lag_{lag}"] = df["exposures"].shift(lag)
    df["roll4"] = df["exposures"].rolling(4).mean()
    df = df.dropna()
    return df

def ml_forecast(df, horizon, n_lags=4):
    sup = make_supervised(df, n_lags=n_lags)
    feature_cols = [c for c in sup.columns if c not in ["week_start", "exposures"]]
    X = sup[feature_cols]
    y = sup["exposures"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)

    # Recursive forecasting
    last_row = sup.iloc[-1].copy()
    last_date = df["week_start"].max()
    forecasts = []
    dates = []

    current_values = list(df["exposures"].tail(n_lags).values)

    for i in range(horizon):
        # Build feature vector from current_values
        feat = {}
        for lag in range(1, n_lags+1):
            feat[f"lag_{lag}"] = current_values[-lag]
        feat["roll4"] = np.mean(current_values[-4:])

        X_future = pd.DataFrame([feat])[feature_cols]
        y_pred = model.predict(X_future)

        # Append prediction
        current_values.append(y_pred)
        dates.append(last_date + pd.Timedelta(weeks=i+1))
        forecasts.append(y_pred)

    point_forecast = np.array(forecasts)
    lower = np.clip(point_forecast - 3, 0, None)
    upper = point_forecast + 3

    return pd.DataFrame({
        "week_start": dates,
        "yhat": point_forecast,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })

def build_forecast_summary(history_df, forecast_df):
    last_hist = history_df["week_start"].max().date().isoformat()
    avg_hist = history_df["exposures"].tail(8).mean()

    next4 = forecast_df.head(4)
    avg_next4 = next4["yhat"].mean()

    summary = (
        f"Historical EMS exposures up to {last_hist}, "
        f"recent 8-week average: {avg_hist:.1f} per week. "
        f"Forecast average for next 4 weeks: {avg_next4:.1f} per week.\n"
    )
    return summary

# ---------- 5. Main logic ----------
if not run_button:
    st.info("Set options in the left sidebar, then click **Run forecast**.")
else:
    #Switch on model_choice in the main block

    # In the future you will branch here for SARIMA vs Prophet
    #forecast_df = simple_naive_forecast(history_df, horizon_weeks)

    if model_choice == "Naive":
        forecast_df = naive_forecast(history_df, horizon_weeks)
    elif model_choice == "Moving Average":
        forecast_df = moving_average_forecast(history_df, horizon_weeks)
    elif model_choice == "Exponential Smoothing":
        forecast_df = ses_forecast(history_df, horizon_weeks)
    elif model_choice == "Holt":
        forecast_df = holt_forecast(history_df, horizon_weeks)
    elif model_choice == "Holt-Winters" and len(history_df) < 2 * 52:
        st.warning("Not enough history for Holt-Winters; using Holt instead.")
        forecast_df = holt_forecast(history_df, horizon_weeks)
    elif model_choice == "SARIMA":
        forecast_df = sarima_forecast(history_df, horizon_weeks)
    elif model_choice == "ML (Experimental)":
        forecast_df = ml_forecast(history_df, horizon_weeks)
    else: # "Prophet (default)" placeholder for now
        forecast_df = naive_forecast(history_df, horizon_weeks)

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
