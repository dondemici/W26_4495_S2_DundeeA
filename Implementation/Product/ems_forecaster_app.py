# forecaster_app.py
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import URL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import requests  # new
from google import genai

def load_css(file_name: str):
    css_path = Path(__file__).parent / file_name
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_NAME = "gemini-3-flash-preview"  # or any current model

# ---------- 1. Page config ----------
st.set_page_config(
    page_title="EMS Injury / Exposure Forecaster",
    layout="wide"
    
)

st.title("EMS Work‑Related Exposure Forecast")
st.caption("Prototype app – weekly exposures forecast with several forecasting models")

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

# Approx Richmond / Vancouver coordinates
EMS_LAT = 49.18
EMS_LON = -123.13
EMS_TZ = "America/Vancouver"

@st.cache_data(ttl=6*3600)
def fetch_weather_daily_openmeteo(start_date, end_date,
                                  lat=EMS_LAT, lon=EMS_LON, timezone=EMS_TZ):
    # Clamp to a max range (e.g., 365 days)
    if (end_date - start_date).days > 365:
        start_date = end_date - pd.Timedelta(days=365)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,snowfall_sum",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": timezone,
    }

    r = requests.get(url, params=params)
    if not r.ok:
        # Return empty df so app can keep running
        try:
            err = r.json().get("reason", r.text)
        except Exception:
            err = r.text
        st.warning(f"Weather API error from Open-Meteo: {err}")
        return pd.DataFrame(columns=["date", "precip", "tempmax", "tempmin", "snow"])

    data = r.json()
    daily = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "precip": data["daily"]["precipitation_sum"],
        "tempmax": data["daily"]["temperature_2m_max"],
        "tempmin": data["daily"]["temperature_2m_min"],
        "snow": data["daily"]["snowfall_sum"],
    })
    return daily

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


# 3b. All sidebar UI goes inside st.sidebar
BASE_DIR = Path(__file__).parent
logo_path = BASE_DIR / "assets" / "ems_logo_s.png"

with st.sidebar:
    st.image(str(logo_path), width="stretch")
    #st.markdown("### EMS Forecaster")
    st.markdown("---")

    st.header("Forecast settings")

    model_choice = st.selectbox(
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

    horizon_weeks = st.slider(
        "Forecast horizon (weeks)",
        min_value=1,
        max_value=24,
        value=1,
        step=1,
    )

    st.markdown("---")
    st.markdown("**Options**")

    show_conf = st.checkbox("Show confidence band", value=False)
    use_events = st.checkbox("Include major events (concerts, rallies, holidays)", value=False)
    use_weather = st.checkbox("Include weather effects (rain/snow)", value=False)
    show_original_vs_updated = st.checkbox("Show original vs updated forecast", value=False)
    use_ai_recommendation = st.checkbox("Generate AI manager recommendation", value=False)

    run_button = st.button("Run forecast")

# Weather-enriched history
# Define allowed weather window (from Open-Meteo error message)
WEATHER_START = pd.to_datetime("2025-03-08")

if use_weather:
    # Keep only weeks where we *can* get weather
    history_df_weather = history_df[history_df["week_start"] >= WEATHER_START].copy()

    if history_df_weather.empty or len(history_df_weather) < 16:
        st.warning(
            "Not enough overlapping history with available weather data; "
            "running forecast without weather effects."
        )
        use_weather = False
        history_with_weather = history_df.copy()
    else:
        # Use trimmed history to drive both exposures + weather
        history_df = history_df_weather
        # fetch weather only over this trimmed window
        start = history_df["week_start"].min() - pd.Timedelta(days=7)
        end = history_df["week_start"].max() + pd.Timedelta(days=7)

        weather_daily = fetch_weather_daily_openmeteo(start.date(), end.date())

        if not weather_daily.empty:
            weather_daily = weather_daily.set_index("date").asfreq("D")
            weather_weekly = (
                weather_daily
                .resample("W-MON")
                .agg({
                    "precip": "sum",
                    "snow": "sum",
                    "tempmax": "mean",
                    "tempmin": "mean",
                })
                .reset_index()
                .rename(columns={"date": "week_start"})
            )

            history_with_weather = pd.merge(
                history_df,
                weather_weekly,
                on="week_start",
                how="left",
            )
        else:
            st.info("Weather data not available; running forecast without weather.")
            history_with_weather = history_df.copy()
            use_weather = False
else:
    history_with_weather = history_df.copy()


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

def sarima_forecast(df, horizon, use_weather=False):
    df = df.sort_values("week_start").copy()
    y = df["exposures"].astype(float)
    y.index = pd.DatetimeIndex(df["week_start"])

    exog_train = None
    exog_cols = []
    if use_weather:
        exog_cols = ["precip", "snow", "tempmax", "tempmin"]
        exog_train = df[exog_cols].ffill()

    model = SARIMAX(
        y,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
        exog=exog_train,
    )
    results = model.fit(disp=False)

    last_date = y.index.max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    exog_future = None
    if use_weather and exog_train is not None:
        last_row = exog_train.iloc[-1]
        exog_future = pd.DataFrame(
            [last_row.values] * horizon,
            columns=exog_train.columns,
            index=future_dates,
        )

    fcst_obj = results.get_forecast(steps=horizon, exog=exog_future)
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

# AI Recommendation Input
def generate_recommendation(history_df, forecast_df, model_choice, horizon_weeks):
        summary = build_forecast_summary(history_df, forecast_df)

        prompt = f"""
You are an EMS operations advisor for a Canadian ambulance service.

Here is a concise summary of recent history and forecast:
{summary}

Model used: {model_choice}
Forecast horizon: {horizon_weeks} weeks.

Using this information, write a short, practical recommendation
(4–6 sentences) for managers:
- focus on staffing, training, and PPE planning
- be concrete but not alarmist
- assume audience is non-technical.
"""

        response = genai_client.models.generate_content(
            model=MODEL_NAME,      # e.g. "gemini-3-flash-preview"
            contents=prompt,
        )

        return response.text


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
        base_df = history_with_weather if use_weather else history_df
        forecast_df = sarima_forecast(base_df, horizon_weeks, use_weather=use_weather)
    elif model_choice == "ML (Experimental)":
        forecast_df = ml_forecast(history_df, horizon_weeks)
    else: # "Prophet (default)" placeholder for now
        forecast_df = naive_forecast(history_df, horizon_weeks)


    # Generate AI recommendation
    with st.spinner("Generating manager recommendation..."):
        try:
            ai_reco = generate_recommendation(
                history_df, forecast_df, model_choice, horizon_weeks
            )
        except Exception as e:
            ai_reco = f"(Error generating recommendation: {e})"


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
    #st.dataframe(
    #    forecast_df[["week_start", "yhat", "yhat_lower", "yhat_upper"]]
    #    .rename(columns={
    #        "week_start": "Week start",
    #        "yhat": "Forecast",
    #        "yhat_lower": "Lower",
    #        "yhat_upper": "Upper"
    #    })
    #)
    table_df = (
    forecast_df[["week_start", "yhat", "yhat_lower", "yhat_upper"]]
    .rename(columns={
        "week_start": "Week start",
        "yhat": "Forecast",
        "yhat_lower": "Lower",
        "yhat_upper": "Upper"
    })
)
    st.data_editor(
        table_df,
        hide_index=True,
        disabled=True,  # make it read-only like a table    
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

    # Existing numeric summary
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
        text += "In a future version, major events will be added as predictors. "

    if use_weather and not history_with_weather.empty:
        corr_rain = history_with_weather["exposures"].corr(
            history_with_weather["precip"]
        )
        if pd.notna(corr_rain):
            text += (
                f" Historically, weeks with more rain have a correlation of "
                f"{corr_rain:.2f} with exposure counts, so wet weeks may need "
                f"a bit more staffing and PPE buffer. "
            )
        else:
            text += (
                " Historically, there is not enough data to estimate how rain "
                "relates to exposure counts yet. "
            )

    st.write(text)

    st.markdown("**AI‑generated operational recommendation**")

    if use_ai_recommendation:
        with st.spinner("Generating manager recommendation..."):
            try:
                ai_reco = generate_recommendation(
                    history_df, forecast_df, model_choice, horizon_weeks
                )
            except Exception as e:
                ai_reco = f"(Error generating recommendation: {e})"
    else:
        ai_reco = "(AI recommendation skipped – uncheck the box to enable it.)"

    st.write(ai_reco)