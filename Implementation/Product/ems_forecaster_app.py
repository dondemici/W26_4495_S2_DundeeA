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
from sklearn.metrics import mean_absolute_error, mean_squared_error

import requests  # new
from google import genai
from openai import OpenAI

def load_css(file_name: str):
    css_path = Path(__file__).parent / file_name
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#MODEL_NAME = "gemini-3-flash-preview"  # or any current model
GEMINI_MODEL_NAME = "gemini-2.0-flash"
OPENAI_MODEL_NAME = "gpt-5"

# Optional OpenAI client (may not be installed)
try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except ImportError:
    openai_client = None
    OPENAI_AVAILABLE = False


# ---------- 1. Page config ----------
st.set_page_config(
    page_title="EMS Injury / Exposure Forecaster",
    layout="wide"
    
)




st.title("EMS Work‑Related Injury Forecast")
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

#load from SQL
#@st.cache_data(ttl=3600)
#def load_weekly_exposures_from_fact():
#    engine = get_engine()

#    chunks = pd.read_sql(
#        """
#        SELECT PcrKey, EventTime_raw, exposure_flag
#        FROM dbo.PCR_Exposure_All
#        """,
#        engine,
#        chunksize=100_000,
#    )

#    daily_counts = {}

#    for chunk in chunks:
        # 1) Parse datetime exactly as in notebook
#        chunk["event_dt"] = pd.to_datetime(
#            chunk["EventTime_raw"].astype(str).str.strip(),
#            format="%d%b%Y:%H:%M:%S",
#            errors="coerce",
#        )

        # 2) Keep valid datetimes and exposure_flag == 1
#        chunk = chunk[
#            chunk["event_dt"].notna()
#            & (chunk["exposure_flag"] == 1)
#        ]

        # 3) Count per calendar day
#        vc = chunk["event_dt"].dt.date.value_counts()
#        for d, c in vc.items():
#            daily_counts[d] = daily_counts.get(d, 0) + c

    # 4) Build daily series
#    if not daily_counts:
#        return pd.DataFrame(columns=["week_start", "exposures"])

#    daily = pd.Series(daily_counts).sort_index()
#    daily.index = pd.to_datetime(daily.index)
#    daily = daily.asfreq("D", fill_value=0)
#    daily.name = "exposure_count"

    # 5) Collapse to weekly (Mon-based) for the app
#    weekly = (
#        daily
#        .resample("W-MON")
#        .sum()
#        .rename("exposures")
#        .reset_index()
#        .rename(columns={"index": "week_start"})
#    )

#    return weekly

CSV_PATH = Path(__file__).resolve().parents[2] / "Misc" / "PCR_Exposure_Final.csv"

@st.cache_data(ttl=3600)
def load_weekly_exposures_from_fact():
    chunks = pd.read_csv(
        CSV_PATH,
        header=None,
        names=["PcrKey", "EventTime_raw"],
        chunksize=100_000,
    )

    daily_counts = {}
    debug_rows = []
    total_rows = 0
    total_valid = 0
    total_2023_rows = 0
    total_2023_valid = 0

    for chunk in chunks:
        total_rows += len(chunk)

        raw_str = chunk["EventTime_raw"].astype(str).str.strip()

        is_2023 = raw_str.str.contains("2023", na=False)
        total_2023_rows += int(is_2023.sum())

        chunk["event_dt"] = pd.to_datetime(
            raw_str,
            format="%d%b%Y:%H:%M:%S",
            errors="coerce",
        )

        valid_mask = chunk["event_dt"].notna()
        total_valid += int(valid_mask.sum())
        total_2023_valid += int((is_2023 & valid_mask).sum())

        sample_good_2023 = chunk.loc[is_2023 & valid_mask, "EventTime_raw"].head(5).tolist()
        sample_bad_2023 = chunk.loc[is_2023 & ~valid_mask, "EventTime_raw"].head(5).tolist()

        if sample_good_2023 or sample_bad_2023:
            debug_rows.append({
                "sample_good_2023": sample_good_2023,
                "sample_bad_2023": sample_bad_2023,
            })

        chunk = chunk[valid_mask]

        vc = chunk["event_dt"].dt.date.value_counts()
        for d, c in vc.items():
            daily_counts[d] = daily_counts.get(d, 0) + c

    if not daily_counts:
        weekly = pd.DataFrame(columns=["week_start", "exposures"])
    else:
        daily = pd.Series(daily_counts).sort_index()
        daily.index = pd.to_datetime(daily.index)
        daily = daily.asfreq("D", fill_value=0)
        daily.name = "exposure_count"

        weekly = (
            daily
            .resample("W-MON")
            .sum()
            .rename("exposures")
            .reset_index()
            .rename(columns={"index": "week_start"})
        )

    debug_info = {
        "csv_path": str(CSV_PATH),
        "total_rows": total_rows,
        "total_valid": total_valid,
        "total_2023_rows": total_2023_rows,
        "total_2023_valid": total_2023_valid,
        "weekly_min": None if weekly.empty else weekly["week_start"].min(),
        "weekly_max": None if weekly.empty else weekly["week_start"].max(),
        "debug_samples": debug_rows[:3],
    }

    return weekly, debug_info



TM_API_KEY = st.secrets["TICKETMASTER_API_KEY"]

@st.cache_data(ttl=6*3600)
def fetch_ticketmaster_events(start_dt, end_dt):
    """
    Fetch events from Ticketmaster Discovery API for Greater Vancouver
    between start_dt and end_dt (Python datetimes).
    Returns a DataFrame with week_start and n_events.
    """
    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    # Ticketmaster expects ISO 8601 with Z
    start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apikey": TM_API_KEY,
        "latlong": "49.2827,-123.1207",
        "radius": 250,
        "unit": "km",
        "startDateTime": start_iso,
        "endDateTime": end_iso,
        "size": 200,
    }

    import math
    all_events = []

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "_embedded" not in data or "events" not in data["_embedded"]:
            #st.write("Ticketmaster params used:", params)
            #st.write("Ticketmaster raw response:", data)
            return pd.DataFrame(columns=["week_start", "n_events"])
        
        events = data["_embedded"]["events"]
        for ev in events:
            # pick start date
            dates = ev.get("dates", {})
            start = dates.get("start", {})
            date_str = start.get("localDate") or start.get("dateTime")
            if not date_str:
                continue
            event_date = pd.to_datetime(date_str).date()
            all_events.append(event_date)

        if not all_events:
            return pd.DataFrame(columns=["week_start", "n_events"])

        events_df = pd.DataFrame({"date": pd.to_datetime(all_events)})
        # convert to Monday week start
        events_df["week_start"] = events_df["date"] - pd.to_timedelta(
            events_df["date"].dt.weekday, unit="D"
        )

        weekly = (
            events_df.groupby("week_start")
            .size()
            .reset_index(name="n_events")
        )
        return weekly

    except Exception as e:
        st.warning(f"Ticketmaster API error: {e}")
        return pd.DataFrame(columns=["week_start", "n_events"])


history_df, debug_info = load_weekly_exposures_from_fact()


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
            #"ML (Experimental)",
        ],
        index=0,
    )
#
# horizon_weeks = st.slider(
#    "Forecast horizon (weeks)",
#    min_value=1,
#    max_value=24,
#    value=1,
#    step=1,
#)

    st.markdown("---")
    st.markdown("**Options**")

    #show_conf = st.checkbox("Show confidence band", value=False)
    show_forecast_table = st.checkbox("Show forecast table", value=False)
    show_model_comparison = st.checkbox("Show model comparison", value=False)
    #show_original_vs_updated = st.checkbox("Show original vs updated forecast", value=False)
    use_ai_recommendation = st.checkbox("Generate AI manager recommendation", value=False)
    use_weather = st.checkbox("Include weather effects (rain/snow)", value=False)
    use_events = st.checkbox("Include major events (concerts, rallies, holidays)", value=False)
    exclude_2023_from_training = st.checkbox("Exclude 2023 from SARIMA training",value=True)


    run_button = st.button("Run forecast")
    if run_button:
        st.session_state["run_forecast"] = True

# Weather-enriched history
# Define allowed weather window (from Open-Meteo error message)

WEATHER_START = pd.to_datetime("2025-03-08")

if "horizon_weeks" not in st.session_state:
    st.session_state["horizon_weeks"] = 1

horizon_weeks = st.session_state["horizon_weeks"]

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



# Starting from your existing logic:
# history_df: base weekly exposures
# history_with_weather: either history_df or merged with weather
base_for_events = history_with_weather.copy()
history_with_events = base_for_events.copy()
future_events_weekly = pd.DataFrame(columns=["week_start", "n_events"])

if use_events:
    event_start = base_for_events["week_start"].max()
    event_end = event_start + pd.Timedelta(weeks=horizon_weeks)

    future_events_weekly = fetch_ticketmaster_events(event_start, event_end)

    if not future_events_weekly.empty:
        st.subheader("Ticketmaster events retrieved")
        st.dataframe(
            future_events_weekly.sort_values("week_start"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No Ticketmaster events found for the selected date range.")
else:
    history_with_events = base_for_events.copy()


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

def sarima_forecast(df, horizon, use_weather=False, use_events=False):
    df = df.sort_values("week_start").copy()
    y = df["exposures"].astype(float)
    y.index = pd.DatetimeIndex(df["week_start"])

    exog_cols = []
    if use_weather:
        exog_cols += ["precip", "snow", "tempmax", "tempmin"]
    if use_events and "n_events" in df.columns:
        exog_cols += ["n_events"]

    exog_train = df[exog_cols].ffill() if exog_cols else None

    n = len(y)

    # If we only have ~1 year (like 2024 only), drop the seasonal part
    if n < 80:
        order = (1, 0, 1)
        seasonal_order = (0, 0, 0, 0)
    else:
        order = (1, 0, 0)
        seasonal_order = (0, 1, 1, 52)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
        exog=exog_train,
    )
    results = model.fit(disp=False, method="lbfgs")

    last_date = y.index.max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    exog_future = None
    if exog_cols:
        # For now, assume future weeks have same exog as last observed
        last_row = exog_train.iloc[-1]
        exog_future = pd.DataFrame(
            [last_row.values] * horizon,
            columns=exog_cols,
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


def backtest_and_score(df, model_name, horizon=8):
    """
    Use the last `horizon` weeks as a test set.
    Fit the chosen model on the earlier history, forecast those weeks,
    and compute MAE / RMSE / MAPE.
    """
    df = df.sort_values("week_start").reset_index(drop=True)

    if len(df) <= horizon + 8:  # need some history
        return None

    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    if model_name == "Naive":
        fcst = naive_forecast(train, horizon)
    elif model_name == "Moving Average":
        fcst = moving_average_forecast(train, horizon)
    elif model_name == "Exponential Smoothing":
        fcst = ses_forecast(train, horizon)
    elif model_name == "Holt":
        fcst = holt_forecast(train, horizon)
    elif model_name == "Holt-Winters":
        try:
            fcst = holt_winters_forecast(train, horizon)
        except ValueError:
            return None
    elif model_name == "SARIMA":
        fcst = sarima_forecast(train, horizon, use_weather=False, use_events=False)
    elif model_name == "ML (Experimental)":
        fcst = ml_forecast(train, horizon)
    else:
        fcst = naive_forecast(train, horizon)

    merged = pd.merge(
        test[["week_start", "exposures"]],
        fcst[["week_start", "yhat"]],
        on="week_start",
        how="inner",
    )
    if merged.empty:
        return None

    y_true = merged["exposures"].values
    y_pred = merged["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = (np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))).mean() * 100
    avg_actual = np.mean(y_true)
    mae_pct_of_level = (mae / avg_actual) * 100 if avg_actual > 0 else None

    return {
        "horizon": horizon,
        "n_points": len(merged),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "avg_actual": avg_actual,
        "mae_pct_of_level": mae_pct_of_level,
    }



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


def run_model_forecast(model_name, df, horizon, use_weather=False, use_events=False):
    if model_name == "Naive":
        return naive_forecast(df, horizon)

    elif model_name == "Moving Average":
        return moving_average_forecast(df, horizon)

    elif model_name == "Exponential Smoothing":
        return ses_forecast(df, horizon)

    elif model_name == "Holt":
        return holt_forecast(df, horizon)

    elif model_name == "Holt-Winters":
        if len(df) < 2 * 52:
            st.warning("Not enough history for Holt-Winters; using Holt instead.")
            return holt_forecast(df, horizon)
        return holt_winters_forecast(df, horizon)

    elif model_name == "SARIMA":
        base_df = history_with_events.copy()

        if year_filter != "All years":
            selected_year = int(year_filter)
            base_df = base_df[base_df["week_start"].dt.year == selected_year]

        if exclude_2023_from_training:
            base_df = base_df[base_df["week_start"].dt.year >= 2024]

        if base_df.empty or len(base_df) < 30:
            st.error(
                f"SARIMA needs more clean data to fit (have {len(base_df)} weeks after filters). "
                "Try including more years or disabling the 2023 exclusion."
            )
            st.stop()

        with st.expander("Debug SARIMA input"):
            st.write("Rows passed to SARIMA:", len(base_df))
            st.write(base_df[["week_start", "exposures"]].head())
            st.write(base_df[["week_start", "exposures"]].tail())

        return sarima_forecast(
            base_df,
            horizon,
            use_weather=use_weather,
            use_events=use_events,
        )

    elif model_name == "ML (Experimental)":
        return ml_forecast(history_df, horizon)

    else:
        return naive_forecast(df, horizon)


# AI Recommendation Input
#def generate_recommendation(history_df, forecast_df, model_choice, horizon_weeks, metrics=None):
def generate_recommendation(history_df, forecast_df, model_choice, horizon_weeks, ai_provider, metrics=None):
    summary = build_forecast_summary(history_df, forecast_df)

    reliability_text = "Backtest accuracy was not available."
    if metrics is not None:
        reliability_text = (
            f"Backtest over the last {metrics['horizon']} weeks: "
            f"MAE = {metrics['mae']:.1f}, "
            f"RMSE = {metrics['rmse']:.1f}. "
            f"Lower values indicate better forecast accuracy. "
        )

        if metrics["rmse"] > metrics["mae"] * 1.4:
            reliability_text += (
                "RMSE is meaningfully higher than MAE, suggesting some weeks had larger forecast misses. "
            )
        else:
            reliability_text += (
                "RMSE is fairly close to MAE, suggesting forecast errors were relatively stable week to week. "
            )

        if "mae_pct_of_level" in metrics and metrics["mae_pct_of_level"] is not None:
            reliability_text += (
                f"MAE is about {metrics['mae_pct_of_level']:.1f}% of the average weekly observed level. "
            )

    prompt = f"""
You are an EMS operations advisor for a Canadian ambulance service.

Here is a concise summary of recent history and forecast:
{summary}

Model used: {model_choice}
Forecast horizon: {horizon_weeks} weeks.

Forecast reliability from backtesting:
{reliability_text}

Using this information, write a short, practical recommendation
(4–6 sentences) for managers:
- clearly mention the MAE and RMSE values in plain language
  (e.g. 'on average we miss by about X exposures per week')
- if available, mention what MAE as a % of average weekly volume implies
  for planning (e.g. 'typical error is about Y% of normal volume')
- highlight whether RMSE being higher than MAE suggests occasional large misses
- focus on staffing, training, and PPE planning
- be concrete but not alarmist
- explain the forecast confidence in plain language
- if backtest accuracy is weaker, recommend using the forecast more cautiously
- assume the audience is non-technical.
"""

    if ai_provider == "OpenAI":
        if not OPENAI_AVAILABLE:
            return "(OpenAI SDK not installed. Install `openai` and set OPENAI_API_KEY.)"
        response = openai_client.responses.create(
            model=OPENAI_MODEL_NAME,
            input=prompt,
        )
        return response.output_text

    elif ai_provider == "Gemini":
        try:
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
            return response.text
        except Exception:
            return "(Gemini quota exceeded or unavailable. Try OpenAI instead.)"

    elif ai_provider == "Claude":
        return "(Claude integration not implemented yet.)"

    return "(AI recommendation skipped.)"

if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False

if "horizon_weeks" not in st.session_state:
    st.session_state["horizon_weeks"] = 1


# ---------- 5. Main logic ----------
if not st.session_state["run_forecast"]:
    st.info("Set options in the left sidebar, then click **Run forecast**.")
else:
    # Main-page slider above the graph/results
    st.subheader("Forecast controls")

    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        horizon_weeks = st.slider(
            "Forecast horizon (weeks)",
            min_value=1,
            max_value=24,
            value=st.session_state.get("horizon_weeks", 1),
            step=1,
            key="horizon_weeks",
        )

    with control_col2:
        available_years = sorted(history_df["week_start"].dt.year.dropna().unique().tolist())
        year_options = ["All years"] + [str(y) for y in available_years]

        year_filter = st.selectbox(
            "Year",
            options=year_options,
            index=0,
            key="year_filter_main",
        )

    with control_col3:
        ai_model_choice = st.selectbox(
            "AI model used",
            options=[
                "Gemini",
                "OpenAI",
                "None",
            ],
            index=1,  # 0=Gemini, 1=OpenAI
            key="ai_model_choice_main",
        )

    

    # make sure week_start is datetime
    history_df["week_start"] = pd.to_datetime(history_df["week_start"])

    filtered_history_df = history_df.copy()


    if year_filter != "All years":
        selected_year = int(year_filter)
        filtered_history_df = filtered_history_df[
            filtered_history_df["week_start"].dt.year == selected_year
        ]

    sarima_train_df = filtered_history_df.copy()

    if exclude_2023_from_training:
        sarima_train_df = sarima_train_df[
            sarima_train_df["week_start"].dt.year >= 2024
        ]

    if filtered_history_df.empty:
        st.warning("No data available for the selected year.")
        st.stop()

    # Switch on model_choice in the main block
    #if model_choice == "Naive":
    #    forecast_df = naive_forecast(filtered_history_df, horizon_weeks)
    #elif model_choice == "Moving Average":
    #    forecast_df = moving_average_forecast(filtered_history_df, horizon_weeks)
    #elif model_choice == "Exponential Smoothing":
    #    forecast_df = ses_forecast(filtered_history_df, horizon_weeks)
    #elif model_choice == "Holt":
    #    forecast_df = holt_forecast(filtered_history_df, horizon_weeks)
    #elif model_choice == "Holt-Winters" and len(filtered_history_df) < 2 * 52:
    #    st.warning("Not enough history for Holt-Winters; using Holt instead.")
    #    forecast_df = holt_forecast(filtered_history_df, horizon_weeks)
    #elif model_choice == "SARIMA":
        # Start from the weather/events-enriched history
    #    base_df = history_with_events.copy()

        # Optional: limit to display year for consistency
    #    if year_filter != "All years":
    #        selected_year = int(year_filter)
    #        base_df = base_df[base_df["week_start"].dt.year == selected_year]

        # Optional: drop 2023 from training if needed
    #    if exclude_2023_from_training:
    #        base_df = base_df[base_df["week_start"].dt.year >= 2024]

        # Hard guard: need enough points to fit SARIMA
    #    if base_df.empty or len(base_df) < 30:
    #        st.error(
    #            f"SARIMA needs more clean data to fit (have {len(base_df)} weeks after filters). "
    #            "Try including more years or disabling the 2023 exclusion."
    #        )
    #        st.stop()

        # Debug: show what SARIMA is actually seeing
    #    with st.expander("Debug SARIMA input"):
    #        st.write("Rows passed to SARIMA:", len(base_df))
    #        st.write(base_df[["week_start", "exposures"]].head())
    #        st.write(base_df[["week_start", "exposures"]].tail())

    #    forecast_df = sarima_forecast(
    #        base_df,
    #        horizon_weeks,
    #        use_weather=use_weather,
    #        use_events=use_events,
    #    )
    #elif model_choice == "ML (Experimental)":
    #    forecast_df = ml_forecast(history_df, horizon_weeks)
    #else:  # Prophet placeholder
    #    forecast_df = naive_forecast(filtered_history_df, horizon_weeks)
    
    model_choice_1 = model_choice
    model_choice_2 = None

    forecast_df = run_model_forecast(
        model_choice_1,
        filtered_history_df,
        horizon_weeks,
        use_weather=use_weather,
        use_events=use_events,
    )

    #st.subheader("Historical weekly exposures + forecast")

    chart_data = pd.concat([
        filtered_history_df[["week_start", "exposures"]]
            .rename(columns={"exposures": "Historical"})
            .set_index("week_start"),
        forecast_df[["week_start", "yhat"]]
            .rename(columns={"yhat": "Forecast (baseline)"})
            .set_index("week_start"),
    ], axis=1)

    #st.line_chart(chart_data)

    #st.line_chart(chart_data)

    updated_forecast_df = None
    #if show_original_vs_updated:
    #    updated_forecast_df = forecast_df.copy()
    #    updated_forecast_df["yhat"] = updated_forecast_df["yhat"] + 2
    #    updated_forecast_df["yhat_lower"] = updated_forecast_df["yhat_lower"] + 2
    #    updated_forecast_df["yhat_upper"] = updated_forecast_df["yhat_upper"] + 2

    


    # ---------- 5a. Summary metrics ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Last historical week",
            filtered_history_df["week_start"].max().date().isoformat()
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

    plot_df_hist = filtered_history_df[["week_start", "exposures"]].rename(
        columns={"exposures": "value"}
    )
    plot_df_hist["type"] = "Historical"

    plot_df_fcst = forecast_df[["week_start", "yhat"]].rename(
        columns={"yhat": "value"}
    )
    plot_df_fcst["type"] = "Forecast (baseline)"

    plot_df = pd.concat([plot_df_hist, plot_df_fcst], ignore_index=True)

    if plot_df.empty:
        st.warning("No chart data available.")
    else:
        chart_data = plot_df.pivot(
            index="week_start",
            columns="type",
            values="value"
        )
        st.line_chart(chart_data)

    #st.markdown("**Forecast table (baseline)**")
    #table_df = (
    #    forecast_df[["week_start", "yhat", "yhat_lower", "yhat_upper"]]
    #    .rename(columns={
    #        "week_start": "Week start",
    #        "yhat": "Forecast",
    #        "yhat_lower": "Lower",
    #        "yhat_upper": "Upper"
    #    })
    #)
    #st.data_editor(
    #    table_df,
    #    hide_index=True,
    #    disabled=True,
    #)

    # After (wrapped):
    if plot_df.empty:
        st.warning("No chart data available.")
    else:
        chart_data = plot_df.pivot(
            index="week_start",
            columns="type",
            values="value"
        )
    #    st.line_chart(chart_data)

    if show_forecast_table:
        st.markdown("**Forecast table (baseline)**")
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
            disabled=True,
        )

        # ---------- 5a. Backtest accuracy ----------
    metrics = backtest_and_score(filtered_history_df, model_choice, horizon=horizon_weeks)

    if metrics is not None:
        st.subheader(f"Backtest accuracy (last {metrics['horizon']} weeks)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test horizon", f"{metrics['horizon']} weeks")
        c2.metric("Points", str(metrics["n_points"]))
        c3.metric("MAE", f"{metrics['mae']:.1f}")
        c4.metric("RMSE", f"{metrics['rmse']:.1f}")
    else:
        st.info("Not enough history to compute backtest accuracy yet.")


    # ---------- 5c. Original vs updated overlay ----------
    #if show_original_vs_updated and updated_forecast_df is not None:
    #    st.subheader("Original vs updated forecast (illustration)")

    #    comp = pd.DataFrame({
    #        "week_start": forecast_df["week_start"],
    #        "Baseline forecast": forecast_df["yhat"],
    #        "Updated forecast": updated_forecast_df["yhat"]
    #    }).set_index("week_start")

    #    st.line_chart(comp)

    # ---------- 5d. Narrative explanation ----------
     # --- Model comparison controls under subheader ---
    if show_model_comparison:
        st.subheader("Model comparison")

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            model_choice_1 = st.selectbox(
                "Model 1",
                [
                    "Prophet (default)",
                    "Naive",
                    "Moving Average",
                    "Exponential Smoothing",
                    "Holt",
                    "Holt-Winters",
                    "SARIMA",
                ],
                index=0,
                key="model_choice_1",
            )
        with comp_col2:
            model_choice_2 = st.selectbox(
                "Model 2",
                [
                    "Prophet (default)",
                    "Naive",
                    "Moving Average",
                    "Exponential Smoothing",
                    "Holt",
                    "Holt-Winters",
                    "SARIMA",
                ],
                index=1,
                key="model_choice_2",
            )
    else:
        # Fallback: use sidebar model as Model 1, no Model 2
        model_choice_1 = model_choice
        model_choice_2 = None
    
    
    if show_model_comparison and model_choice_2 is not None:
        #st.subheader("Model comparison")

        fcst1 = forecast_df.copy()
        fcst2 = run_model_forecast(
            model_choice_2,
            filtered_history_df,
            horizon_weeks,
            use_weather=use_weather,
            use_events=use_events,
        )

        hist_comp = filtered_history_df[["week_start", "exposures"]].rename(
            columns={"exposures": "Historical"}
        ).set_index("week_start")

        model1_comp = fcst1[["week_start", "yhat"]].rename(
            columns={"yhat": f"{model_choice_1}"}
        ).set_index("week_start")

        model2_comp = fcst2[["week_start", "yhat"]].rename(
            columns={"yhat": f"{model_choice_2}"}
        ).set_index("week_start")

        comp = pd.concat([hist_comp, model1_comp, model2_comp], axis=1)

        st.line_chart(comp)
    
    
    
    st.subheader("Interpretation (for managers)")

    next4 = forecast_df.head(4)
    avg_next4 = next4["yhat"].mean()
    avg_lower = next4["yhat_lower"].mean()
    avg_upper = next4["yhat_upper"].mean()

    text = (
        f"Over the next 4 weeks, the model expects about **{avg_next4:.1f}** "
        f"work-related exposures per week "
        f"(roughly {avg_lower:.0f} to {avg_upper:.0f}). "
    )

    if use_events:
        event_start = base_for_events["week_start"].min()
        event_end = base_for_events["week_start"].max() + pd.Timedelta(weeks=horizon_weeks)

        st.write("Ticketmaster request window:", event_start, "to", event_end)

        events_weekly = fetch_ticketmaster_events(event_start, event_end)

    if use_weather and not history_with_weather.empty:
        corr_rain = history_with_weather["exposures"].corr(
            history_with_weather["precip"]
        )
        if pd.notna(corr_rain):
            text += (
                f"Historically, weeks with more rain have a correlation of "
                f"{corr_rain:.2f} with exposure counts, so wet weeks may need "
                f"a bit more staffing and PPE buffer. "
            )
        else:
            text += (
                "Historically, there is not enough data to estimate how rain "
                "relates to exposure counts yet. "
            )

    st.write(text)

    st.markdown("**AI-generated operational recommendation**")

    if use_ai_recommendation:
        with st.spinner("Generating manager recommendation..."):
            try:
                ai_reco = generate_recommendation(
                    history_df,
                    forecast_df,
                    model_choice,
                    horizon_weeks,
                    ai_model_choice,
                    metrics=metrics,
                )
            except Exception as e:
                ai_reco = f"(Error generating recommendation: {e})"
    else:
        ai_reco = "(AI recommendation skipped.)"

    st.write(ai_reco)


    st.caption(
        "MAE = average absolute error in weekly exposures. "
        "MAPE = average error as a percentage of actual volume."
    )

    #with st.expander("Debug exposure load"):
    #    st.write(debug_info)
    #    st.write(history_df.head())
    #    st.write(history_df.tail())