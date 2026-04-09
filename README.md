# EMS Work-Related Exposure Forecasting – Riipen Project
Forecasting weekly work‑related exposure events among Emergency Medical Services (EMS) clinicians using a cleaned 2024 cohort derived from the NEMSIS public‑use dataset and local extracts. The project delivers a practical product for EMS managers: a forecasting dashboard that highlights higher‑risk weeks in advance so they can plan staffing and safety interventions before exposures spike.


## Project / Product Description
This repository contains the code and assets for my Douglas College CSIS 4495 Riipen project with Solaris Canada. The final prototype focuses on a single, well‑defined outcome and an interpretable, operationally useful forecasting workflow.

### Outcome: 
Work‑related exposure events among EMS clinicians (for example, blood and body‑fluid exposures), identified using the NEMSIS work‑related exposure flag and restricted to “Yes” cases only.

The project forecasts weekly counts of exposure‑flagged events rather than attempting to model detailed injury mechanisms or severity, which are not consistently available in the public research data.

### Data layer:
- A SQL Server pipeline reads large NEMSIS v3 tables and local FACT/PCR tables, filters to exposure‑positive PCRs, and handles Not Applicable / Not Recorded codes.
- Operational time fields from the FACT/PCR layer are used as the primary time base for exposure events, after determining that hospital outcome times (e.g., eOutcome11) are largely unusable for this cohort.
- The pipeline aggregates exposure events per calendar day and then resamples to a Monday‑based weekly series of exposure counts.
- The final app reads from a compact CSV export (for example, PCR_Exposure_Final.csv), avoiding heavy runtime joins and making the Streamlit app lightweight and portable.

### Scope decisions & constraints:
- The NEMSIS public‑use data and local extracts do not provide complete, reliable provider injury‑mechanism or hospital‑outcome detail at scale, so the project models when exposures occur, not the exact cause or long‑term outcome.
- To keep the project feasible within the course timeline, the scope narrows to:
  #### - One year of exposure‑flagged events (2024) with a consistent time field.
  #### - A single weekly exposure count time series.
  #### - A small, interpretable set of time‑series models plus explicit forecast‑accuracy checks
- Earlier ideas such as multi‑year demand forecasting, disabling‑injury prediction, and geospatial “hot‑spot” analysis were dropped in favor of delivering a complete, end‑to‑end exposure‑forecasting product.

### Model layer:
- The app exposes several interpretable time‑series models rather than a single black‑box ML pipeline. Available methods include:
  #### - Naive baseline.
  #### - Moving‑average forecast.
  #### - Simple Exponential Smoothing.
  #### - Holt trend model.
  #### - Holt‑Winters (trend + seasonality; runs only when there is enough history).
  #### - SARIMA, with optional weather and major‑event covariates.
- For each selected model and forecast horizon, the app produces:
  #### - Weekly point forecasts (yhat).
  #### - Simple or model‑based upper/lower bounds (yhat_lower, yhat_upper).
- A rolling backtest routine holds out recent weeks, refits the model on earlier data, and reports:
  #### - Mean Absolute Error (MAE).
  #### - Root Mean Squared Error (RMSE).
  #### - MAE as a percentage of the average weekly exposure level.

### Application layer (product):

- The core product is a Streamlit web app called EMS Work‑Related Injury / Exposure Forecaster that EMS leaders can open in a browser and use as a decision‑support tool.

### Key features:

- Sidebar controls:
  Model selector for choosing among Naive, Moving Average, Exponential Smoothing, Holt, Holt‑Winters, SARIMA, and (optionally) an experimental ML model.
- Options checkboxes to:
#### - Show the forecast table.
#### - Show model comparison.
#### - Generate an AI manager recommendation (Planning Assistant, PAi).
#### - Include weather effects (rain/snow) via Open‑Meteo.
#### - Include major events (concerts, rallies, holidays) via the Ticketmaster Discovery API.
#### - Exclude 2023 from SARIMA training when desired.
- Run forecast button to execute the selected configuration.
  
- Main view:
  A Forecast controls panel with:
#### - Forecast horizon slider (1–24 weeks).
#### - Year filter (e.g., “All years” or a specific year).
#### - AI provider selector (Gemini, OpenAI, or None) for PAi.
  A summary panel showing the last historical week, selected model, forecast horizon, and key context.

  An interactive line chart displaying:
#### - Historical weekly exposure counts.
#### - Forward forecasts with optional uncertainty bands and overlays.
  
  An optional forecast table listing week start dates, point forecasts, and bounds.
  A Backtest accuracy section summarizing MAE, RMSE, MAPE, and MAE as a percentage of the average weekly level.
  An optional Ticketmaster events table when major‑event enrichment is enabled and data are available.

### AI Planning Assistant (PAi)
- The app integrates an AI Planning Assistant (PAi) that turns numeric forecasts and error metrics into short, manager‑friendly recommendations.
- PAi uses Google’s Gemini API (and optionally OpenAI) with a structured prompt that includes:
#### - A recent 8‑week exposure summary.
#### - The next 4 weeks of forecasts.
#### - Backtest metrics (MAE, RMSE, MAE % of level).
#### - The chosen model and horizon.

- It generates a 4–6 sentence narrative that:
#### - Explains forecast levels and typical error in plain language.
#### - Comments on reliability (e.g., whether RMSE is much higher than MAE).
#### - Suggests practical actions related to staffing, training, and PPE planning.
#### - Encourages cautious use when error is high.

Overall, the project delivers a pragmatic forecasting product that uses routinely collected exposure data, simple time‑series methods, explicit accuracy feedback, and an AI narrative layer to support proactive injury‑prevention decisions in EMS operations.

## Installation
These instructions explain how to set up the repository locally and run the Streamlit app.

### 1. Clone the repository
- git clone https://github.com/dondemici/W26_4495_S2_DundeeA.git
- cd W26_4495_S2_DundeeA

### 2. Create and activate a virtual environment (recommended)
Using venv:
- python -m venv .venv
#### Windows
- .venv\Scripts\activate
#### macOS/Linux
- source .venv/bin/activate

### 3. Install Python dependencies
Install the core packages manually:
- pip install streamlit pandas numpy statsmodels sqlalchemy pyodbc scikit-learn requests google-genai openai
If you use Prophet or other optional libraries, install them separately and update the app configuration as needed.

## Usage – Run the EMS Forecaster
Run the Streamlit app locally to see the end‑to‑end forecasting workflow.

### From the repository root, locate the Streamlit app file, for example:Implementation/Product/forecaster_app.py
(or your final filename, such as ems_forecaster_app.py).

### Run the Streamlit app:
- streamlit run Implementation/Product/forecaster_app.py
  
### Open the provided local URL (usually http://localhost:8501) in your browser.

### In the app:
- Choose a model (e.g., Naive, Moving Average, Exponential Smoothing, Holt, Holt‑Winters, SARIMA, or the experimental ML option).
- In the sidebar, select which options to show (forecast table, model comparison, AI recommendation, weather effects, major events).
- Click Run forecast.
- In the main panel:
#### - Adjust the forecast horizon (weeks).
#### - Optionally filter by year and choose the AI provider for PAi.
#### - Review the historical vs forecast chart and the forecast table.
#### - Review backtest metrics (MAE, RMSE, MAPE, MAE % of level).
#### - Read the PAi narrative if the AI recommendation option is enabled.

### To see the Original vs Updated forecast behavior, enable the checkbox in the sidebar and run the forecast again. The app will show how an updated series (with higher recent exposure counts) shifts the forecast upward, illustrating how the production system would respond to new weekly data.

## Project Team
- Student: Dundee Adriatico
- Student ID: 300393449
- Email: adriaticom@student.douglascollege.ca

Company Details (Riipen)

- Partner: Solaris Canada
- Contact: Tony Tsui, CEO – tonytsui.solaris@gmail.com
- Scope: Data analysis and forecasting for EMS operations using the public NEMSIS dataset.
