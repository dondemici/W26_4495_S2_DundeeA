# EMS Work-Related Exposure Forecasting – Riipen Project
Predicting weekly work-related exposure events among Emergency Medical Services (EMS) clinicians using the NEMSIS Public-Use Dataset. The project focuses on building a product for EMS managers: a forecasting tool that surfaces high‑risk weeks in advance so they can plan staffing and safety interventions before exposures spike.


## Project / Product Description
This repository contains the code and assets for my Douglas College CSIS 4495 Riipen project with Solaris Canada. The current midterm prototype focuses on:

### Outcome: Work‑related exposure events among EMS clinicians (e.g., blood and body fluid exposures), identified from NEMSIS work‑related exposure coding and filtered to “Yes” cases only.

### Data layer:

SQL Server pipeline reading large NEMSIS v3 tables (e.g., PCR_Events, FACTPCRWorkRelatedExposure, FACTPCR time data).

Staging tables used to query tens of millions of records efficiently from Jupyter.

Cleaning and exploration of January 2024 (~90K rows) to validate exposure fields and coding.

Identification that hospital admit time (eOutcome11) is not usable for exposure cases (mostly “Not Applicable”), leading to the adoption of FACTPCR operational times as the main time source.

In‑progress construction of a 2024 weekly work‑related exposure time series (~20K exposure events expected).

### Scope decisions & constraints:

NEMSIS public data does not include detailed provider injury mechanisms at the national level; this information is only available in some state‑level systems, so the project models when exposures occur, not the exact cause.

Midterm prototype uses a synthetic weekly baseline series that mimics realistic exposure patterns while the full PCR‑time join and aggregation are completed.

### Model layer:

Model‑agnostic design: the app can plug in SARIMA, Prophet, or simpler baseline forecasters depending on signal strength in the final weekly series.

Current midterm demo runs on a placeholder forecaster over the synthetic weekly series, producing:

Point forecasts for future weekly exposure counts.

Simple lower/upper bounds for each week.

### Application layer (product):

A Streamlit web app called EMS Work‑Related Exposure Forecaster that EMS leaders can open in a browser and use as if it were a production tool.

### Key features:

Sidebar controls:

Model selector (Prophet default, SARIMA option).

Forecast horizon slider (weeks ahead).

Toggles for Major events and Weather effects (wired to synthetic placeholders now; planned integration with real APIs later).

Option to show Original vs Updated forecast to illustrate how new data shifts predictions.

Main view:

Summary panel (last historical week, selected model, forecast horizon).

Chart of historical weekly exposures plus forecast and uncertainty band.

Table of forecasted values with lower/upper bounds.

Optional chart comparing original vs updated forecast.

Narrative explanation written for non‑technical EMS managers.

Overall, the project is moving towards a practical forecasting product that supports proactive injury‑prevention decisions in EMS operations, leveraging the large NEMSIS public‑release dataset.

## Installation
These instructions explain how to set up the repository locally and run the Streamlit demo.

### 1. Clone the repository
bash
git clone https://github.com/dondemici/W26_4495_S2_DundeeA.git
cd W26_4495_S2_DundeeA

### 2. Create and activate a virtual environment (recommended)
Using venv:

bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

### 3. Install Python dependencies
If you have a requirements.txt in the repo (recommended), run:

bash
pip install -r requirements.txt
If not, install the core packages manually (adjust as needed for the repo):

bash
pip install streamlit pandas numpy plotly prophet statsmodels sqlalchemy pyodbc

## Usage – See a Demo of the Product
You can run the Streamlit prototype locally to see the forecasting workflow end‑to‑end.

From the repository root, locate the Streamlit app file (for example):

app.py or

streamlit_app.py
(Use the actual filename from your repo.)

Run the Streamlit app:

bash
streamlit run app.py
Open the provided local URL (usually http://localhost:8501) in your browser.

In the app:

Select the model (Prophet default or SARIMA).

Choose the forecast horizon in weeks.

Optionally toggle Major events and Weather effects (currently placeholder / synthetic integrations).

Click Run forecast to generate:

Historical vs forecast chart.

Forecast table with bounds.

Narrative explanation.

To see the Original vs Updated forecast behavior, enable the checkbox in the sidebar and run the forecast again. The app will show how an updated series (with higher recent exposure counts) shifts the forecast upward, illustrating how the production system would respond to new weekly data.

## Reproducing Data Pipeline Work (Optional)
If you have access to the NEMSIS Public‑Release Research Dataset and a SQL Server environment:

Use the SQL scripts in the sql/ or notebooks/ folders (if present) to:

Create staging tables for PCR_Events, FACTPCRWorkRelatedExposure, and related FACTPCR time tables.

Run joins that filter to work‑related exposure = “Yes” and exclude “Not Applicable” / “Not Recorded”.

Export or query the result from Jupyter notebooks to build the weekly time series.

Note: Due to NEMSIS data distribution policies, this repository does not include raw NEMSIS data files; researchers must request access directly from NEMSIS.

## Project Team
Student: Dundee Adriatico
Student ID: 300393449
Email: adriaticom@student.douglascollege.ca

Company Details (Riipen)
Partner: Solaris Canada
Contact: Tony Tsui, CEO – tonytsui.solaris@gmail.com

Scope: Data analysis and forecasting for EMS operations using the public NEMSIS dataset.
