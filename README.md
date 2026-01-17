# CO₂ Analytics (ML Regression + Forecasting) — Streamlit App

## What this app does
1) **Panel ML Regression** (all countries + years)
- Predicts selected targets: `co2_per_capita`, `co2`, `consumption_co2`, `co2_per_gdp`
- Uses numeric features (population, energy shares, sector CO₂, etc.) + `iso_code` as a categorical identifier.

2) **Country Forecasting**
- Univariate forecast for the selected country and target.
- Uses **Exponential Smoothing** if `statsmodels` is installed; otherwise falls back to a robust linear trend.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- Push these files to a GitHub repo
- Set **app.py** as the main file
- The app will use the bundled **CO₂_Data.csv**
- Or upload your CSV from the sidebar

## Notes
- Aggregates like "World" are filtered out automatically using `iso_code`.
