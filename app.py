import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

import plotly.express as px
import plotly.graph_objects as go

# Optional forecasting (statsmodels). Falls back gracefully if not installed.
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


APP_TITLE = "CO₂ Analytics: ML Regression + Forecasting"
DATA_HINT = "Country-year CO₂ dataset (Our World in Data-like)."


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def country_only(df: pd.DataFrame) -> pd.DataFrame:
    if "iso_code" in df.columns:
        return df[df["iso_code"].notna()].copy()
    return df.copy()


def build_panel_dataset(df: pd.DataFrame, target: str, min_year: int, max_year: int) -> pd.DataFrame:
    d = country_only(df)
    d = d[(d["year"] >= min_year) & (d["year"] <= max_year)].copy()
    d = d.dropna(subset=[target, "year", "iso_code", "Name"])

    # Engineered features
    if "gdp" in d.columns and "population" in d.columns:
        d["gdp_per_capita"] = d["gdp"] / d["population"]
    if "co2" in d.columns and "population" in d.columns:
        d["co2_per_capita_from_total"] = d["co2"] / d["population"]

    return d


def make_model(model_name: str) -> Pipeline:
    categorical = ["iso_code"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, []),
        ("cat", categorical_transformer, categorical),
    ],
    remainder="drop",
    sparse_threshold=0.0,   # force dense output
)

    if model_name == "Ridge (Linear)":
        estimator = Ridge(alpha=1.0, random_state=0)
    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=400,
            random_state=0,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2
        )
    else:
        estimator = HistGradientBoostingRegressor(
            random_state=0,
            max_depth=6,
            learning_rate=0.08,
            max_iter=600
        )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def set_numeric_columns(pipe: Pipeline, df: pd.DataFrame, target: str) -> Pipeline:
    num_cols = [
        c for c in df.columns
        if c not in ["Name", target, "iso_code"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    ct: ColumnTransformer = pipe.named_steps["preprocess"]
    new_transformers = []
    for name, trans, cols in ct.transformers:
        if name == "num":
            new_transformers.append((name, trans, num_cols))
        else:
            new_transformers.append((name, trans, cols))
    ct.transformers = new_transformers
    ct._validate_transformers()
    return pipe


def time_split(df: pd.DataFrame, test_years: int = 5):
    years = sorted(df["year"].dropna().unique().tolist())
    if len(years) <= test_years + 2:
        return train_test_split(df.index, test_size=0.2, random_state=0)
    cutoff = years[-test_years]
    train_idx = df[df["year"] < cutoff].index
    test_idx = df[df["year"] >= cutoff].index
    return train_idx, test_idx


def eval_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def forecast_country(series: pd.Series, horizon: int):
    s = series.dropna()
    if len(s) < 6:
        x = np.arange(len(s))
        coef = np.polyfit(x, s.values.astype(float), 1)
        x_future = np.arange(len(s), len(s) + horizon)
        return coef[0] * x_future + coef[1]

    if _HAS_STATSMODELS:
        try:
            model = ExponentialSmoothing(
                s.values.astype(float),
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            return fit.forecast(horizon)
        except Exception:
            pass

    x = np.arange(len(s))
    coef = np.polyfit(x, s.values.astype(float), 1)
    x_future = np.arange(len(s), len(s) + horizon)
    return coef[0] * x_future + coef[1]


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(DATA_HINT)

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CO₂_Data.csv (optional)", type=["csv"])
        st.write("If you don't upload a file, the bundled dataset will be used (if present).")

        st.header("Modeling")
        target = st.selectbox(
            "Target",
            options=["co2_per_capita", "co2", "consumption_co2", "co2_per_gdp"],
            index=0
        )
        model_name = st.selectbox("Regression model", ["HistGradientBoosting", "Random Forest", "Ridge (Linear)"], index=0)
        test_years = st.slider("Holdout (last N years for test)", min_value=3, max_value=15, value=7)

        st.header("Forecast")
        horizon = st.slider("Forecast horizon (years)", min_value=3, max_value=30, value=10)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        try:
            df = load_data("CO₂_Data.csv")
        except Exception:
            # Some systems dislike the unicode filename; try ASCII fallback
            df = load_data("CO2_Data.csv")

    # Year bounds
    year_min = int(pd.to_numeric(df["year"], errors="coerce").min())
    year_max = int(pd.to_numeric(df["year"], errors="coerce").max())

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        min_year = st.slider("Training start year", min_value=year_min, max_value=year_max, value=max(year_min, 1990))
    with c2:
        max_year = st.slider("Training end year", min_value=year_min, max_value=year_max, value=year_max)
    with c3:
        st.metric("Rows (raw)", f"{len(df):,}")
        st.metric("Countries (iso)", f"{df['iso_code'].dropna().nunique():,}" if "iso_code" in df.columns else "—")

    if target not in df.columns:
        st.error(f"Target column '{target}' not found.")
        return

    panel = build_panel_dataset(df, target=target, min_year=min_year, max_year=max_year)
    if panel.empty:
        st.warning("No rows remain after filtering. Try widening years or changing target.")
        return

    countries = panel.sort_values("Name")["Name"].unique().tolist()
    default_country = "Egypt" if "Egypt" in countries else countries[0]
    selected_country = st.selectbox("Country for detailed view", countries, index=countries.index(default_country))

    train_idx, test_idx = time_split(panel, test_years=test_years)
    train = panel.loc[train_idx].copy()
    test = panel.loc[test_idx].copy()

    X_train = train.drop(columns=["Name", target])
    y_train = train[target].astype(float)
    X_test = test.drop(columns=["Name", target])
    y_test = test[target].astype(float)

    pipe = make_model(model_name)
    pipe = set_numeric_columns(pipe, panel, target=target)

    with st.spinner("Training regression model..."):
        pipe.fit(X_train, y_train)

    yhat_test = pipe.predict(X_test)
    metrics = eval_metrics(y_test, yhat_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m3.metric("R²", f"{metrics['R²']:.3f}")

    pred_df = test[["Name", "iso_code", "year", target]].copy()
    pred_df["pred"] = yhat_test
    pred_df["residual"] = pred_df[target] - pred_df["pred"]
    pred_df = pred_df.sort_values(["Name", "year"])

    with st.expander("Test set predictions (downloadable)", expanded=False):
        st.dataframe(pred_df.head(2000), use_container_width=True)
        buf = io.StringIO()
        pred_df.to_csv(buf, index=False)
        st.download_button("Download predictions CSV", data=buf.getvalue(), file_name=f"pred_{target}.csv", mime="text/csv")

    st.subheader("1) Global panel: Actual vs Predicted (test period)")
    fig_sc = px.scatter(
        pred_df,
        x=target,
        y="pred",
        hover_data=["Name", "year"],
        trendline="ols",
        opacity=0.6,
        labels={target: "Actual", "pred": "Predicted"},
    )
    fig_sc.update_layout(height=520)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader(f"2) {selected_country}: Historical series + ML predictions")
    country_hist = panel[panel["Name"] == selected_country].sort_values("year").copy()
    X_all = country_hist.drop(columns=["Name", target])
    country_hist["pred"] = pipe.predict(X_all)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=country_hist["year"], y=country_hist[target], mode="lines+markers", name="Actual"))
    fig_ts.add_trace(go.Scatter(x=country_hist["year"], y=country_hist["pred"], mode="lines", name="ML Pred"))
    fig_ts.update_layout(height=520, xaxis_title="Year", yaxis_title=target)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader(f"3) {selected_country}: Forecast next {horizon} years (univariate)")
    y_fore = forecast_country(country_hist[target], horizon=horizon)
    last_year = int(country_hist["year"].max())
    years_fore = np.arange(last_year + 1, last_year + 1 + horizon)

    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=country_hist["year"], y=country_hist[target], mode="lines+markers", name="Actual"))
    fig_f.add_trace(go.Scatter(x=years_fore, y=y_fore, mode="lines+markers", name="Forecast"))
    fig_f.update_layout(height=520, xaxis_title="Year", yaxis_title=target)
    st.plotly_chart(fig_f, use_container_width=True)

    st.info(
    """Notes:
- Regression is a **panel model** (all countries + years) with `iso_code` encoded.
- Forecast is a **country-only univariate** model (Exponential Smoothing if available; otherwise linear trend)."""
)


if __name__ == "__main__":
    main()
