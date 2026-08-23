# Forecasting HDFC Bank Volatility: GARCH vs. XGBoost

**Author:** Udit Chauhan

A comparison of a classical econometric volatility model (GARCH) against a gradient-boosted tree model (XGBoost) for forecasting stock return volatility, using **HDFC Bank Limited** (NSE: `HDFCBANK`) as the case study.

## What this notebook does

Volatility forecasting matters for option pricing, portfolio risk management, and position sizing. This notebook:

1. Pulls ~15 years of daily HDFC Bank price history **live from Yahoo Finance** via `yfinance` (no static CSV — the data refreshes every time you run it).
2. Computes daily returns and explores volatility clustering (ACF/PACF diagnostics, return distribution, rolling volatility).
3. Fits a **GARCH(4,4)** model with a generalized error distribution and produces a rolling, one-step-ahead out-of-sample forecast.
4. Trains an **XGBoost regressor** on calendar features (day of week, month, quarter, etc.) plus lagged rolling volatility, and forecasts the same out-of-sample window.
5. Evaluates both models against **22-day rolling realised volatility** using RMSE and MAPE, in-sample and out-of-sample, and compares results side by side.

## Files

| File | Description |
|---|---|
| `GARCH_vs_XGBoost_Volatility_Forecasting.ipynb` | Main analysis notebook |
| `README.md`|

## Requirements

- Python 3.10+
- Internet access (the notebook downloads data from Yahoo Finance at run time — it will not work in a fully offline/sandboxed environment)

### Install dependencies

```bash
pip install numpy pandas plotly matplotlib seaborn xgboost arch statsmodels tabulate scikit-learn yfinance
```

## How to run

1. Open `GARCH_vs_XGBoost_Volatility_Forecasting.ipynb` in Jupyter, JupyterLab, VS Code, or Google Colab.
2. Run all cells top to bottom (`Run All`). No manual data upload is needed — the notebook fetches `HDFCBANK.NS` price history itself via `yfinance`.
3. Since the data is pulled live, exact metric values (RMSE/MAPE) and the last few rows of data will shift slightly each time you re-run it as new trading days become available.

## Structure

1. **Introduction** — overview, methodology, and the models being compared
2. **Library Setup** — imports
3. **Loading & Cleaning HDFC Bank Data** — live `yfinance` download, cleaning, return calculation
4. **Exploratory Analysis** — return distribution, volatility stats, ACF/PACF diagnostics
5. **Modelling** — GARCH(4,4) fit + rolling forecast; XGBoost feature engineering, training, forecast
6. **Results & Comparison** — RMSE/MAPE table, in-sample vs. out-of-sample, GARCH vs. XGBoost

## Key parameters

- **Lookback window:** trailing 15 years of daily data
- **Test window:** most recent 251 trading days (~1 trading year), held out from training
- **Volatility target:** 22-day (≈1 trading month) rolling standard deviation of daily returns
- **GARCH spec:** GARCH(4,4), generalized error distribution
- **XGBoost features:** day of week, quarter, month, year, day of year, day of month, plus 4 lags of rolling volatility

