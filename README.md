# MEPS Inpatient Risk Prediction

Streamlit dashboard for panel-validated prediction of Year-2 inpatient admissions using Year-1 MEPS features.

## What the app shows

- multiclass target: `0 / 1 / 2+` Year-2 inpatient admissions
- panel-based validation: train on earlier panels, test on the latest panel
- model comparison across `CatBoost`, `HGB`, `GB`, and `RF`
- class-2 ranking diagnostics: PR curve, calibration, and lift by decile
- export-ready high-risk cohort tables

## Files used for deployment

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `data/processed/meps_group6_analysis_ready.parquet`
- `data/processed/meps_group6_analysis_ready_events.parquet`

## Local run

```bash
streamlit run app.py
```

## Deployment target

Deploy `app.py` from the repository root on Streamlit Community Cloud.
