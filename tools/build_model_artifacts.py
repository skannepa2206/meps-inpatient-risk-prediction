from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier, __version__ as catboost_version
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_EVENTS = BASE_DIR / "data" / "processed" / "meps_group6_analysis_ready_events.parquet"
DATA_PATH_BASE = BASE_DIR / "data" / "processed" / "meps_group6_analysis_ready.parquet"
DATA_PATH = DATA_PATH_EVENTS if DATA_PATH_EVENTS.exists() else DATA_PATH_BASE
ARTIFACT_DIR = BASE_DIR / "artifacts" / "models"
ARTIFACT_FEATURES_PATH = ARTIFACT_DIR / "feature_columns.json"
ARTIFACT_CATS_PATH = ARTIFACT_DIR / "categorical_columns.json"

NEG_CODES = [-1, -7, -8, -9]
RANDOM_STATE = 42


def select_features(df, allowed_prefixes, exclude_cols, include_suffixes=("_y1",)):
    cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in allowed_prefixes) or any(c.endswith(s) for s in include_suffixes)
    ]
    cols = [c for c in cols if "Y2" not in c]
    cols = [c for c in cols if c not in exclude_cols]
    if "AGE_MAX" in df.columns and "AGE_MAX" not in cols:
        cols.append("AGE_MAX")
    return cols


def build_dataset(df):
    df = df.copy()
    df = df.replace(NEG_CODES, np.nan)
    target_raw = "IPDISY2"
    df[target_raw] = df[target_raw].replace(NEG_CODES, np.nan)
    df = df.dropna(subset=[target_raw]).copy()

    def make_target_class(value):
        if pd.isna(value):
            return np.nan
        if value == 0:
            return 0
        if value == 1:
            return 1
        return 2

    df["Y_IPDISY2_CLASS"] = df[target_raw].apply(make_target_class).astype(int)

    allowed = (
        "AGE", "SEX", "RACE", "HISP",
        "EDUC", "POVCAT", "INSUR",
        "ASTH", "DIAB", "ARTH", "HYPER", "CHRON",
        "RTHLTH", "MNHLTH",
        "ERDISY1", "IPDISY1",
        "TOTEXPY1", "RXEXPY1",
    )
    exclude = {
        "DUPERSID", "SOURCE_PANEL", "CANCERY1", "CANCERY2",
        "IPDISY2", "Y_IPDISY2_CLASS",
    }

    X = df[select_features(df, allowed, exclude)].copy()
    missing_rate = X.isna().mean()
    high_missing = missing_rate[missing_rate > 0.5].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)

    if "AGE_MAX" in X.columns:
        age_cols_to_drop = [c for c in X.columns if c.startswith("AGE") and c != "AGE_MAX"]
        X = X.drop(columns=age_cols_to_drop)

    y = df["Y_IPDISY2_CLASS"].astype(int)
    panels = df["PANEL"].astype(int)
    return X, y, panels


def identify_categorical_columns(X):
    cost_count_cols = [c for c in X.columns if c.endswith("_cost_y1") or c.endswith("_count_y1")]
    numeric_force = set(cost_count_cols + ["AGE_MAX", "TOTEXPY1", "RXEXPY1", "IPDISY1", "ERDISY1"])
    cat_cols = []
    for c in X.columns:
        if c in numeric_force:
            continue
        if X[c].dtype == "object":
            cat_cols.append(c)
        elif pd.api.types.is_integer_dtype(X[c]) and X[c].nunique(dropna=True) <= 50:
            cat_cols.append(c)
        elif pd.api.types.is_float_dtype(X[c]):
            vals = X[c].dropna()
            if len(vals) > 0 and (vals % 1 == 0).all() and vals.nunique() <= 50:
                cat_cols.append(c)
    return cat_cols


def prepare_baseline_frames(X_train, X_test, cat_cols):
    X_train_base = X_train.copy()
    X_test_base = X_test.copy()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    fill_values = {}
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train_base[num_cols] = num_imputer.fit_transform(X_train_base[num_cols])
        X_test_base[num_cols] = num_imputer.transform(X_test_base[num_cols])
        fill_values = {col: float(val) for col, val in zip(num_cols, num_imputer.statistics_)}

    if cat_cols:
        for col in cat_cols:
            X_train_base[col] = X_train_base[col].astype("string").fillna("NA")
            X_test_base[col] = X_test_base[col].astype("string").fillna("NA")
        X_train_base = pd.get_dummies(X_train_base, columns=cat_cols, dummy_na=False)
        X_test_base = pd.get_dummies(X_test_base, columns=cat_cols, dummy_na=False)
        X_train_base, X_test_base = X_train_base.align(X_test_base, join="left", axis=1, fill_value=0)

    bundle = {
        "cat_cols": cat_cols,
        "num_fill_values": fill_values,
        "feature_columns": X_train_base.columns.tolist(),
    }
    return X_train_base, X_test_base, bundle


def prepare_catboost_frames(X_train, X_test, cat_cols):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    fill_values = {}
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train_cb[num_cols] = num_imputer.fit_transform(X_train_cb[num_cols])
        X_test_cb[num_cols] = num_imputer.transform(X_test_cb[num_cols])
        fill_values = {col: float(val) for col, val in zip(num_cols, num_imputer.statistics_)}

    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("NA")
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("NA")

    cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]
    bundle = {
        "cat_cols": cat_cols,
        "num_fill_values": fill_values,
        "feature_order": X_train_cb.columns.tolist(),
    }
    return X_train_cb, X_test_cb, cat_idx, bundle


def compute_lift_table(y_true_binary, scores):
    score_df = pd.DataFrame({"y_true": y_true_binary, "score": scores})
    score_df["decile"] = pd.qcut(score_df["score"], 10, labels=False, duplicates="drop")
    lift_table = score_df.groupby("decile").agg(n=("y_true", "size"), positives=("y_true", "sum")).reset_index()
    lift_table["rate"] = lift_table["positives"] / lift_table["n"]
    base_rate = score_df["y_true"].mean()
    lift_table["lift"] = lift_table["rate"] / base_rate if base_rate > 0 else 0.0
    lift_table = lift_table.sort_values("decile", ascending=False).reset_index(drop=True)
    return lift_table, float(base_rate)


def metric_row(name, y_test, y_pred, y_proba):
    y_test_bin = (y_test == 2).astype(int)
    lift_table, _ = compute_lift_table(y_test_bin, y_proba[:, 2])
    top_decile_lift = float(lift_table.loc[0, "lift"]) if not lift_table.empty else np.nan
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (2+)": recall_score(y_test_bin, (y_pred == 2).astype(int), zero_division=0),
        "PR-AUC (2+)": average_precision_score(y_test_bin, y_proba[:, 2]),
        "Top-Decile Lift": top_decile_lift,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    X, y, panels = build_dataset(df)
    feature_cols = X.columns.tolist()
    cat_cols = identify_categorical_columns(X)

    unique_panels = sorted(int(p) for p in panels.dropna().unique())
    holdout_panel = max(unique_panels)
    train_panels = [p for p in unique_panels if p != holdout_panel]
    train_idx = panels.isin(train_panels)
    test_idx = panels.isin([holdout_panel])

    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    X_train_base, X_test_base, base_bundle = prepare_baseline_frames(X_train, X_test, cat_cols)
    X_train_cb, X_test_cb, cat_idx, catboost_bundle = prepare_catboost_frames(X_train, X_test, cat_cols)

    class_counts = y_train.value_counts().sort_index()
    class_weights = (class_counts.sum() / class_counts).values.tolist()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    models = {
        "GB": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=RANDOM_STATE,
            verbose=False,
            class_weights=class_weights,
            thread_count=-1,
        ),
        "HGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=6),
        "RF": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    metrics = []
    saved_models = []

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "CatBoost":
            model.fit(X_train_cb, y_train, cat_features=cat_idx)
            y_pred = np.asarray(model.predict(X_test_cb)).astype(int).flatten()
            y_proba = model.predict_proba(X_test_cb)
            model.save_model(str(ARTIFACT_DIR / "catboost_model.cbm"))
            bundle = dict(catboost_bundle)
            bundle["artifact_type"] = "catboost_native"
            joblib.dump(bundle, ARTIFACT_DIR / "catboost_bundle.joblib", compress=3)
        else:
            model.fit(X_train_base, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_test_base)
            y_proba = model.predict_proba(X_test_base)
            bundle = dict(base_bundle)
            bundle["artifact_type"] = "sklearn_bundle"
            bundle["model"] = model
            joblib.dump(bundle, ARTIFACT_DIR / f"{name.lower()}_bundle.joblib", compress=3)

        metrics.append(metric_row(name, y_test, y_pred, y_proba))
        saved_models.append(name)

    pd.DataFrame(metrics).sort_values("PR-AUC (2+)", ascending=False).to_csv(
        ARTIFACT_METRICS_PATH := ARTIFACT_DIR / "model_metrics.csv",
        index=False,
    )
    ARTIFACT_FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    ARTIFACT_CATS_PATH.write_text(json.dumps(cat_cols, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH.relative_to(BASE_DIR)),
        "target": "Y_IPDISY2_CLASS",
        "target_definition": "0 / 1 / 2+ Year-2 inpatient admissions",
        "train_panels": train_panels,
        "holdout_panel": holdout_panel,
        "training_rows": int(train_idx.sum()),
        "holdout_rows": int(test_idx.sum()),
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
        "saved_models": saved_models,
        "sklearn_version": sklearn.__version__,
        "catboost_version": catboost_version,
        "random_state": RANDOM_STATE,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Saved artifact bundle:")
    for path in sorted(ARTIFACT_DIR.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
