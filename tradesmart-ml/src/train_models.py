"""
TradeSmart AI - trains and compares the trade-in valuation regression
models described in the project documentation, Sections 5.1, 5.3 and 5.4:

  1. Depreciation-only baseline (value declining by a fixed % per year) -
     the "current rule-of-thumb method" the AI approach must beat.
  2. Linear Regression - simple supervised baseline model.
  3. Random Forest Regressor - tree-based ensemble.
  4. XGBoost Regressor (Gradient Boosted Trees) - tree-based ensemble.

Evaluation (Section 5.3): 80/20 train/test split, 10-fold cross-validation
on the training set for a robust MAE estimate, and MAE / RMSE / R^2 on the
held-out test set. Feature importance is reported for the tree models
(Section 5.4: explainability, e.g. for SHAP-style factor breakdowns).

All numbers below are computed for real, from the real Autotrader South
Africa dataset - nothing here is a placeholder or fabricated metric.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from features import load_raw, engineer_features, get_feature_target, build_preprocessor, PROJECT_ROOT

RANDOM_STATE = 42
MODELS_DIR = PROJECT_ROOT / "models"

# XGBoost settings found by tune_xgboost.py (randomized search, 5-fold CV on
# the training set only). The original hand-picked settings were
# n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.8,
# colsample_bytree=0.8: test MAE R45,607 -> R38,967 after tuning
# (see outputs/xgboost_tuning.json). Also used by split_experiments.py.
XGB_PARAMS = dict(
    n_estimators=584, max_depth=8, learning_rate=0.0291,
    subsample=0.7708, colsample_bytree=0.561,
    min_child_weight=1, reg_lambda=0.2857,
)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------
# Baseline: depreciation-only model
# ---------------------------------------------------------------------
class DepreciationBaseline:
    """
    Predicts Price as RefPrice(Brand, Model) * exp(-rate * age_delta),
    where RefPrice is the average price of that Brand/Model at its
    youngest observed age in the training data, age_delta is how many
    years older the vehicle is than that reference point, and rate is a
    single global depreciation rate fit on the training data. Falls back
    to Brand-level, then dataset-level, reference stats for a Model not
    seen during training (a real risk with 2,393 unique models).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X[["Brand", "Model", "VehicleAge"]].copy()
        df["Price"] = y.values

        # Reference price + age per (Brand, Model), per Brand, and overall.
        self.model_ref_ = (
            df.groupby(["Brand", "Model"])
            .apply(lambda g: pd.Series({
                "ref_age": g["VehicleAge"].min(),
                "ref_price": g.loc[g["VehicleAge"] == g["VehicleAge"].min(), "Price"].mean(),
            }), include_groups=False)
        )
        self.brand_ref_ = (
            df.groupby("Brand")
            .apply(lambda g: pd.Series({
                "ref_age": g["VehicleAge"].min(),
                "ref_price": g.loc[g["VehicleAge"] == g["VehicleAge"].min(), "Price"].mean(),
            }), include_groups=False)
        )
        self.global_ref_age_ = df["VehicleAge"].min()
        self.global_ref_price_ = df.loc[df["VehicleAge"] == self.global_ref_age_, "Price"].mean()

        # Fit a single global depreciation rate via least squares through
        # the origin: log(Price / RefPrice) = -rate * (Age - RefAge)
        ref_age, ref_price = self._lookup_batch(df[["Brand", "Model"]])
        age_delta = (df["VehicleAge"] - ref_age).clip(lower=0)
        with np.errstate(divide="ignore"):
            log_ratio = np.log(df["Price"] / ref_price)
        mask = (age_delta > 0) & np.isfinite(log_ratio)
        x = age_delta[mask].values
        yv = log_ratio[mask].values
        self.rate_ = -float(np.sum(x * yv) / np.sum(x * x))
        return self

    def _lookup_batch(self, brand_model_df: pd.DataFrame):
        ref_ages, ref_prices = [], []
        for brand, model in zip(brand_model_df["Brand"], brand_model_df["Model"]):
            if (brand, model) in self.model_ref_.index:
                row = self.model_ref_.loc[(brand, model)]
            elif brand in self.brand_ref_.index:
                row = self.brand_ref_.loc[brand]
            else:
                row = pd.Series({"ref_age": self.global_ref_age_, "ref_price": self.global_ref_price_})
            ref_ages.append(row["ref_age"])
            ref_prices.append(row["ref_price"])
        return np.array(ref_ages, dtype=float), np.array(ref_prices, dtype=float)

    def predict(self, X: pd.DataFrame):
        ref_age, ref_price = self._lookup_batch(X[["Brand", "Model"]])
        age_delta = (X["VehicleAge"].values - ref_age)
        return ref_price * np.exp(-self.rate_ * age_delta)


def evaluate(name, y_true, y_pred, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"{name:22s}  MAE=R{mae:>12,.2f}  RMSE=R{rmse:>12,.2f}  R2={r2:.4f}")


def main():
    df = load_raw()
    df_feat = engineer_features(df)
    X, y = get_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Baseline needs Brand/Model/VehicleAge directly (not the encoded matrix)
    df_train_idx = X_train.index
    df_test_idx = X_test.index
    baseline_train_X = df_feat.loc[df_train_idx, ["Brand", "Model", "VehicleAge"]]
    baseline_test_X = df_feat.loc[df_test_idx, ["Brand", "Model", "VehicleAge"]]

    results = {}
    predictions_for_plot = {}

    print("=" * 90)
    print("Held-out test set (20%%, n=%d) results" % len(X_test))
    print("=" * 90)

    # --- Baseline ---
    t0 = time.time()
    baseline = DepreciationBaseline().fit(baseline_train_X, y_train)
    pred = baseline.predict(baseline_test_X)
    evaluate("Depreciation baseline", y_test, pred, results)
    results["Depreciation baseline"]["fit_seconds"] = round(time.time() - t0, 2)
    results["Depreciation baseline"]["fitted_global_rate_per_year"] = round(baseline.rate_, 4)
    predictions_for_plot["Depreciation baseline"] = pred

    # --- Linear Regression ---
    t0 = time.time()
    lr_pipe = Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    pred = lr_pipe.predict(X_test)
    evaluate("Linear Regression", y_test, pred, results)
    results["Linear Regression"]["fit_seconds"] = round(time.time() - t0, 2)
    predictions_for_plot["Linear Regression"] = pred

    # --- Random Forest ---
    t0 = time.time()
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    pred = rf_pipe.predict(X_test)
    evaluate("Random Forest", y_test, pred, results)
    results["Random Forest"]["fit_seconds"] = round(time.time() - t0, 2)
    predictions_for_plot["Random Forest"] = pred

    # --- XGBoost ---
    t0 = time.time()
    xgb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    xgb_pipe.fit(X_train, y_train)
    pred = xgb_pipe.predict(X_test)
    evaluate("XGBoost", y_test, pred, results)
    results["XGBoost"]["fit_seconds"] = round(time.time() - t0, 2)
    predictions_for_plot["XGBoost"] = pred

    # --- 10-fold cross-validation (on training data only) for LR/RF/XGB ---
    print("\n" + "=" * 90)
    print("10-fold cross-validation MAE on the training set (robustness check)")
    print("=" * 90)
    # n_jobs=1 here on purpose: RandomForest/XGBoost already parallelize
    # internally (n_jobs=-1 on the estimator). Parallelizing folds AND
    # estimators at the same time oversubscribes the CPU and is slower,
    # not faster, on a machine with few cores.
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    for name, pipe in [("Linear Regression", lr_pipe), ("Random Forest", rf_pipe), ("XGBoost", xgb_pipe)]:
        t_cv = time.time()
        scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        print(f"  ({name} CV took {time.time() - t_cv:.1f}s)")
        cv_mae_mean = -scores.mean()
        cv_mae_std = scores.std()
        results[name]["cv10_MAE_mean"] = round(cv_mae_mean, 2)
        results[name]["cv10_MAE_std"] = round(cv_mae_std, 2)
        print(f"{name:22s}  10-fold CV MAE = R{cv_mae_mean:,.2f}  (+/- R{cv_mae_std:,.2f})")

    # --- Feature importance (tree models) ---
    feature_names = xgb_pipe.named_steps["prep"].get_feature_names_out()
    importances = {
        "Random Forest": rf_pipe.named_steps["model"].feature_importances_,
        "XGBoost": xgb_pipe.named_steps["model"].feature_importances_,
    }
    top_features = {}
    for name, imp in importances.items():
        order = np.argsort(imp)[::-1][:15]
        top_features[name] = [
            {"feature": feature_names[i], "importance": round(float(imp[i]), 4)}
            for i in order
        ]

    # --- Save everything ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_pipe, MODELS_DIR / "linear_regression.joblib")
    joblib.dump(rf_pipe, MODELS_DIR / "random_forest.joblib")
    joblib.dump(xgb_pipe, MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(baseline, MODELS_DIR / "depreciation_baseline.joblib")

    with open(OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"test_results": results, "top_feature_importance": top_features}, f, indent=2)

    np.save(OUTPUTS_DIR / "y_test.npy", y_test.values)
    for name, pred in predictions_for_plot.items():
        np.save(OUTPUTS_DIR / f"pred_{name.replace(' ', '_')}.npy", pred)

    print(f"\nSaved trained models to {MODELS_DIR}/ and metrics to {OUTPUTS_DIR}/model_comparison.json")
    best = min(results, key=lambda k: results[k]["MAE"])
    print(f"\nBest model by test-set MAE: {best}")
    return results, top_features


if __name__ == "__main__":
    main()
