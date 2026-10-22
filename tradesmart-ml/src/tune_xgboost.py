"""
TradeSmart AI - hyperparameter tuning for the XGBoost valuation model.

train_models.py trains XGBoost with hand-picked settings (250 trees, depth
6, learning rate 0.05, ...). This script searches for better settings and
measures whether they actually improve accuracy - one of the ways the
model's accuracy is improved (see the documentation's improvement plan).

How it stays honest:
  - Same 80/20 split as train_models.py (random_state=42), so the test set
    is the same 3,291 cars every model is reported on.
  - The search (RandomizedSearchCV, 5-fold cross-validation) only ever
    sees the training 80%. The test set is used once at the end, to
    compare the tuned model with the current one.

Writes outputs/xgboost_tuning.json. If the tuned model is better on the
test set, it's also saved as models/xgboost_tuned.joblib - it doesn't
replace models/xgboost_model.joblib (the one the app serves) unless you
copy it over yourself. Run from the src folder:
    python tune_xgboost.py
"""

import json
import time

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from features import load_raw, get_feature_target, build_preprocessor, PROJECT_ROOT

RANDOM_STATE = 42
N_CANDIDATES = 40
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SEARCH_SPACE = {
    "model__n_estimators": randint(200, 1200),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__subsample": uniform(0.6, 0.4),          # 0.6 - 1.0
    "model__colsample_bytree": uniform(0.5, 0.5),   # 0.5 - 1.0
    "model__min_child_weight": randint(1, 10),
    "model__reg_lambda": loguniform(0.1, 10),
}


def scores(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    X, y = get_feature_target(load_raw())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    current = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    before = scores(y_test, current.predict(X_test))
    print(f"Current XGBoost (train_models.py settings): {before}")

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACE, n_iter=N_CANDIDATES,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    search_seconds = time.time() - t0

    tuned = search.best_estimator_
    after = scores(y_test, tuned.predict(X_test))
    best_params = {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else int(v))
                   for k, v in search.best_params_.items()}
    improved = after["MAE"] < before["MAE"] and after["RMSE"] < before["RMSE"]

    report = {
        "method": f"RandomizedSearchCV, {N_CANDIDATES} candidates, 5-fold CV on the training 80% only, "
                  "scored by MAE; then evaluated once on the same held-out test set as train_models.py",
        "search_seconds": round(search_seconds, 1),
        "best_cv_MAE": round(float(-search.best_score_), 2),
        "best_params": best_params,
        "test_before": before,
        "test_after": after,
        "MAE_change_pct": round(100 * (after["MAE"] - before["MAE"]) / before["MAE"], 1),
        "RMSE_change_pct": round(100 * (after["RMSE"] - before["RMSE"]) / before["RMSE"], 1),
        "improved_on_both_MAE_and_RMSE": improved,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "xgboost_tuning.json", "w") as f:
        json.dump(report, f, indent=2)
    if improved:
        joblib.dump(tuned, MODELS_DIR / "xgboost_tuned.joblib")

    print(f"Search: {N_CANDIDATES} candidates x 5 folds in {search_seconds:.0f}s, best CV MAE "
          f"R{-search.best_score_:,.0f}")
    print(f"Best settings: {best_params}")
    print(f"Tuned XGBoost on the test set: {after}")
    print(f"MAE {report['MAE_change_pct']:+.1f}%, RMSE {report['RMSE_change_pct']:+.1f}%"
          + (" - saved as models/xgboost_tuned.joblib" if improved else " - not better on both, not saved"))


if __name__ == "__main__":
    main()
