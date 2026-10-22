"""
TradeSmart AI - does the result depend on how the data is split?

The reported accuracy (R^2 ~0.97) comes from one 80/20 split: the model is
trained on 13,162 cars and tested on the other 3,291 it has never seen. A
fair question is whether that number depends on that particular split, or
on the amount of training data. This script answers both with two
experiments, using exactly the XGBoost settings the app uses:

  1. Split ratio: train/test at 60/40, 70/30, 80/20 and 90/10, each repeated
     with 5 different random shuffles of the data. If the model is sound,
     the results should barely move between shuffles or ratios.

  2. Training size (a learning curve): train on random samples of 1,000,
     2,000, 3,000, 5,000, 8,000 and all 13,162 training cars (3 samples
     each), always tested on the same 3,291 unseen test cars. This shows
     how much accuracy depends on the amount of training data.

Writes outputs/split_experiments.json and outputs/figures/learning_curve.png.
Run from the src folder (takes a few minutes):
    python split_experiments.py
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from features import load_raw, get_feature_target, build_preprocessor, PROJECT_ROOT
from train_models import XGB_PARAMS, RANDOM_STATE

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TEST_SHARES = [0.4, 0.3, 0.2, 0.1]
SHUFFLES = [0, 1, 2, 3, 4]
TRAIN_SIZES = [1000, 2000, 3000, 5000, 8000]
SAMPLES_PER_SIZE = 3

BLUE, INK, INK_SECONDARY, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def fit_and_score(X_train, y_train, X_test, y_test):
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("model", XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    pct_err = np.abs(pred - y_test.values) / y_test.values
    return (float(r2_score(y_test, pred)), float(mean_absolute_error(y_test, pred)),
            float(np.mean(pct_err <= 0.10)), float(np.mean(pct_err <= 0.15)))


def summary(values):
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4)}


def main():
    t0 = time.time()
    X, y = get_feature_target(load_raw())
    report = {"cars": int(len(X)), "split_ratio": [], "learning_curve": []}

    # 1. Split ratio x random shuffle
    print("Experiment 1: split ratio, 5 random shuffles each")
    for test_share in TEST_SHARES:
        r2s, maes, w10s, w15s = [], [], [], []
        for seed in SHUFFLES:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_share, random_state=seed)
            r2, mae, w10, w15 = fit_and_score(X_tr, y_tr, X_te, y_te)
            r2s.append(r2); maes.append(mae); w10s.append(w10); w15s.append(w15)
        row = {"train_pct": round(100 * (1 - test_share)), "test_pct": round(100 * test_share),
               "train_cars": int(len(X_tr)), "test_cars": int(len(X_te)), "R2": summary(r2s), "MAE": summary(maes),
               "within_10pct": summary(w10s), "within_15pct": summary(w15s)}
        report["split_ratio"].append(row)
        print(f"  {row['train_pct']}/{row['test_pct']} ({row['train_cars']:,} train / {row['test_cars']:,} test): "
              f"R2 {row['R2']['mean']:.3f} (range {row['R2']['min']:.3f}-{row['R2']['max']:.3f}), "
              f"MAE R{row['MAE']['mean']:,.0f} (+/- R{row['MAE']['std']:,.0f}), "
              f"within 15%: {row['within_15pct']['mean']:.1%}")

    # 2. Learning curve: fixed test set, growing random training samples
    print("Experiment 2: training size, same 3,291 test cars every time")
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    for size in TRAIN_SIZES + [len(X_tr_full)]:
        r2s, maes = [], []
        for _ in range(1 if size == len(X_tr_full) else SAMPLES_PER_SIZE):
            idx = rng.choice(len(X_tr_full), size=size, replace=False)
            r2, mae, _, _ = fit_and_score(X_tr_full.iloc[idx], y_tr_full.iloc[idx], X_te, y_te)
            r2s.append(r2); maes.append(mae)
        row = {"train_cars": int(size), "test_cars": int(len(X_te)), "R2": summary(r2s), "MAE": summary(maes)}
        report["learning_curve"].append(row)
        print(f"  {size:>6,} training cars: R2 {row['R2']['mean']:.3f}, MAE R{row['MAE']['mean']:,.0f}")

    report["seconds"] = round(time.time() - t0)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "split_experiments.json", "w") as f:
        json.dump(report, f, indent=2)

    # Figure: learning curve (R^2 by training size, min-max band over samples)
    plt.rcParams["font.family"] = "Arial"
    lc = report["learning_curve"]
    xs = [r["train_cars"] for r in lc]
    mean = [r["R2"]["mean"] for r in lc]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.fill_between(xs, [r["R2"]["min"] for r in lc], [r["R2"]["max"] for r in lc], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(xs, mean, "-o", color=BLUE, linewidth=2, markersize=8, markeredgecolor="white", markeredgewidth=2)
    for x, m in zip(xs, mean):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
    ax.set_title("Accuracy by amount of training data (tested on the same 3,291 unseen cars)",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel("Cars used for training", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("R² on unseen test cars", color=INK_SECONDARY, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    lo = min(r["R2"]["min"] for r in lc)
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curve.png")
    plt.close(fig)

    print(f"Saved {OUTPUTS_DIR / 'split_experiments.json'} and figures/learning_curve.png ({report['seconds']}s)")


if __name__ == "__main__":
    main()
