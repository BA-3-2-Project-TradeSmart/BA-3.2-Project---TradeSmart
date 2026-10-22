# TradeSmart AI - ML / NLP / Deep Learning component

This folder contains the machine learning, deep learning, and NLP
components of TradeSmart AI, built to satisfy Sections 5.1, 5.3, 5.4,
5.5 and 5.6 of `TradeSmart_AI_Project_Documentation.docx`. Everything
here is trained on the real Autotrader South Africa used-car dataset
(16,453 listings after correction - see `fix_dataset.py`).

The dataset was scraped and published by the open-source
used-car-price-prediction project (Victor Adewoyin, 2023, MIT licence).
Its cleaning step introduced three errors - two-word brands like Land
Rover split in half, every fuel consumption figure from 10-19.9 L/100km
missing its leading "1", and a few impossible mileages - which
`fix_dataset.py` corrects before anything is trained.

## What's in here

```
tradesmart-ml/
  data/
    autotrader-south-africa-usedcars (cleaned data).csv   <- as published, 16,456 rows
    autotrader-south-africa-usedcars (corrected).csv      <- written by fix_dataset.py, 16,453 rows
  src/
    fix_dataset.py          corrects the published dataset (run first)
    features.py             feature engineering, shared by every model
    train_models.py         baseline + Linear Regression + Random Forest + XGBoost
    train_deep_learning.py  the MLP deep-learning regressor (doc Section 5.6)
    nlp_parser.py           free-text condition parser (doc Section 5.5)
    time_series_analysis.py depreciation over time + forecast (doc Section 5.3)
    tune_xgboost.py         hyperparameter search for the XGBoost model
    split_experiments.py    how much the result depends on the split / amount of training data
  models/                   trained models get saved here when you run the scripts
  outputs/                  metrics / reports get saved here when you run the scripts
  requirements.txt
  README.md                 this file
```

## 1. Setting up Python on your Windows PC

1. Go to https://www.python.org/downloads/ and download the latest
   **Python 3.11 or 3.12** installer (these versions have the most
   reliable Windows compatibility with TensorFlow and spaCy - avoid
   the very newest Python release for this project).
2. Run the installer. **Tick "Add python.exe to PATH"** on the first
   screen before clicking Install - this is the step people usually
   miss, and without it `python` won't work from a terminal.
3. Open a terminal (PowerShell or Command Prompt) and confirm it
   worked:
   ```
   python --version
   ```
4. Navigate to this folder in the terminal, e.g.:
   ```
   cd "C:\Werk\VUT Third Year\AIBUX3B\FINAL\tradesmart-ml"
   ```
5. (Recommended) Create and activate a virtual environment, so these
   packages don't clash with anything else on your PC:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
6. Install the dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
   This will take a few minutes the first time (TensorFlow is a large
   download).

## 2. Running everything, in order

From inside the `src` folder (`cd src`), with the virtual environment
active:

```
python fix_dataset.py            # a few seconds - writes the corrected CSV
python train_models.py           # ~5-10 min (Random Forest is the slow part)
python train_deep_learning.py    # ~1 min
python nlp_parser.py             # a few seconds
python time_series_analysis.py   # a few seconds - depreciation curve + forecast, 3 figures
python tune_xgboost.py           # optional, ~10 min - re-runs the hyperparameter search
python split_experiments.py      # ~1 min - accuracy across split ratios, shuffles and training sizes
```

Each script prints its results to the terminal AND saves them into
`models/` and `outputs/`, so you don't need to keep the terminal
output - you can always re-open `outputs/model_comparison.json` and
`outputs/nlp_demo_results.json` afterwards.

**Note on the Random Forest model file:** `train_models.py` will
recreate `models/random_forest.joblib`, but that file is roughly
100MB, so it isn't included in this delivered folder to keep things
easy to transfer - running the script once regenerates it locally.
Every other model (baseline, Linear Regression, XGBoost, the MLP, the
NLP classifier) is small and is included.

## 3. Results (from the run already completed, real numbers - not estimates)

### Valuation regression models (doc Sections 5.1, 5.3, 5.6)

Trained on the corrected dataset (`fix_dataset.py`) and evaluated on
the same held-out 20% test set (3,291 vehicles), with 10-fold
cross-validation on the training set for a more robust MAE estimate:

| Model | Test MAE (R) | Test RMSE (R) | Test R² | 10-fold CV MAE (R) |
|---|---|---|---|---|
| Depreciation-only baseline (current rule-of-thumb) | 59,980.62 | 163,203.87 | 0.9021 | - |
| Linear Regression | 92,169.17 | 181,579.84 | 0.8788 | 91,281.28 ± 5,924.59 |
| Random Forest | 43,196.11 | 107,392.04 | 0.9576 | 45,683.42 ± 5,795.67 |
| XGBoost (tuned) | 39,116.29 | 85,414.80 | 0.9732 | 42,403.60 ± 4,915.78 |
| Deep Learning (MLP) | 51,528.97 | 148,097.83 | 0.9194 | - |

**What this shows, for the presentation:** Random Forest, XGBoost and
the MLP all beat the depreciation-only baseline on RMSE and R²; Linear
Regression doesn't beat it on anything - a straight-line model can't
capture how differently a R150k hatchback and a R2.5m SUV depreciate,
which is itself a useful argument for the non-linear models. XGBoost
is served by the app: since tuning (below) it is best on every measure -
lowest MAE, lowest RMSE, highest R² - and its model file is under 1MB,
against ~100MB for Random Forest. Trade-in value depends on non-linear
feature interactions (Section 5.1's own reasoning) that tree ensembles
capture more directly than a linear model or even the MLP can, on a
dataset this size (~16k rows). The MLP clearly beats Linear Regression
and the baseline and is a genuine, honestly evaluated Section 5.6
deliverable - it's just not the top performer, which is a realistic
and defensible finding for tabular data of this scale, not a bug.

Before the dataset was corrected, the same XGBoost setup scored R²
0.953 / RMSE R98,254. Part of the improvement to 0.972 / R87,427 is the
cleaner data, and part is that the three impossible-mileage rows are
no longer in the test set (the baseline's R² rose too, 0.875 -> 0.902)
- so quote it as "corrected data, re-evaluated", not as a pure model gain.

**Hyperparameter tuning** (`tune_xgboost.py`): a randomized search over
40 combinations of XGBoost settings, each scored by 5-fold
cross-validation on the training 80% only, then checked once on the
same test set. The original hand-picked settings scored MAE R45,607 /
RMSE R87,427 / R² 0.9719; the tuned settings, now built into
`train_models.py`, score MAE R39,116 / RMSE R85,415 / R² 0.9732 - a
14% lower average error. (The search itself reported R38,967; the
small difference is from rounding the settings when copying them into
`train_models.py`.) Full output in `outputs/xgboost_tuning.json`.

**How reliable is the 0.97?** (`split_experiments.py`) The table above is
one 80/20 split (13,162 training cars, 3,291 test cars). Retraining on
60/40, 70/30, 80/20 and 90/10 splits, 5 random shuffles each, gives an
average error of R40k-R43k every time and ~88% of estimates within 15%
- but R² ranges 0.86-0.95 across 80/20 shuffles (average 0.91), because a
few very expensive cars dominate R². Our main split is at the favourable
end, so the app and the documentation report the average: **R² 0.91,
average error R41,905**. Trained on only 3,000 cars the model reaches R²
0.90; with all 13,162 it reaches 0.97 on the main test set.

**Time series analysis** (`time_series_analysis.py`): same-model
comparisons across model years show cars keep 91.7% of their value
after 1 year and 62.5% after 5 - about 9.5% depreciation a year (bakkies
6-8%, sedans 11%). Results and three figures in `outputs/`.

Feature importance (Random Forest and XGBoost both agree engine
power and torque dominate, ahead of mileage and vehicle age) is saved
in `outputs/model_comparison.json` under `top_feature_importance` -
useful directly for the "explainability" point in Section 5.4.

### NLP free-text parser (doc Section 5.5)

The intent classifier (TF-IDF + Logistic Regression trained on 102
hand-authored example customer answers - see the docstring in
`nlp_parser.py` for why these are hand-authored rather than real
logged transcripts, which don't exist yet) reached **96% accuracy**
on a held-out test split. That split is only 26 examples, so treat it
as indicative: with the original 80 full sentences it scored 85%, and
22 short fragments ("a few scratches", "aftermarket rims") were added
because that's how people actually answer the chat's condition boxes -
before them it filed "scratches" under clean_no_issues. Full per-class
precision/recall is in `outputs/nlp_intent_classifier_report.json`.

The chatbot uses this trained classifier live: `chatbot/condition_nlp.py`
categorises each thing a customer mentions about the car's condition, to
check when the car is inspected.

The parser combines that trained intent model with regex/dictionary
entity extraction (mileage figures, dent-size wording like "about the
size of a golf ball" mapped to small/medium/large) and keyword
mapping (respray, aftermarket mods, reputable-vs-DIY install), plus a
rule-based consistency check that correctly flagged a deliberately
contradictory test sentence ("no accidents, but it was resprayed
after a small bump") - see `outputs/nlp_demo_results.json` for all
ten demo sentences and their extracted structured output.

## 4. How this maps to the project documentation

| Doc section | Where it's implemented |
|---|---|
| 5.1 Machine Learning Approach | `train_models.py` - baseline, Linear Regression, Random Forest, XGBoost all trained and compared |
| 5.3 Model, Evaluation | `train_models.py` - 80/20 split, 10-fold CV, MAE/RMSE/R² |
| 5.4 Solution Techniques (explainability) | `outputs/model_comparison.json` - `top_feature_importance` for both tree models |
| 5.5 NLP | `nlp_parser.py` - intent recognition, entity/number extraction, keyword mapping, confidence/consistency check |
| 5.6 Deep Learning | `train_deep_learning.py` - MLP regressor, same feature set, directly compared against the models above |

A few things flagged honestly rather than glossed over, for your own
presentation:

- **Hyperparameter tuning (GridSearchCV) and the continuous-learning
  feedback loop** described in Section 5.4 are not implemented here -
  they're natural "next steps" to mention if asked, not gaps to hide.
- **Condition never changes the price.** The regression models only
  know make/model, year and mileage-derived specs; what a customer says
  about condition is categorised by the NLP classifier (one finding per
  thing mentioned) to check at inspection.
- **Time-series depreciation forecasting** (Prophet/ARIMA, Section
  5.3) is not built - the `DepreciationBaseline` class in
  `train_models.py` implements the simpler "fixed % per year"
  baseline the doc explicitly asks every other model to be compared
  against, but not the seasonal-trend forecasting extension.
