"""
TradeSmart AI - Feature engineering for the trade-in valuation model.

Loads the real Autotrader South Africa used-car dataset and builds the
feature matrix / target used by every model in this project, so all
models (baseline, Linear Regression, Random Forest, XGBoost, and the
deep learning MLP in train_deep_learning.py) are trained on an
identical, reproducible feature set.

Design notes (for the project write-up):
- Model (the specific trim name, e.g. "Polo Vivo Hatch 1.4 Trendline")
  has 2,393 unique values across 16,453 rows - too high-cardinality to
  one-hot encode without badly overfitting. Instead we engineer
  ModelFrequency: how many times that exact model string appears in the
  dataset, which captures "how common/popular is this model" as a
  single numeric signal without exploding the feature space.
- VehicleAge is derived from Year relative to the newest model year
  seen in the dataset (2023), rather than the wall-clock date, so the
  feature set is reproducible independent of when this script is run.
- Colour, exact tyre sizes, and Fuel range are dropped: weak/redundant
  price signal once Body Type, Engine capacity and Fuel consumption are
  already included, and keeping the feature set lean makes the
  regression coefficients / feature-importance output easier to
  explain in the presentation.
"""

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Resolved relative to this file's location (not the process cwd). This is
# the tradesmart-backend copy of features.py - here data/ is a direct
# sibling of this file (unlike the tradesmart-ml/src/ copy, where it's one
# level up), so PROJECT_ROOT is just this file's own directory.
PROJECT_ROOT = Path(__file__).resolve().parent
# The corrected copy written by tradesmart-ml/src/fix_dataset.py - see that
# script for the three errors in the published dataset it fixes.
DATA_PATH = PROJECT_ROOT / "data" / "autotrader-south-africa-usedcars (corrected).csv"

NEWEST_MODEL_YEAR = 2023  # fixed reference point - see module docstring

NUMERIC_FEATURES = [
    "VehicleAge",
    "Mileage",
    "Engine capacity (litre)",
    "Seats (quantity)",
    "No of doors",
    "Fuel capacity",
    "Fuel consumption (average)",
    "Power maximum (detail)",
    "Torque maximum",
    "Acceleration 0-100 km/h",
    "Maximum/top speed",
    "CO2 emissions (average)",
    "ModelFrequency",
]

CATEGORICAL_FEATURES = [
    "Brand",
    "Transmission",
    "Fuel Type",
    "Previous Owners",
    "Body Type",
    "Engine position",
    "Cylinder layout and quantity",
    "Driven wheels",
]

TARGET = "Price"


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VehicleAge"] = NEWEST_MODEL_YEAR - df["Year"] + 1
    df["ModelFrequency"] = df.groupby("Model")["Model"].transform("count")
    # Previous Owners mixes numeric strings ("1","2"...) with "Unknown" -
    # treat as a categorical string so both encode cleanly.
    df["Previous Owners"] = df["Previous Owners"].astype(str)
    return df


def get_feature_target(df: pd.DataFrame):
    df = engineer_features(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    A fit-on-train-only preprocessing pipeline (imputation + scaling +
    one-hot encoding), so it can be safely used inside cross-validation
    without leaking information from the test fold.
    """
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    df = load_raw()
    X, y = get_feature_target(df)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print(X.head())
