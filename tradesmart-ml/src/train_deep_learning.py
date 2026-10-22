"""
TradeSmart AI - deep learning regressor (project documentation Section 5.6).

Section 5.6 commits to "a Multi-Layer Perceptron (deep neural network)
regressor, trained on the same structured feature set as the tree-based
models (Section 5.1) and compared/ensembled with them to see whether a
neural network captures additional non-linear patterns and improves
overall valuation accuracy."

This script:
  - loads the exact same feature set as train_models.py (features.py is
    shared between both scripts, so there's only one definition of the
    feature set to keep in sync);
  - uses the identical train/test split (same random_state=42, same
    test_size=0.2) so the MLP's test-set predictions are directly
    comparable to the baseline / Linear Regression / Random Forest /
    XGBoost numbers already saved in outputs/model_comparison.json -
    the script verifies this by checking its y_test against the one
    train_models.py already saved to disk;
  - standardises the target (Price) before training. Rand prices here
    range from the tens of thousands to several million; without target
    scaling, MSE loss on raw Rand values produces huge gradients early
    in training and the network struggles to converge smoothly;
  - uses EarlyStopping on a validation split so the network doesn't
    just memorise the ~13,000 training rows (a real risk for a dataset
    this size).

Run after train_models.py (it reuses outputs/y_test.npy for the sanity
check, though it does not strictly require train_models.py to have run
first - it will just skip that check if the file isn't there yet).
"""

import json
import time
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import load_raw, get_feature_target, build_preprocessor, PROJECT_ROOT

RANDOM_STATE = 42
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

tf.random.set_seed(RANDOM_STATE)


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def main():
    df = load_raw()
    X, y = get_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Fit the SAME preprocessing pipeline used by the tree/linear models
    # (features.py's build_preprocessor), fit on the training fold only.
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):  # ColumnTransformer may return sparse
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    # Standardise the target for training stability; invert for reporting.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = build_mlp(X_train_proc.shape[1])
    print(model.summary())

    # NOTE: TensorFlow training is not perfectly reproducible across
    # different machines even with a fixed random seed (thread count,
    # CPU instruction set, and op-level parallelism all introduce small
    # variation) - patience=15 was too tight and let a slow-starting
    # run stop at epoch ~19, well before convergence, on some machines.
    # A longer patience plus a learning-rate reduction is a more robust
    # way to reach a comparable result run-to-run.
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=40, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5
    )

    t0 = time.time()
    history = model.fit(
        X_train_proc, y_train_scaled,
        validation_split=0.15,
        epochs=300,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )
    fit_seconds = time.time() - t0
    epochs_trained = len(history.history["loss"])

    pred_scaled = model.predict(X_test_proc, verbose=0).ravel()
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    print("\n" + "=" * 90)
    print(f"Deep Learning MLP   MAE=R{mae:>12,.2f}  RMSE=R{rmse:>12,.2f}  R2={r2:.4f}"
          f"   ({epochs_trained} epochs, {fit_seconds:.1f}s)")
    print("=" * 90)

    # --- Save model + the two artefacts needed to reuse it later ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "mlp_regressor.keras")
    joblib.dump(preprocessor, MODELS_DIR / "mlp_preprocessor.joblib")
    joblib.dump(y_scaler, MODELS_DIR / "mlp_target_scaler.joblib")

    # --- Merge into the existing model comparison report ---
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUTS_DIR / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            report = json.load(f)
    else:
        report = {"test_results": {}, "top_feature_importance": {}}

    report["test_results"]["Deep Learning (MLP)"] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "fit_seconds": round(fit_seconds, 2),
        "epochs_trained": epochs_trained,
        "architecture": "Dense(128)-BN-Dropout(0.2)-Dense(64)-BN-Dropout(0.2)-Dense(32)-Dense(1), Adam lr=1e-3, target standardised",
    }
    with open(comparison_path, "w") as f:
        json.dump(report, f, indent=2)

    np.save(OUTPUTS_DIR / "pred_Deep_Learning_MLP.npy", pred)

    # --- Sanity check: same split as train_models.py? ---
    saved_y_test_path = OUTPUTS_DIR / "y_test.npy"
    if saved_y_test_path.exists():
        saved_y_test = np.load(saved_y_test_path)
        if saved_y_test.shape == y_test.values.shape and np.allclose(saved_y_test, y_test.values):
            print("Confirmed: identical train/test split to train_models.py - "
                  "the MLP result above is directly comparable to the other four models.")
        else:
            print("WARNING: y_test does not match train_models.py's saved split - "
                  "re-run train_models.py after this script, or investigate before "
                  "quoting these numbers side-by-side.")

    print(f"\nSaved model to {MODELS_DIR}/mlp_regressor.keras and updated {comparison_path}")
    return mae, rmse, r2


if __name__ == "__main__":
    main()
