"""
Train the AEGIS multiclass network-traffic classifier.

This proof-of-concept uses CICIoT2023 CSV files and trains on the original
34 labels (33 attack types plus BenignTraffic).  The inference pipeline
routes those detailed predictions to the appropriate operational response.

The script deliberately samples the data before fitting XGBoost.  The source
CSV files are large, so numeric columns are downcast immediately after each
CSV is read and the full DataFrame is released as soon as its sample has been
created.  After XGBoost training, a benign-only MLP autoencoder is fitted and
saved with its scaler and reconstruction-error threshold for zero-day checks.

Example:
    python 1_train_aegis.py

Or with explicit paths:
    python 1_train_aegis.py --train path/to/train.csv --test path/to/test.csv
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Project-level constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_COLUMN = "label"
AUTOENCODER_SAMPLE_SIZE = 50_000

DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "aegis_xgb_model.joblib"
DEFAULT_ENCODER_PATH = DEFAULT_ARTIFACT_DIR / "aegis_label_encoder.joblib"
DEFAULT_SCALER_PATH = DEFAULT_ARTIFACT_DIR / "aegis_scaler.joblib"
DEFAULT_AUTOENCODER_PATH = DEFAULT_ARTIFACT_DIR / "aegis_autoencoder.joblib"
DEFAULT_THRESHOLD_PATH = DEFAULT_ARTIFACT_DIR / "aegis_threshold.joblib"


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_csv_optimized(path: Path) -> pd.DataFrame:
    """Read CICIoT2023 with its fixed schema and memory-efficient dtypes."""

    print(f"Loading {path} ...")
    feature_columns = [
        "flow_duration",
        "Header_Length",
        "Protocol Type",
        "Duration",
        "Rate",
        "Srate",
        "Drate",
        "fin_flag_number",
        "syn_flag_number",
        "rst_flag_number",
        "psh_flag_number",
        "ack_flag_number",
        "ece_flag_number",
        "cwr_flag_number",
        "ack_count",
        "syn_count",
        "fin_count",
        "urg_count",
        "rst_count",
        "HTTP",
        "HTTPS",
        "DNS",
        "Telnet",
        "SMTP",
        "SSH",
        "IRC",
        "TCP",
        "UDP",
        "DHCP",
        "ARP",
        "ICMP",
        "IPv",
        "LLC",
        "Tot sum",
        "Min",
        "Max",
        "AVG",
        "Std",
        "Tot size",
        "IAT",
        "Number",
        "Magnitue",
        "Radius",
        "Covariance",
        "Variance",
        "Weight",
    ]
    dtypes = {column: np.float32 for column in feature_columns}
    dtypes[LABEL_COLUMN] = "string"

    frame = pd.read_csv(path, dtype=dtypes)

    if LABEL_COLUMN not in frame.columns:
        raise ValueError(
            f"{path} does not contain the required target column "
            f"{LABEL_COLUMN!r}. Columns found: {list(frame.columns)}"
        )

    memory_mb = frame.memory_usage(deep=True).sum() / (1024**2)
    print(f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns "
          f"({memory_mb:.1f} MiB with hardcoded dtypes).")
    return frame


def stratified_sample(
    frame: pd.DataFrame,
    fraction: float,
    random_state: int,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Take a reproducible per-label sample while retaining rare classes."""

    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be greater than 0 and at most 1")

    sampled_parts = []
    for _, group in frame.groupby(label_column, sort=False, dropna=False):
        sample_size = max(1, int(np.ceil(len(group) * fraction)))
        sample_size = min(sample_size, len(group))
        sampled_parts.append(group.sample(n=sample_size, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    # Shuffle after concatenating groups so the estimator does not see labels
    # in contiguous blocks.
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def resolve_dataset_path(argument: str | None, filename: str) -> Path:
    """Resolve an explicit dataset path or a useful project-local default."""

    if argument:
        path = Path(argument).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path

    candidates = (
        Path.cwd() / filename,
        SCRIPT_DIR / filename,
        SCRIPT_DIR / "archive" / "CICIOT23" / ("train" if filename == "train.csv" else "test") / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    formatted = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {filename!r}. Checked:\n{formatted}\n"
        "Pass an explicit --train or --test path."
    )


def validate_numeric_features(features: pd.DataFrame) -> None:
    """Fail early if a non-numeric feature would reach XGBoost."""

    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise TypeError(
            "All feature columns must be numeric for XGBoost. "
            f"Non-numeric columns: {non_numeric}"
        )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_aegis(args: argparse.Namespace) -> None:
    train_path = resolve_dataset_path(args.train, "train.csv")
    test_path = resolve_dataset_path(args.test, "test.csv")
    model_path = Path(args.model_out).expanduser().resolve()
    encoder_path = Path(args.encoder_out).expanduser().resolve()
    scaler_path = Path(args.scaler_out).expanduser().resolve()
    autoencoder_path = Path(args.autoencoder_out).expanduser().resolve()
    threshold_path = Path(args.threshold_out).expanduser().resolve()

    # Load and sample train data first, then release the full DataFrame before
    # loading the test CSV.  This keeps the peak resident memory lower on a
    # 16-GiB machine.
    train_full = load_csv_optimized(train_path)
    # Preserve the complete benign subset before releasing the full training
    # frame.  The normal XGBoost sample fraction can be much smaller than
    # 50,000 benign rows, so taking the autoencoder sample from train_sample
    # would not reliably satisfy the dual-engine training requirement.
    train_full[LABEL_COLUMN] = train_full[LABEL_COLUMN].astype("string").str.strip()
    benign_training_rows = train_full.loc[
        train_full[LABEL_COLUMN] == "BenignTraffic"
    ].copy()
    train_sample = stratified_sample(
        train_full, args.sample_frac, args.random_state
    )
    del train_full
    gc.collect()

    test_full = load_csv_optimized(test_path)
    test_sample = stratified_sample(
        test_full, args.sample_frac, args.random_state
    )
    del test_full
    gc.collect()

    print(
        f"Using {len(train_sample):,} stratified training rows and "
        f"{len(test_sample):,} stratified test rows "
        f"({args.sample_frac:.1%} sample fraction)."
    )

    # Strip whitespace and use all 34 raw dataset labels directly.
    train_sample[LABEL_COLUMN] = train_sample[LABEL_COLUMN].astype("string").str.strip()
    test_sample[LABEL_COLUMN] = test_sample[LABEL_COLUMN].astype("string").str.strip()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_sample[LABEL_COLUMN]).astype(np.int32)

    unknown_test_categories = sorted(
        set(test_sample[LABEL_COLUMN].dropna().unique())
        - set(label_encoder.classes_)
    )
    if unknown_test_categories:
        raise ValueError(
            "The test sample contains categories missing from the training "
            f"sample: {unknown_test_categories}"
        )
    y_test = label_encoder.transform(test_sample[LABEL_COLUMN]).astype(np.int32)

    feature_columns = [column for column in train_sample.columns if column != LABEL_COLUMN]
    test_feature_columns = [column for column in test_sample.columns if column != LABEL_COLUMN]
    if feature_columns != test_feature_columns:
        raise ValueError(
            "Train/test feature columns differ or have a different order.\n"
            f"Train-only/order: {feature_columns}\n"
            f"Test-only/order: {test_feature_columns}"
        )

    X_train = train_sample[feature_columns]
    X_test = test_sample[feature_columns]
    validate_numeric_features(X_train)
    validate_numeric_features(X_test)

    print(f"Label encoder classes: {list(label_encoder.classes_)}")
    print("Training XGBClassifier ...")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        max_bin=args.max_bin,
        eval_metric="mlogloss",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # Engine 1 training: benign-only autoencoder for zero-day detection
    # ------------------------------------------------------------------
    # The autoencoder learns the normal shape of network traffic.  It is
    # deliberately trained only on BenignTraffic, so attack traffic should
    # produce a larger reconstruction error at inference time.
    if len(benign_training_rows) < AUTOENCODER_SAMPLE_SIZE:
        raise ValueError(
            "The training data contains only "
            f"{len(benign_training_rows):,} BenignTraffic rows; at least "
            f"{AUTOENCODER_SAMPLE_SIZE:,} are required for autoencoder training."
        )

    # Isolate exactly 50,000 benign rows for the normal-traffic model.  This
    # sample is intentionally independent of the smaller XGBoost sample.
    benign_sample = benign_training_rows.sample(
        n=AUTOENCODER_SAMPLE_SIZE,
        random_state=args.random_state,
    )
    X_benign = benign_sample[feature_columns]

    print(
        "Training benign-only autoencoder on "
        f"{len(X_benign):,} rows and {len(feature_columns):,} features ..."
    )
    scaler = StandardScaler()
    X_benign_scaled = scaler.fit_transform(X_benign)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32),
        max_iter=50,
        random_state=args.random_state,
    )
    # Reconstruction training uses the same benign matrix as both input and
    # target: the network learns to reproduce normal traffic features.
    autoencoder.fit(X_benign_scaled, X_benign_scaled)

    benign_reconstructions = autoencoder.predict(X_benign_scaled)
    benign_mse = np.mean(
        np.square(X_benign_scaled - benign_reconstructions),
        axis=1,
    )
    mean_mse = np.mean(benign_mse)
    std_mse = np.std(benign_mse)
    zero_day_threshold = float(mean_mse + (3 * std_mse))

    print(
        "Benign reconstruction MSE mean + 3 standard deviations threshold: "
        f"{zero_day_threshold:.6f}"
    )
    # The fitted artifacts retain the learned parameters; release the large
    # temporary benign matrices before serializing all artifacts.
    del (
        benign_training_rows,
        benign_sample,
        X_benign,
        X_benign_scaled,
        benign_reconstructions,
        benign_mse,
    )
    gc.collect()

    predictions = np.asarray(model.predict(X_test)).reshape(-1).astype(np.int32)
    accuracy = accuracy_score(y_test, predictions)
    class_ids = np.arange(len(label_encoder.classes_), dtype=np.int32)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Store the feature schema on the estimator so inference can protect
    # against columns arriving in the wrong order.  joblib preserves this
    # custom attribute along with the fitted XGBoost model.
    model.aegis_feature_columns_ = tuple(feature_columns)
    model.aegis_category_names_ = tuple(label_encoder.classes_)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    autoencoder_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    joblib.dump(label_encoder, encoder_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)
    joblib.dump(autoencoder, autoencoder_path, compress=3)
    joblib.dump(zero_day_threshold, threshold_path, compress=3)

    print(f"Saved model: {model_path}")
    print(f"Saved label encoder: {encoder_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved autoencoder: {autoencoder_path}")
    print(f"Saved zero-day threshold: {threshold_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    default_n_jobs = min(4, max(1, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="Path to train.csv")
    parser.add_argument("--test", help="Path to test.csv")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        help="Per-label fraction used for this proof-of-concept run (default: 0.05)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs)
    parser.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--encoder-out",
        default=str(DEFAULT_ENCODER_PATH),
        help=f"Output LabelEncoder path (default: {DEFAULT_ENCODER_PATH})",
    )
    parser.add_argument(
        "--scaler-out",
        default=str(DEFAULT_SCALER_PATH),
        help=f"Output StandardScaler path (default: {DEFAULT_SCALER_PATH})",
    )
    parser.add_argument(
        "--autoencoder-out",
        default=str(DEFAULT_AUTOENCODER_PATH),
        help=f"Output autoencoder path (default: {DEFAULT_AUTOENCODER_PATH})",
    )
    parser.add_argument(
        "--threshold-out",
        default=str(DEFAULT_THRESHOLD_PATH),
        help=f"Output zero-day threshold path (default: {DEFAULT_THRESHOLD_PATH})",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    train_aegis(args)


if __name__ == "__main__":
    main()
