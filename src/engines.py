"""The AEGIS training and prediction engines.

This module owns model behaviour only.  CSV handling belongs to data_loader,
and response decisions belong to router.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    AUTOENCODER_SAMPLE_SIZE,
    BENIGN_LABEL,
    DEFAULT_AUTOENCODER_PATH,
    DEFAULT_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_THRESHOLD_PATH,
    LABEL_COLUMN,
)


def _validate_numeric_features(features: pd.DataFrame) -> None:
    """Protect both estimators from accidental categorical feature columns."""

    invalid = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if invalid:
        raise TypeError(f"All AEGIS features must be numeric; found: {invalid}")


class AegisTrainer:
    """Fit the supervised classifier and benign-only reconstruction engine."""

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        max_bin: int = 256,
        n_jobs: int = 4,
    ) -> None:
        self.random_state = random_state
        self.xgb_options = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "max_bin": max_bin,
            "n_jobs": n_jobs,
        }
        self.model: XGBClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.autoencoder: MLPRegressor | None = None
        self.zero_day_threshold: float | None = None

    def fit(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        benign_features: pd.DataFrame,
        validation_features: pd.DataFrame | None = None,
        validation_labels: pd.Series | None = None,
    ) -> "AegisTrainer":
        """Fit both engines and calculate the benign reconstruction threshold.

        ``benign_features`` must come from the *full* training set rather than
        a small classifier sample, ensuring the autoencoder receives exactly
        ``AUTOENCODER_SAMPLE_SIZE`` normal records.
        """

        if len(train_features) != len(train_labels):
            raise ValueError("train_features and train_labels must have equal length")
        if train_features.empty:
            raise ValueError("Cannot train AEGIS with no training rows")
        _validate_numeric_features(train_features)

        feature_columns = tuple(str(column) for column in train_features.columns)
        if list(benign_features.columns) != list(feature_columns):
            raise ValueError("benign_features columns must match training feature order")
        _validate_numeric_features(benign_features)
        if len(benign_features) < AUTOENCODER_SAMPLE_SIZE:
            raise ValueError(
                f"At least {AUTOENCODER_SAMPLE_SIZE:,} benign rows are required; "
                f"received {len(benign_features):,}."
            )

        labels = train_labels.astype("string").str.strip()
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(labels).astype(np.int32)

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.label_encoder.classes_),
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=self.random_state,
            verbosity=1,
            **self.xgb_options,
        )
        if (validation_features is None) != (validation_labels is None):
            raise ValueError("Supply both validation_features and validation_labels, or neither")
        fit_kwargs: dict[str, Any] = {}
        if validation_features is not None and validation_labels is not None:
            if list(validation_features.columns) != list(feature_columns):
                raise ValueError("validation feature columns must match training order")
            _validate_numeric_features(validation_features)
            unknown = set(validation_labels.astype("string").str.strip()) - set(
                self.label_encoder.classes_
            )
            if unknown:
                raise ValueError(f"Validation labels absent from training: {sorted(unknown)}")
            y_validation = self.label_encoder.transform(
                validation_labels.astype("string").str.strip()
            ).astype(np.int32)
            fit_kwargs = {"eval_set": [(validation_features, y_validation)], "verbose": False}
        self.model.fit(train_features, y_train, **fit_kwargs)

        # A reconstruction network learns normal traffic by mapping its input
        # back to itself.  It is deliberately never exposed to attack rows.
        benign_sample = benign_features.sample(
            n=AUTOENCODER_SAMPLE_SIZE, random_state=self.random_state
        )
        self.scaler = StandardScaler()
        benign_scaled = self.scaler.fit_transform(benign_sample)
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(32, 16, 32), max_iter=50, random_state=self.random_state
        )
        self.autoencoder.fit(benign_scaled, benign_scaled)
        reconstructed = self.autoencoder.predict(benign_scaled)
        errors = np.mean(np.square(benign_scaled - reconstructed), axis=1)
        self.zero_day_threshold = float(np.mean(errors) + 3 * np.std(errors))

        # The schema travels with the model, preventing feature-order drift at
        # deployment when artifacts are loaded independently of training data.
        self.model.aegis_feature_columns_ = feature_columns
        self.model.aegis_category_names_ = tuple(self.label_encoder.classes_)
        return self

    def save_artifacts(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        encoder_path: str | Path = DEFAULT_ENCODER_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        autoencoder_path: str | Path = DEFAULT_AUTOENCODER_PATH,
        threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    ) -> None:
        """Persist the fitted components as independently loadable joblib files."""

        components = (
            (self.model, model_path),
            (self.label_encoder, encoder_path),
            (self.scaler, scaler_path),
            (self.autoencoder, autoencoder_path),
            (self.zero_day_threshold, threshold_path),
        )
        if any(component is None for component, _ in components):
            raise RuntimeError("Call fit() before save_artifacts()")
        for component, output_path in components:
            destination = Path(output_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(component, destination, compress=3)


class AegisPredictor:
    """Load AEGIS artifacts and produce model-only traffic predictions."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        encoder_path: str | Path = DEFAULT_ENCODER_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        autoencoder_path: str | Path = DEFAULT_AUTOENCODER_PATH,
        threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    ) -> None:
        paths = {
            "model": Path(model_path).expanduser().resolve(),
            "label encoder": Path(encoder_path).expanduser().resolve(),
            "scaler": Path(scaler_path).expanduser().resolve(),
            "autoencoder": Path(autoencoder_path).expanduser().resolve(),
            "zero-day threshold": Path(threshold_path).expanduser().resolve(),
        }
        for description, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Saved {description} not found: {path}")

        self.model: Any = joblib.load(paths["model"])
        self.label_encoder: Any = joblib.load(paths["label encoder"])
        self.scaler: Any = joblib.load(paths["scaler"])
        self.autoencoder: Any = joblib.load(paths["autoencoder"])
        self.zero_day_threshold = float(joblib.load(paths["zero-day threshold"]))
        self.feature_columns = self._validate_artifacts(paths)

    def _validate_artifacts(self, paths: dict[str, Path]) -> tuple[str, ...]:
        """Validate interfaces once, at startup, rather than during an alert."""

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"Artifact is not a probability classifier: {paths['model']}")
        if not hasattr(self.label_encoder, "inverse_transform"):
            raise TypeError(f"Artifact is not a label encoder: {paths['label encoder']}")
        if not hasattr(self.scaler, "transform") or not hasattr(self.autoencoder, "predict"):
            raise TypeError("Scaler or autoencoder artifact has an invalid interface")
        if not np.isfinite(self.zero_day_threshold) or self.zero_day_threshold < 0:
            raise ValueError(f"Invalid zero-day threshold: {self.zero_day_threshold!r}")
        columns = getattr(self.model, "aegis_feature_columns_", None)
        if columns is None:
            columns = getattr(self.model, "feature_names_in_", None)
        if columns is None:
            raise ValueError("Saved model has no feature schema; retrain AEGIS first")
        return tuple(str(column) for column in columns)

    def _prepare_features(self, features: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
        """Convert one record into the exact ordered, numeric saved schema."""

        if isinstance(features, pd.Series):
            frame = features.to_frame().T
        elif isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            array = np.asarray(features)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2:
                raise ValueError("features must be a one- or two-dimensional array")
            frame = pd.DataFrame(array, columns=self.feature_columns)

        frame = frame.drop(columns=[LABEL_COLUMN], errors="ignore")
        if len(frame) != 1:
            raise ValueError(f"predict() accepts exactly one record; received {len(frame)}")
        missing = [name for name in self.feature_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Traffic record is missing model features: {missing}")
        frame = frame.loc[:, self.feature_columns]
        for column in self.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        _validate_numeric_features(frame)
        return frame.astype(np.float32, copy=False)

    def predict(self, features: pd.DataFrame | pd.Series | np.ndarray) -> dict[str, float | str]:
        """Return classifier confidence/category and autoencoder reconstruction MSE."""

        frame = self._prepare_features(features)
        probabilities = np.asarray(self.model.predict_proba(frame))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise RuntimeError(f"Unexpected classifier probability shape: {probabilities.shape}")
        prediction_id = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0, prediction_id])
        category = str(self.label_encoder.inverse_transform([prediction_id])[0])

        scaled = np.clip(np.asarray(self.scaler.transform(frame)), -10.0, 10.0)
        reconstructed = np.asarray(self.autoencoder.predict(scaled))
        if reconstructed.ndim == 1:
            reconstructed = reconstructed.reshape(1, -1)
        if reconstructed.shape != scaled.shape:
            raise RuntimeError(
                f"Unexpected autoencoder shape {reconstructed.shape}; expected {scaled.shape}"
            )
        mse = float(np.mean(np.square(scaled - reconstructed)))
        return {
            "confidence": confidence,
            "predicted_category": category,
            "reconstruction_mse": mse,
        }
