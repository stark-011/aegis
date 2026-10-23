"""Central, side-effect-free configuration for the AEGIS application.

Keeping paths, thresholds, and the CSV schema here means the training and
inference layers use exactly the same contract without importing each other.
"""

from pathlib import Path

import numpy as np


# Project locations.  ``parents[1]`` is the repository root because this file
# lives at ``<project>/src/config.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Persisted model artifacts.  These defaults can be overridden at the CLI.
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "aegis_xgb_model.joblib"
DEFAULT_ENCODER_PATH = ARTIFACT_DIR / "aegis_label_encoder.joblib"
DEFAULT_SCALER_PATH = ARTIFACT_DIR / "aegis_scaler.joblib"
DEFAULT_AUTOENCODER_PATH = ARTIFACT_DIR / "aegis_autoencoder.joblib"
DEFAULT_THRESHOLD_PATH = ARTIFACT_DIR / "aegis_threshold.joblib"

# Dataset/model constants shared by training and inference.
LABEL_COLUMN = "label"
BENIGN_LABEL = "BenignTraffic"
AUTOENCODER_SAMPLE_SIZE = 50_000
XGB_CONFIDENCE_THRESHOLD = 0.80
GREY_ZONE_CONFIDENCE_THRESHOLD = 0.50

# CICIoT2023 has a stable feature schema.  Specifying it to pandas prevents
# accidental float64 expansion when reading the very large CSV files.  The
# target is intentionally a pandas StringDtype, so category cleanup is safe.
CICIOT2023_DTYPES: dict[str, object] = {
    "flow_duration": np.float32,
    "Header_Length": np.float32,
    "Protocol Type": np.float32,
    "Duration": np.float32,
    "Rate": np.float32,
    "Srate": np.float32,
    "Drate": np.float32,
    "fin_flag_number": np.float32,
    "syn_flag_number": np.float32,
    "rst_flag_number": np.float32,
    "psh_flag_number": np.float32,
    "ack_flag_number": np.float32,
    "ece_flag_number": np.float32,
    "cwr_flag_number": np.float32,
    "ack_count": np.float32,
    "syn_count": np.float32,
    "fin_count": np.float32,
    "urg_count": np.float32,
    "rst_count": np.float32,
    "HTTP": np.float32,
    "HTTPS": np.float32,
    "DNS": np.float32,
    "Telnet": np.float32,
    "SMTP": np.float32,
    "SSH": np.float32,
    "IRC": np.float32,
    "TCP": np.float32,
    "UDP": np.float32,
    "DHCP": np.float32,
    "ARP": np.float32,
    "ICMP": np.float32,
    "IPv": np.float32,
    "LLC": np.float32,
    "Tot sum": np.float32,
    "Min": np.float32,
    "Max": np.float32,
    "AVG": np.float32,
    "Std": np.float32,
    "Tot size": np.float32,
    "IAT": np.float32,
    "Number": np.float32,
    "Magnitue": np.float32,
    "Radius": np.float32,
    "Covariance": np.float32,
    "Variance": np.float32,
    "Weight": np.float32,
    LABEL_COLUMN: "string",
}
