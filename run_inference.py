"""Run a bounded-memory AEGIS live-traffic simulation from CICIoT2023 test.csv.

Example:
    python run_inference.py --test archive/CICIOT23/test/test.csv
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from src.config import (
    DEFAULT_AUTOENCODER_PATH,
    DEFAULT_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_THRESHOLD_PATH,
    LABEL_COLUMN,
)
from src.data_loader import reservoir_sample_csv, resolve_test_path
from src.engines import AegisPredictor
from src.router import route_traffic


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI without performing I/O, keeping this entrypoint testable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", help="Path to CICIoT2023 test.csv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--encoder", default=str(DEFAULT_ENCODER_PATH))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER_PATH))
    parser.add_argument("--autoencoder", default=str(DEFAULT_AUTOENCODER_PATH))
    parser.add_argument("--threshold", default=str(DEFAULT_THRESHOLD_PATH))
    parser.add_argument("--rows", type=int, default=5, help="Reservoir sample size")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    # Artifact loading occurs once at startup.  The predictor is intentionally
    # unaware of source IPs and remediation policy.
    predictor = AegisPredictor(
        model_path=args.model,
        encoder_path=args.encoder,
        scaler_path=args.scaler,
        autoencoder_path=args.autoencoder,
        threshold_path=args.threshold,
    )
    test_path = resolve_test_path(args.test)
    sampled_rows = reservoir_sample_csv(
        test_path,
        n_rows=args.rows,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )

    print("\n--- AEGIS inference simulation ---")
    ip_rng = np.random.default_rng(args.random_state)
    for row_number in range(1, len(sampled_rows) + 1):
        # Keep a one-row DataFrame: unlike iterrows(), it preserves numerical
        # dtypes even when the original CSV also includes a string label.
        features = sampled_rows.iloc[[row_number - 1]].drop(
            columns=[LABEL_COLUMN], errors="ignore"
        )
        source_ip = f"192.0.2.{int(ip_rng.integers(1, 255))}"

        started = perf_counter()
        prediction = predictor.predict(features)
        routed = route_traffic(prediction, source_ip, predictor.zero_day_threshold)
        latency_ms = (perf_counter() - started) * 1000.0

        print(
            f"Row {row_number}/{len(sampled_rows)}: "
            f"source_ip={routed['source_ip']} category={routed['category']} "
            f"tier={routed['tier']} action={routed['action']} "
            f"latency={latency_ms:.3f} ms\n"
        )


if __name__ == "__main__":
    main()
