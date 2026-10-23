"""CSV ingestion and sampling utilities; this module contains no ML logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CICIOT2023_DTYPES, LABEL_COLUMN, PROJECT_ROOT


def load_csv_optimized(path: str | Path) -> pd.DataFrame:
    """Load a CICIoT2023 CSV with the project's fixed memory-efficient schema."""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading {csv_path} ...")
    frame = pd.read_csv(csv_path, dtype=CICIOT2023_DTYPES)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(
            f"{csv_path} does not contain required column {LABEL_COLUMN!r}. "
            f"Columns found: {list(frame.columns)}"
        )

    memory_mb = frame.memory_usage(deep=True).sum() / (1024**2)
    print(
        f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns "
        f"({memory_mb:.1f} MiB using the CICIoT2023 schema)."
    )
    return frame


def stratified_sample(
    frame: pd.DataFrame,
    fraction: float,
    random_state: int,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Return a reproducible, shuffled per-class sample, retaining rare labels."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    if label_column not in frame.columns:
        raise ValueError(f"Missing stratification column: {label_column!r}")

    sampled_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(label_column, sort=False, dropna=False):
        # ``max(1, ...)`` means a rare attack type is never silently discarded.
        rows = min(len(group), max(1, int(np.ceil(len(group) * fraction))))
        sampled_parts.append(group.sample(n=rows, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def reservoir_sample_csv(
    path: str | Path,
    n_rows: int = 5,
    chunksize: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Uniformly sample CSV rows in bounded memory using random reservoir keys.

    Retaining the ``n_rows`` lowest random keys seen across all chunks is
    equivalent to a reservoir sample.  It avoids loading the full test set.
    """

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if n_rows < 1 or chunksize < 1:
        raise ValueError("n_rows and chunksize must both be positive")

    rng = np.random.default_rng(random_state)
    key_column = "_aegis_reservoir_key"
    reservoir: pd.DataFrame | None = None
    print(f"Sampling {n_rows} rows from {csv_path} in chunks of {chunksize:,} ...")

    for chunk in pd.read_csv(csv_path, dtype=CICIOT2023_DTYPES, chunksize=chunksize):
        chunk[key_column] = rng.random(len(chunk))
        candidates = chunk if reservoir is None else pd.concat(
            [reservoir, chunk], ignore_index=True
        )
        reservoir = candidates.nsmallest(n_rows, key_column).reset_index(drop=True)

    if reservoir is None or reservoir.empty:
        raise ValueError(f"No rows were read from {csv_path}")
    return reservoir.drop(columns=key_column).reset_index(drop=True)


def resolve_test_path(argument: str | None) -> Path:
    """Resolve an explicit test CSV path or the conventional project location."""

    if argument:
        path = Path(argument).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Test CSV not found: {path}")
        return path

    candidates = (
        Path.cwd() / "test.csv",
        PROJECT_ROOT / "test.csv",
        PROJECT_ROOT / "archive" / "CICIOT23" / "test" / "test.csv",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    checked = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(f"Could not find test.csv. Checked:\n{checked}")
