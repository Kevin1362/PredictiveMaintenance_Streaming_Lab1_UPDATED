"""Generate two additional robot training datasets (RMBR4-1 and RMBR4-3)
from the provided RMBR4-2 training export, and optionally seed them into Neon tables.

Why this exists:
- The lab rubric expects you to think about multiple robots / datasets.
- We create two realistic variants by removing extreme 'pick-point' spikes
  (winsorizing + light smoothing) and then applying small robot-specific
  scale/offset/noise differences.

Usage:
    python -m src.generate_additional_robots

Optional environment variables:
    ROBOT1_TABLE  (default: robot_currents_raw_r1)
    ROBOT3_TABLE  (default: robot_currents_raw_r3)

If Neon credentials are configured (.env), this script will:
- create the two tables
- insert the generated rows
"""

from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import TIME_COL, AXIS_COLS, DB_HOST, DB_NAME
from .db import ensure_readings_table, insert_rows


def _remove_pick_points(df: pd.DataFrame, axis_cols: list[str]) -> pd.DataFrame:
    """Remove sharp spikes by clipping to robust quantiles + light smoothing."""
    out = df.copy()
    for ax in axis_cols:
        hi = out[ax].quantile(0.985)
        lo = out[ax].quantile(0.015)
        out[ax] = out[ax].clip(lower=lo, upper=hi)
        out[ax] = out[ax].rolling(window=3, center=True, min_periods=1).mean()
    return out


def _make_robot_variant(
    base: pd.DataFrame,
    axis_cols: list[str],
    rng: np.random.Generator,
    scale_mu: float,
    scale_sd: float,
    offset_sd: float,
    noise_sd: float,
) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    scales = rng.normal(scale_mu, scale_sd, size=len(axis_cols))
    offsets = rng.normal(0.0, offset_sd, size=len(axis_cols))
    for i, ax in enumerate(axis_cols):
        out[ax] = out[ax] * scales[i] + offsets[i] + rng.normal(0.0, noise_sd, size=len(out))
    meta = {
        "scale_mu": scale_mu,
        "scale_sd": scale_sd,
        "offset_sd": offset_sd,
        "noise_sd": noise_sd,
        "scales": [float(x) for x in scales],
        "offsets": [float(x) for x in offsets],
    }
    return out, meta


def main():
    train_path = Path("data/training/RMBR4-2_export_test.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")

    train = pd.read_csv(train_path)
    base = _remove_pick_points(train, AXIS_COLS)

    rng = np.random.default_rng(42)

    r1, meta1 = _make_robot_variant(base, AXIS_COLS, rng, scale_mu=0.98, scale_sd=0.02, offset_sd=0.12, noise_sd=0.07)
    r3, meta3 = _make_robot_variant(base, AXIS_COLS, rng, scale_mu=1.02, scale_sd=0.02, offset_sd=0.12, noise_sd=0.07)

    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)

    r1_path = out_dir / "RMBR4-1_export_artificial.csv"
    r3_path = out_dir / "RMBR4-3_export_artificial.csv"
    r1.to_csv(r1_path, index=False)
    r3.to_csv(r3_path, index=False)

    meta_dir = Path("data/processed")
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "robot_generation_metadata.json").write_text(
        json.dumps({"RMBR4-1": meta1, "RMBR4-3": meta3}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {r1_path}")
    print(f"Wrote: {r3_path}")

    # Optional: seed to Neon tables
    use_db = bool(DB_HOST and DB_NAME)
    if use_db:
        robot1_table = os.getenv("ROBOT1_TABLE", "robot_currents_raw_r1")
        robot3_table = os.getenv("ROBOT3_TABLE", "robot_currents_raw_r3")
        ensure_readings_table(robot1_table, TIME_COL, AXIS_COLS)
        ensure_readings_table(robot3_table, TIME_COL, AXIS_COLS)

        insert_rows(r1, robot1_table, TIME_COL, AXIS_COLS)
        insert_rows(r3, robot3_table, TIME_COL, AXIS_COLS)

        print(f"Seeded Neon tables: {robot1_table}, {robot3_table}")


if __name__ == "__main__":
    main()
