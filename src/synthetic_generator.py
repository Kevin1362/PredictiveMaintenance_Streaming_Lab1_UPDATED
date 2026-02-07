import numpy as np
import pandas as pd

def generate_synthetic(df_train: pd.DataFrame, time_col: str, axis_cols: list[str], n_rows: int = 3000, seed: int = 42):
    """
    Create synthetic test data with:
    - time metadata similar to training (min, max, typical step)
    - per-axis mean/std similar to training
    """
    rng = np.random.default_rng(seed)

    t_train = df_train[time_col].to_numpy()
    t_min = float(t_train.min())
    steps = np.diff(t_train)
    step = float(np.median(steps)) if len(steps) else 1.0

    time_s = np.arange(t_min, t_min + n_rows * step, step)
    out = pd.DataFrame({time_col: time_s})

    for ax in axis_cols:
        mu = float(df_train[ax].mean())
        sd = float(df_train[ax].std(ddof=0))
        sd = sd if sd > 0 else 1e-6
        out[ax] = rng.normal(mu, sd, size=len(out))

    return out

def inject_anomalies(df: pd.DataFrame, time_col: str, axis: str, start_time: float, duration_s: float, bump: float):
    """
    Adds a sustained positive bump to one axis values directly (same units as your data).
    """
    out = df.copy()
    t = out[time_col].to_numpy()
    mask = (t >= start_time) & (t <= start_time + duration_s)
    out.loc[mask, axis] = out.loc[mask, axis] + bump
    return out
