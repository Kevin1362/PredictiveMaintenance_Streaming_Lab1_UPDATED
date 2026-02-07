import pandas as pd

def fit_train_scalers(df_train: pd.DataFrame, cols: list[str]):
    # Min-Max
    mins = df_train[cols].min()
    maxs = df_train[cols].max()
    # Z-score
    means = df_train[cols].mean()
    stds = df_train[cols].std(ddof=0).replace(0, 1e-9)

    return {
        "min": mins.to_dict(),
        "max": maxs.to_dict(),
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    }

def transform_minmax(df: pd.DataFrame, cols: list[str], scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mn = scalers["min"][c]
        mx = scalers["max"][c]
        denom = (mx - mn) if (mx - mn) != 0 else 1e-9
        out[c] = (out[c] - mn) / denom
    return out

def transform_zscore(df: pd.DataFrame, cols: list[str], scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = scalers["mean"][c]
        sd = scalers["std"][c]
        out[c] = (out[c] - mu) / (sd if sd != 0 else 1e-9)
    return out
