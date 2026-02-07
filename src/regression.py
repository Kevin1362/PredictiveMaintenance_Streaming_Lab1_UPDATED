import numpy as np
import pandas as pd

def fit_univariate_lr(x: np.ndarray, y: np.ndarray):
    """
    Closed-form least squares for y = b0 + b1*x
    """
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    b1 = ((x - x_mean) @ (y - y_mean)) / (((x - x_mean) @ (x - x_mean)) + 1e-12)
    b0 = y_mean - b1 * x_mean
    return b0, b1

def predict(b0: float, b1: float, x: np.ndarray):
    return b0 + b1 * x

def fit_models(df_train: pd.DataFrame, time_col: str, axis_cols: list[str]):
    x = df_train[time_col].to_numpy()
    models = {}
    for ax in axis_cols:
        y = df_train[ax].to_numpy()
        b0, b1 = fit_univariate_lr(x, y)
        models[ax] = {"intercept": float(b0), "slope": float(b1)}
    return models

def residuals(df: pd.DataFrame, time_col: str, axis: str, model: dict):
    x = df[time_col].to_numpy()
    y = df[axis].to_numpy()
    yhat = predict(model["intercept"], model["slope"], x)
    r = y - yhat
    return r, yhat
