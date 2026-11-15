from typing import Dict, Union

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(error)))


def rmse(y_true, y_pred) -> float:
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def evaluate_forecast(
    y_true: pd.Series, forecast: Union[pd.Series, pd.DataFrame]
) -> Dict[str, float]:
    if isinstance(forecast, pd.DataFrame):
        forecast = forecast['yhat']
    return {'mae': mae(y_true, forecast), 'rmse': rmse(y_true, forecast)}
