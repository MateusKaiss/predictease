import pandas as pd
import pytest

from predictease.analysis.metrics import evaluate_forecast, mae, rmse


def test_mae():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(12.5**0.5)


def test_evaluate_forecast_series():
    metrics = evaluate_forecast(pd.Series([1.0, 2.0]), pd.Series([1.0, 4.0]))

    assert metrics == {
        'mae': pytest.approx(1.0),
        'rmse': pytest.approx(2**0.5),
    }


def test_evaluate_forecast_ignores_index():
    y_true = pd.Series([1.0, 2.0], index=[10, 11])

    assert evaluate_forecast(y_true, pd.Series([1.0, 2.0]))['rmse'] == 0


def test_evaluate_forecast_dataframe_uses_yhat():
    forecast = pd.DataFrame({'yhat': [1.0, 2.0], 'yhat_lower': [0.0, 0.0]})

    assert evaluate_forecast(pd.Series([1.0, 2.0]), forecast)['mae'] == 0
