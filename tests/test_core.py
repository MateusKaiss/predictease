import numpy as np
import pandas as pd
import pytest

from predictease.analysis.reader import TimeSeriesDataset
from predictease.core import clip_forecast, get_model, run, run_model

NN_PARAMS = dict(
    window_size=12,
    epochs=1,
    batch_size=16,
    hidden_units=4,
    activation='relu',
)


def generate_seasonal_series(n=96, period=12):
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    values = [float(i % period) for i in range(n)]
    return pd.Series(values, index=dates, name='value')


def generate_dataset(y):
    endog = y.rename_axis('date').reset_index()
    return TimeSeriesDataset(endog=endog)


def test_get_model_arima_uses_seasonal_length():
    y = generate_seasonal_series()
    model = get_model('arima', generate_dataset(y), y, 12, **NN_PARAMS)

    assert model.seasonal is True
    assert model.m == 12


def test_get_model_arima_without_seasonal_length():
    y = generate_seasonal_series()
    model = get_model('arima', generate_dataset(y), y, None, **NN_PARAMS)

    assert model.seasonal is False
    assert model.m == 1


@pytest.mark.parametrize('model', ['naive', 'mean', 'seasonal_naive'])
def test_run_model_fits_baseline_models(model):
    y = generate_seasonal_series()
    forecast = run_model(
        model=model,
        data=generate_dataset(y),
        y=y,
        forecast_steps=5,
        seasonal_length=12,
        **NN_PARAMS,
    )

    assert len(forecast) == 5


def test_run_model_seasonal_naive_repeats_last_season():
    y = generate_seasonal_series()
    forecast = run_model(
        model='seasonal_naive',
        data=generate_dataset(y),
        y=y,
        forecast_steps=24,
        seasonal_length=12,
        **NN_PARAMS,
    )

    expected = [float(i % 12) for i in range(24)]
    assert forecast.tolist() == expected


def test_clip_forecast_series():
    forecast = clip_forecast(pd.Series([-2.0, 0.5, 3.0]))

    assert forecast.tolist() == [0.0, 0.5, 3.0]


def test_clip_forecast_dataframe_keeps_dates():
    dates = pd.date_range('2024-01-01', periods=2, freq='h')
    forecast = clip_forecast(pd.DataFrame({'ds': dates, 'yhat': [-1.0, 2.0]}))

    assert forecast['yhat'].tolist() == [0.0, 2.0]
    assert forecast['ds'].tolist() == dates.tolist()


@pytest.mark.parametrize('non_negative, expected', [(True, 0.0), (False, -1)])
def test_run_model_non_negative(non_negative, expected):
    y = -generate_seasonal_series() - 1
    forecast = run_model(
        model='naive',
        data=generate_dataset(y),
        y=y,
        forecast_steps=3,
        seasonal_length=None,
        non_negative=non_negative,
        **NN_PARAMS,
    )

    assert (np.sign(forecast) == np.sign(expected)).all()


def test_run_with_test_size_reports_metrics(tmp_path, capsys):
    y = generate_seasonal_series()
    path = tmp_path / 'endog.csv'
    generate_dataset(y).endog.to_csv(path, index=False)

    forecast = run(
        str(path), model='seasonal_naive', seasonal_length=12, test_size=24
    )
    output = capsys.readouterr().out

    assert len(forecast) == 24
    assert 'MAE: 0.00' in output
    assert 'RMSE: 0.00' in output
