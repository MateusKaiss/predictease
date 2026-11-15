import numpy as np
import pandas as pd

from predictease.analysis.metrics import mae
from predictease.models.arima import ARIMAModel

PERIOD = 12


def generate_spike_series(n=132, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    values = [10.0 if i % PERIOD == 6 else 0.0 for i in range(n)]
    return pd.Series(values) + rng.normal(0, noise_std, n)


def test_arima_seasonal_order_uses_period():
    model = ARIMAModel(seasonal=True, m=PERIOD)
    model.fit(generate_spike_series())

    assert model.seasonal_order[3] == PERIOD
    assert sum(model.seasonal_order[:3]) > 0


def test_arima_non_seasonal_has_no_seasonal_order():
    model = ARIMAModel()
    model.fit(generate_spike_series())

    assert tuple(model.seasonal_order) == (0, 0, 0, 0)


def test_arima_seasonal_beats_non_seasonal():
    y = generate_spike_series()
    train, test = y.iloc[:-PERIOD], y.iloc[-PERIOD:]

    seasonal = ARIMAModel(seasonal=True, m=PERIOD)
    seasonal.fit(train)
    non_seasonal = ARIMAModel()
    non_seasonal.fit(train)

    seasonal_error = mae(test, seasonal.predict(steps=PERIOD))
    non_seasonal_error = mae(test, non_seasonal.predict(steps=PERIOD))

    assert seasonal_error < 0.5
    assert seasonal_error < non_seasonal_error
