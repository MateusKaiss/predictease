import numpy as np
import pandas as pd
import pytest

from predictease.models.nn_models_univariate import LSTMModel, MLPModel
from predictease.seed import set_seed


def generate_sine_series(n=120, offset=0.0):
    return pd.Series(np.sin(np.arange(n) * 2 * np.pi / 12) + offset)


def layer_names(model):
    return [type(layer).__name__ for layer in model.model.layers]


def test_mlp_default_has_one_hidden_layer():
    model = MLPModel(window_size=12, epochs=1, hidden_units=8)
    model.fit(generate_sine_series())

    assert [layer.units for layer in model.model.layers] == [8, 1]


def test_mlp_hidden_layers_and_activation():
    model = MLPModel(
        window_size=12, epochs=1, hidden_layers=[8, 4], activation='tanh'
    )
    model.fit(generate_sine_series())

    layers = model.model.layers
    assert [layer.units for layer in layers] == [8, 4, 1]
    assert [layer.activation.__name__ for layer in layers[:-1]] == [
        'tanh',
        'tanh',
    ]
    assert layers[-1].activation.__name__ == 'linear'


def test_lstm_hidden_layers():
    model = LSTMModel(window_size=12, epochs=1, hidden_layers=[4, 4])
    model.fit(generate_sine_series())

    assert layer_names(model) == ['LSTM', 'LSTM', 'Dense']
    assert len(model.predict(steps=3)) == 3


def test_mlp_l2_and_dropout():
    model = MLPModel(
        window_size=12, epochs=1, hidden_layers=[8, 4], l2=0.01, dropout=0.2
    )
    model.fit(generate_sine_series())

    assert layer_names(model) == [
        'Dense',
        'Dropout',
        'Dense',
        'Dropout',
        'Dense',
    ]
    assert model.model.layers[0].kernel_regularizer is not None
    assert model.model.layers[1].rate == pytest.approx(0.2)


def test_mlp_without_regularization():
    model = MLPModel(window_size=12, epochs=1, hidden_units=8)
    model.fit(generate_sine_series())

    assert layer_names(model) == ['Dense', 'Dense']
    assert model.model.layers[0].kernel_regularizer is None


@pytest.mark.parametrize('model_cls', [MLPModel, LSTMModel])
def test_seed_makes_forecast_reproducible(model_cls):
    forecasts = []
    for _ in range(2):
        set_seed(7, deep_learning=True)
        model = model_cls(window_size=12, epochs=2, hidden_units=8)
        model.fit(generate_sine_series())
        forecasts.append(model.predict(steps=6).values)

    np.testing.assert_array_equal(forecasts[0], forecasts[1])


def test_different_seeds_give_different_forecasts():
    forecasts = []
    for seed in [1, 2]:
        set_seed(seed, deep_learning=True)
        model = MLPModel(window_size=12, epochs=2, hidden_units=8)
        model.fit(generate_sine_series())
        forecasts.append(model.predict(steps=6).values)

    assert not np.allclose(forecasts[0], forecasts[1])


@pytest.mark.parametrize('model_cls', [MLPModel, LSTMModel])
def test_non_negative_forecast(model_cls):
    y = generate_sine_series(offset=-0.9)
    model = model_cls(
        window_size=12, epochs=2, hidden_units=8, non_negative=True
    )
    model.fit(y)

    assert model.predict(steps=24).min() >= -1e-6


def test_forecast_can_be_negative_by_default():
    y = generate_sine_series(offset=-0.9)
    model = MLPModel(window_size=12, epochs=30, hidden_units=8)
    model.fit(y)

    assert model.predict(steps=24).min() < 0
