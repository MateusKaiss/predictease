from typing import List, Tuple

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_mlp(
    input_dim: int, hidden_layers: List[int], activation: str
) -> Sequential:
    layers = []
    for index, units in enumerate(hidden_layers):
        params = {'input_shape': (input_dim,)} if index == 0 else {}
        layers.append(Dense(units, activation=activation, **params))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=Adam(), loss='mse')
    return model


def build_lstm(
    input_shape: Tuple[int, int], hidden_layers: List[int], activation: str
) -> Sequential:
    layers = []
    for index, units in enumerate(hidden_layers):
        params = {'input_shape': input_shape} if index == 0 else {}
        layers.append(
            LSTM(
                units,
                activation=activation,
                return_sequences=index < len(hidden_layers) - 1,
                **params,
            )
        )
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=Adam(), loss='mse')
    return model
