from typing import List, Tuple

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 as l2_regularizer


def build_mlp(
    input_dim: int,
    hidden_layers: List[int],
    activation: str,
    l2: float = 0.0,
    dropout: float = 0.0,
) -> Sequential:
    regularizer = l2_regularizer(l2) if l2 > 0 else None
    layers = []
    for index, units in enumerate(hidden_layers):
        params = {'input_shape': (input_dim,)} if index == 0 else {}
        layers.append(
            Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                **params,
            )
        )
        if dropout > 0:
            layers.append(Dropout(dropout))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=Adam(), loss='mse')
    return model


def build_lstm(
    input_shape: Tuple[int, int],
    hidden_layers: List[int],
    activation: str,
    l2: float = 0.0,
    dropout: float = 0.0,
) -> Sequential:
    regularizer = l2_regularizer(l2) if l2 > 0 else None
    layers = []
    for index, units in enumerate(hidden_layers):
        params = {'input_shape': input_shape} if index == 0 else {}
        layers.append(
            LSTM(
                units,
                activation=activation,
                return_sequences=index < len(hidden_layers) - 1,
                kernel_regularizer=regularizer,
                **params,
            )
        )
        if dropout > 0:
            layers.append(Dropout(dropout))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=Adam(), loss='mse')
    return model
