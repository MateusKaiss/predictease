import sys

from predictease.cli import parse_args


def parse(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['predictease', 'endog.csv', *args])
    return parse_args()


def test_default_arguments(monkeypatch):
    args = parse(monkeypatch)

    assert args.seasonal_length is None
    assert args.hidden_layers is None
    assert args.seed is None
    assert args.test_size is None
    assert args.non_negative is False
    assert args.inmet is False
    assert args.l2 == 0.0
    assert args.dropout == 0.0


def test_hidden_layers_argument(monkeypatch):
    args = parse(monkeypatch, '--hidden_layers', '64', '64')

    assert args.hidden_layers == [64, 64]


def test_new_arguments(monkeypatch):
    args = parse(
        monkeypatch,
        '--model',
        'arima',
        '--seasonal_length',
        '24',
        '--non_negative',
        '--seed',
        '3',
        '--l2',
        '0.01',
        '--dropout',
        '0.2',
        '--test_size',
        '48',
        '--inmet',
        '--utc_offset',
        '-3',
        '--start',
        '2024-01-01',
        '--end',
        '2024-04-01',
    )

    assert args.seasonal_length == 24
    assert args.non_negative is True
    assert args.seed == 3
    assert args.l2 == 0.01
    assert args.dropout == 0.2
    assert args.test_size == 48
    assert args.inmet is True
    assert args.utc_offset == -3
    assert args.start == '2024-01-01'
    assert args.end == '2024-04-01'
