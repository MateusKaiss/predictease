import numpy as np
import pandas as pd
import pytest

from predictease.analysis.inmet import (
    read_inmet,
    read_inmet_header,
    solar_elevation,
)
from predictease.analysis.reader import load_data

HEADER = [
    'REGIAO:;S',
    'UF:;PR',
    'ESTACAO:;CURITIBA',
    'CODIGO (WMO):;{station}',
    'LATITUDE:;-25,4486111',
    'LONGITUDE:;-49,23055554',
    'ALTITUDE:;922,91',
    'DATA DE FUNDACAO:;28/01/03',
    'Data;Hora UTC;PRECIPITAÇÃO TOTAL, HORÁRIO (mm);RADIACAO GLOBAL (Kj/m²);',
]


def write_inmet(path, start, days, radiation, station='A807'):
    lines = [line.format(station=station) for line in HEADER]
    for label in pd.date_range(start, periods=24 * days, freq='h'):
        value = radiation.get(label.hour, '')
        lines.append(f'{label:%Y/%m/%d};{label:%H%M} UTC;0;{value};')
    path.write_text('\n'.join(lines) + '\n', encoding='latin1')
    return str(path)


@pytest.fixture
def inmet_path(tmp_path):
    radiation = {14: '1800', 15: '3600', 16: '-3,6'}
    return write_inmet(tmp_path / 'inmet.csv', '2024-01-01', 3, radiation)


def test_read_inmet_header(inmet_path):
    header = read_inmet_header(inmet_path)

    assert header['CODIGO (WMO)'] == 'A807'
    assert header['LATITUDE'] == '-25,4486111'


def test_read_inmet_converts_units_and_local_time(inmet_path):
    df = read_inmet(inmet_path, interpolate=False)
    series = df.set_index('date')['irradiance']

    assert list(df.columns) == ['date', 'irradiance']
    assert series['2024-01-02 11:00'] == pytest.approx(1000.0)
    assert series['2024-01-02 10:00'] == pytest.approx(500.0)
    assert series.idxmax().hour == 11


def test_read_inmet_keeps_utc_when_offset_is_zero(inmet_path):
    series = read_inmet(inmet_path, utc_offset=0).set_index('date')

    assert series['irradiance'].idxmax().hour == 14


def test_read_inmet_clips_negative_values(inmet_path):
    df = read_inmet(inmet_path)

    assert df['irradiance'].min() == 0.0


def test_read_inmet_fills_night_with_zero(inmet_path):
    series = read_inmet(inmet_path, interpolate=False).set_index('date')[
        'irradiance'
    ]

    assert series['2024-01-02 00:00'] == 0.0
    assert series['2024-01-02 23:00'] == 0.0
    assert np.isnan(series['2024-01-02 08:00'])


def test_read_inmet_keeps_night_gaps_when_disabled(inmet_path):
    df = read_inmet(inmet_path, fill_night=False, interpolate=False)

    assert df['irradiance'].notna().sum() == 9


def test_read_inmet_interpolates_gaps(inmet_path):
    df = read_inmet(inmet_path)

    assert df['irradiance'].notna().all()
    assert len(df) == 72


def test_read_inmet_start_and_end(inmet_path):
    df = read_inmet(inmet_path, start='2024-01-02', end='2024-01-02')

    assert len(df) == 24
    assert df['date'].iloc[0] == pd.Timestamp('2024-01-02 00:00')
    assert df['date'].iloc[-1] == pd.Timestamp('2024-01-02 23:00')


def test_read_inmet_fixes_station_clock_change(tmp_path):
    path = write_inmet(tmp_path / 'inmet.csv', '2022-08-16', 9, {15: '3600'})
    series = read_inmet(path, interpolate=False).set_index('date')[
        'irradiance'
    ]

    assert series['2022-08-17 12:00'] == pytest.approx(1000.0)
    assert series['2022-08-23 11:00'] == pytest.approx(1000.0)
    assert series['2022-08-19':'2022-08-21'].max() == 0.0
    assert series['2022-08-19':'2022-08-21'].isna().sum() > 0


def test_read_inmet_without_clock_fix(tmp_path):
    path = write_inmet(tmp_path / 'inmet.csv', '2022-08-16', 9, {15: '3600'})
    series = read_inmet(path, fix_clock=False).set_index('date')['irradiance']

    assert series['2022-08-17 11:00'] == pytest.approx(1000.0)
    assert series['2022-08-20 11:00'] == pytest.approx(1000.0)


def test_read_inmet_other_station_has_no_clock_change(tmp_path):
    path = write_inmet(
        tmp_path / 'inmet.csv', '2022-08-16', 9, {15: '3600'}, station='A000'
    )
    series = read_inmet(path).set_index('date')['irradiance']

    assert series['2022-08-17 11:00'] == pytest.approx(1000.0)


def test_solar_elevation_day_and_night():
    times = pd.DatetimeIndex(['2024-01-01 04:00', '2024-01-01 15:30'])
    elevation = solar_elevation(times, -25.45, -49.23)

    assert elevation[0] < 0
    assert elevation[1] > 80


def test_load_data_reads_inmet(inmet_path):
    data = load_data(inmet_path, inmet=True, start='2024-01-02')

    assert list(data.endog.columns) == ['date', 'irradiance']
    assert len(data.endog) == 44
    assert data.exog is None
