from typing import Dict, Optional

import numpy as np
import pandas as pd

HEADER_ROWS = 8
KJ_M2_TO_W_M2 = 3.6
RADIATION_PREFIX = 'RADIACAO GLOBAL'
CLOCK_CHANGES = {
    'A807': {'start': '2022-08-19 00:00', 'end': '2022-08-22 00:00'},
}


def read_inmet_header(path: str) -> Dict[str, str]:
    header = {}
    with open(path, encoding='latin1') as file:
        for _ in range(HEADER_ROWS):
            key, _, value = file.readline().strip().partition(';')
            header[key.strip().rstrip(':').upper()] = value.strip()
    return header


def solar_elevation(
    times_utc: pd.DatetimeIndex, latitude: float, longitude: float
) -> np.ndarray:
    hour = times_utc.hour + times_utc.minute / 60
    gamma = 2 * np.pi / 365 * (times_utc.dayofyear - 1 + (hour - 12) / 24)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    solar_time = hour * 60 + equation_of_time + 4 * longitude
    hour_angle = np.radians(solar_time / 4 - 180)
    lat = np.radians(latitude)
    sin_elevation = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(
        declination
    ) * np.cos(hour_angle)
    return np.asarray(np.degrees(np.arcsin(sin_elevation)), dtype=float)


def read_inmet(
    path: str,
    utc_offset: int = -3,
    fix_clock: bool = True,
    fill_night: bool = True,
    interpolate: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    header = read_inmet_header(path)
    raw = pd.read_csv(
        path,
        sep=';',
        skiprows=HEADER_ROWS,
        encoding='latin1',
        decimal=',',
        na_values=['', '-9999'],
    )

    radiation_col = [
        col
        for col in raw.columns
        if col.strip().upper().startswith(RADIATION_PREFIX)
    ][0]
    day = raw.iloc[:, 0].astype(str).str.replace('/', '-', regex=False)
    hour = raw.iloc[:, 1].astype(str).str.replace(':', '', regex=False)
    label_utc = pd.to_datetime(
        day + ' ' + hour.str.slice(0, 4), format='%Y-%m-%d %H%M'
    )
    radiation = pd.Series(
        raw[radiation_col].astype(float).values, index=label_utc
    )

    station = header.get('CODIGO (WMO)')
    if fix_clock and station in CLOCK_CHANGES:
        change_start = pd.Timestamp(CLOCK_CHANGES[station]['start'])
        change_end = pd.Timestamp(CLOCK_CHANGES[station]['end'])
        shifted = radiation.index.where(
            radiation.index >= change_start,
            radiation.index + pd.Timedelta(hours=1),
        )
        radiation.index = pd.DatetimeIndex(shifted)
        radiation = radiation[~radiation.index.duplicated(keep='last')]
        outage = (radiation.index >= change_start) & (
            radiation.index < change_end
        )
        radiation[outage] = np.nan

    radiation = radiation.sort_index()
    radiation = radiation.reindex(
        pd.date_range(radiation.index[0], radiation.index[-1], freq='h')
    )
    irradiance = (radiation / KJ_M2_TO_W_M2).clip(lower=0)

    if fill_night:
        elevation = solar_elevation(
            irradiance.index - pd.Timedelta(minutes=30),
            float(header['LATITUDE'].replace(',', '.')),
            float(header['LONGITUDE'].replace(',', '.')),
        )
        irradiance[irradiance.isna().values & (elevation <= 0)] = 0.0

    if interpolate:
        irradiance = irradiance.interpolate(limit_direction='both')

    irradiance.index = irradiance.index + pd.Timedelta(hours=utc_offset - 1)
    irradiance = irradiance.loc[start:end]

    return pd.DataFrame(
        {'date': irradiance.index, 'irradiance': irradiance.values}
    )
