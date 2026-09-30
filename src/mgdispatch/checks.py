"""Data-quality checks used to VERIFY (not assume) time conventions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sunset_hour_standard_time(doy: np.ndarray, lat: float, lon: float,
                              tz_meridian: float = -120.0) -> np.ndarray:
    """Approximate sunset (NOAA formulae) in local STANDARD time, hours."""
    g = 2 * np.pi / 365 * (np.asarray(doy) - 1)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    cos_ha = (np.cos(np.radians(90.833)) / (np.cos(np.radians(lat)) * np.cos(decl))
              - np.tan(np.radians(lat)) * np.tan(decl))
    ha = np.degrees(np.arccos(np.clip(cos_ha, -1, 1)))
    solar_noon_min = 720 - 4 * (lon - tz_meridian) - eqt
    return (solar_noon_min + 4 * ha) / 60.0


def lighting_offset_by_month(kw: pd.Series, lat: float, lon: float,
                             min_jump_frac: float = 0.02) -> pd.Series:
    """Median (evening switch-on clock time - PST sunset) per month, in hours.

    `kw` must be indexed by interval START in the clock of the raw file (tz-naive).
    Exterior lighting switches on near sunset, so a constant offset of about +3 h
    across ALL months means the clock is fixed UTC-5 (EST); a jump of 1 h between
    winter and summer would indicate daylight saving time.
    """
    jumps = kw.diff()
    rows = []
    for day, g in jumps.groupby(jumps.index.normalize()):
        g = g[g.index.hour >= 14]
        if len(g) < 8 or g.max() < min_jump_frac * kw.max():
            continue
        t = g.idxmax()
        rows.append((day.month, t.hour + t.minute / 60
                     - float(sunset_hour_standard_time(day.dayofyear, lat, lon))))
    df = pd.DataFrame(rows, columns=["month", "offset_h"])
    return df.groupby("month")["offset_h"].median()
