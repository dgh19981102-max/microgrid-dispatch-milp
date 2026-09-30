"""Synthetic load / PV profiles.

USED ONLY FOR UNIT TESTS AND THE STEP-0 SANITY CHECK. All reported project
results use public data (NREL ComStock loads, PVGIS irradiance) instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_office_week(
    start: str = "2023-07-10",
    days: int = 7,
    freq: str = "1h",
    peak_load_kw: float = 250.0,
    base_load_kw: float = 80.0,
    pv_kwp: float = 150.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Office-like load (weekday daytime plateau) and clear-sky-like PV with clouds."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(days * 24 / (pd.Timedelta(freq).seconds / 3600)),
                        freq=freq)
    hour = idx.hour + idx.minute / 60.0
    weekday = idx.dayofweek < 5

    # occupancy plateau 8-18 h with smooth ramps, lower on weekends
    occ = 1 / (1 + np.exp(-(hour - 7.5) * 1.5)) * 1 / (1 + np.exp((hour - 18.5) * 1.5))
    scale = np.where(weekday, 1.0, 0.25)
    # afternoon cooling bump
    cooling = 0.15 * np.exp(-0.5 * ((hour - 15.5) / 2.0) ** 2)
    load = base_load_kw + (peak_load_kw - base_load_kw) * scale * (occ + cooling) / 1.15
    load = load * (1 + 0.05 * rng.standard_normal(len(idx)))

    # PV: half-sine between 6 and 20 h, daily cloudiness factor + hourly noise
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 14.0), 0, None)
    day_num = np.asarray((idx.normalize() - idx[0].normalize()).days)
    cloud_day = rng.uniform(0.55, 1.0, size=days + 1)[day_num]
    pv = pv_kwp * 0.85 * daylight * cloud_day * (1 + 0.08 * rng.standard_normal(len(idx)))
    pv = np.clip(pv, 0, None) * (daylight > 0)

    return pd.DataFrame({"load_kw": np.clip(load, 0, None), "pv_kw": pv}, index=idx)
