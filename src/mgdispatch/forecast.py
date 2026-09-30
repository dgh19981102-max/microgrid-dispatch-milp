"""Causal (no look-ahead) forecasts of load and PV, and forecast-error scenarios.

All forecasts are made at a decision step i0 for the next H steps (i0 ... i0+H-1) on a
regular 15-min grid, using ONLY data at positions < i0. Everything is precomputed with
numpy for all decision times at once, because the rolling-horizon simulation needs
~8,000 forecasts per building.

Load model - "similar day + level correction" (no training, fixed a-priori parameters):
    base(j) = 0.5 * x[j - d(j)] + 0.5 * x[j - 7 days]
        d(j) = lag to the most recent day of the same type (weekday / weekend)
    f(i0, k) = base(i0+k) + e0 * exp(-lead_h / tau)
        e0 = mean(x - base) over the last hour before i0 (recent level error)

PV model - "empirical clear-sky envelope x persistence of clearness":
    env(j)  = max of PV at the same time of day over the previous 14 days
    kt_now  = PV / env over the last hour; kt_day = yesterday's energy / env energy
    f(i0,k) = env(i0+k) * (w * kt_now + (1 - w) * kt_day),  w = exp(-lead_h / tau_pv)

These are deliberately simple baselines; experiment 05 scales their errors from 0 to
2x to cover the range from perfect to poor forecasts.

Scenarios for stochastic / robust MPC: error trajectories (load and PV jointly, to keep
their correlation) of PAST decisions made at the same hour of day whose whole horizon
has already been observed are resampled and added to the current point forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96          # 15-min grid
STEPS_PER_HOUR = 4


def _same_daytype_lag(index: pd.DatetimeIndex) -> np.ndarray:
    """Lag in steps to the most recent previous day of the same type."""
    dow = np.asarray(index.dayofweek)
    lag_days = np.ones(len(index), dtype=int)
    lag_days[dow == 0] = 3          # Monday  <- Friday
    lag_days[dow == 5] = 6          # Saturday <- previous Sunday
    lag_days[dow == 6] = 1          # Sunday  <- Saturday
    return lag_days * STEPS_PER_DAY


@dataclass
class ForecastConfig:
    horizon: int = 96               # steps (24 h)
    load_level_tau_h: float = 3.0
    pv_tau_h: float = 2.0
    envelope_days: int = 14


def load_forecast_matrix(x: np.ndarray, index: pd.DatetimeIndex, decisions: np.ndarray,
                         cfg: ForecastConfig) -> np.ndarray:
    """F[n, k]: forecast made at decisions[n] for step decisions[n] + k."""
    H, N = cfg.horizon, len(decisions)
    j = decisions[:, None] + np.arange(H)[None, :]                     # target positions
    jc = np.clip(j, 0, len(x) - 1)
    lag = _same_daytype_lag(index)[jc]
    sd = x[np.clip(jc - lag, 0, None)]
    wk = x[np.clip(jc - 7 * STEPS_PER_DAY, 0, None)]
    base = 0.5 * sd + 0.5 * wk
    # level correction from the last hour before the decision
    hist = decisions[:, None] - np.arange(1, STEPS_PER_HOUR + 1)[None, :]
    hist = np.clip(hist, 0, None)
    hl = _same_daytype_lag(index)[hist]
    hbase = 0.5 * x[np.clip(hist - hl, 0, None)] + 0.5 * x[np.clip(hist - 7 * STEPS_PER_DAY, 0, None)]
    e0 = (x[hist] - hbase).mean(axis=1)
    lead_h = (np.arange(H) + 1) / STEPS_PER_HOUR
    f = base + e0[:, None] * np.exp(-lead_h / cfg.load_level_tau_h)[None, :]
    return np.clip(f, 0, None)


def pv_forecast_matrix(pv: np.ndarray, decisions: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    H, D = cfg.horizon, STEPS_PER_DAY
    n = len(pv)
    # envelope at every position from the previous `envelope_days` days only
    lags = np.arange(1, cfg.envelope_days + 1) * D
    pos = np.arange(n)
    env = np.max(np.stack([pv[np.clip(pos - L, 0, None)] for L in lags]), axis=0)
    j = np.clip(decisions[:, None] + np.arange(H)[None, :], 0, n - 1)
    env_t = env[j]
    # clearness now (last hour) and yesterday (last 24 h), both strictly in the past
    def kt(window):
        h = np.clip(decisions[:, None] - np.arange(1, window + 1)[None, :], 0, None)
        num, den = pv[h].sum(axis=1), env[h].sum(axis=1)
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    k_now, k_day = kt(STEPS_PER_HOUR), kt(D)
    k_day = np.nan_to_num(k_day, nan=0.7)
    k_now = np.where(np.isnan(k_now), k_day, k_now)
    lead_h = (np.arange(H) + 1) / STEPS_PER_HOUR
    w = np.exp(-lead_h / cfg.pv_tau_h)[None, :]
    f = env_t * np.clip(w * k_now[:, None] + (1 - w) * k_day[:, None], 0, 1.2)
    return np.clip(f, 0, None)


def actual_matrix(x: np.ndarray, decisions: np.ndarray, H: int) -> np.ndarray:
    j = decisions[:, None] + np.arange(H)[None, :]
    out = np.full(j.shape, np.nan)
    ok = j < len(x)
    out[ok] = x[j[ok]]
    return out


class ScenarioSampler:
    """Resample past (load, PV) forecast-error trajectories into scenarios."""

    def __init__(self, err_load: np.ndarray, err_pv: np.ndarray, decisions: np.ndarray,
                 hour_of_day: np.ndarray, H: int, pool_days: int = 30):
        self.el, self.ep = err_load, err_pv
        self.dec, self.hod, self.H = decisions, hour_of_day, H
        self.pool_days = pool_days

    def pool(self, n: int) -> np.ndarray:
        """Indices of past decisions usable at decision n (horizon fully observed)."""
        now = self.dec[n]
        done = self.dec + self.H <= now
        same_hour = self.hod == self.hod[n]
        recent = self.dec >= now - self.pool_days * STEPS_PER_DAY
        cand = np.flatnonzero(done & same_hour & recent)
        if len(cand) < 3:          # early in the record: relax the hour-of-day match
            cand = np.flatnonzero(done & recent)
        return cand

    def sample(self, n: int, n_scen: int, rng: np.random.Generator):
        cand = self.pool(n)
        if len(cand) == 0:
            z = np.zeros((n_scen, self.H))
            return z, z
        pick = rng.choice(cand, size=n_scen, replace=len(cand) < n_scen)
        return self.el[pick], self.ep[pick]


@dataclass
class BuildingForecasts:
    decisions: np.ndarray        # step positions of the hourly decisions
    load_fc: np.ndarray          # [N, H] kW
    pv_fc: np.ndarray            # [N, H] kW (already scaled by kWp)
    load_act: np.ndarray         # [N, H]
    pv_act: np.ndarray           # [N, H]
    sampler: ScenarioSampler

    @property
    def err_load(self):
        return self.load_act - self.load_fc

    @property
    def err_pv(self):
        return self.pv_act - self.pv_fc


def make_forecasts(load: np.ndarray, pv: np.ndarray, index: pd.DatetimeIndex,
                   decisions: np.ndarray, cfg: Optional[ForecastConfig] = None,
                   pool_days: int = 30) -> BuildingForecasts:
    cfg = cfg or ForecastConfig()
    lf = load_forecast_matrix(load, index, decisions, cfg)
    pf = pv_forecast_matrix(pv, decisions, cfg)
    la = actual_matrix(load, decisions, cfg.horizon)
    pa = actual_matrix(pv, decisions, cfg.horizon)
    el = np.nan_to_num(la - lf)
    ep = np.nan_to_num(pa - pf)
    hod = np.asarray(index[decisions].hour)
    sampler = ScenarioSampler(el, ep, decisions, hod, cfg.horizon, pool_days)
    return BuildingForecasts(decisions, lf, pf, la, pa, sampler)


def forecast_skill(fc: BuildingForecasts, eval_mask: np.ndarray) -> dict:
    """nMAE (normalised by mean actual) by lead-time bucket over evaluated decisions."""
    out = {}
    for name, f, a in (("load", fc.load_fc, fc.load_act), ("pv", fc.pv_fc, fc.pv_act)):
        f, a = f[eval_mask], a[eval_mask]
        ok = ~np.isnan(a)
        denom = np.nanmean(a)
        for lo, hi, lab in ((0, 4, "0-1h"), (4, 16, "1-4h"), (16, 96, "4-24h")):
            e = np.abs(f[:, lo:hi] - a[:, lo:hi])[ok[:, lo:hi]]
            out[f"{name}_nmae_{lab}"] = float(e.mean() / denom) if denom > 0 else float("nan")
    return out
