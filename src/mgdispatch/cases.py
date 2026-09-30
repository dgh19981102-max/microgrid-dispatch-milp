"""Build a simulation case (one building) from the processed 15-min site data."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import read_processed
from .forecast import ForecastConfig, make_forecasts
from .params import BatteryParams, GridParams
from .simulate import Simulation, WindowLayout
from .tariff import Tariff


@dataclass
class SizingRule:
    """System size relative to each building (so small and large buildings compare).

    pv_energy_fraction: PV kWp chosen so that annual PV energy = this share of the
        building's annual consumption.
    bat_power_fraction: battery power = this share of the annual 15-min peak load.
    bat_hours: battery energy / power (duration).
    """
    pv_energy_fraction: float = 0.30
    bat_power_fraction: float = 0.25
    bat_hours: float = 2.0
    battery_template: dict = field(default_factory=lambda: dict(
        eta_charge=0.95, eta_discharge=0.95, soc_min_frac=0.10, soc_max_frac=1.0,
        deg_cost_per_kwh=0.02))


@dataclass
class Case:
    site: str
    building: str
    index: pd.DatetimeIndex
    load: np.ndarray
    pv: np.ndarray
    pv_kwp: float
    battery: BatteryParams
    grid: GridParams
    eval_start: int
    eval_end: int

    @property
    def btype(self) -> str:
        return self.building.split("_")[1]


def list_buildings(processed_dir: Path, site: str) -> list:
    cols = pd.read_csv(processed_dir / f"{site}_15min.csv", nrows=0).columns
    return [c for c in cols if c.startswith(site + "_")]


def load_case(processed_dir: Path, site: str, building: str, sizing: SizingRule,
              eval_start: str = "2018-02-01", horizon_quarters: int = 96,
              grid: Optional[GridParams] = None) -> Case:
    frame = read_processed(processed_dir / f"{site}_15min.csv")
    load = frame[building].to_numpy(float)
    pv1 = frame["pv_kw_per_kwp"].to_numpy(float)
    kwp = sizing.pv_energy_fraction * load.sum() / pv1.sum()
    peak = load.max()
    p_bat = sizing.bat_power_fraction * peak
    bat = BatteryParams(energy_kwh=p_bat * sizing.bat_hours, power_kw=p_bat,
                        **sizing.battery_template)
    idx = frame.index
    start = int(np.searchsorted(idx, pd.Timestamp(eval_start, tz=idx.tz)))
    end = len(idx) - horizon_quarters          # full look-ahead exists for every decision
    end -= (end - start) % 4
    return Case(site, building, idx, load, pv1 * kwp, float(kwp), bat, grid or GridParams(),
                start, end)


def make_simulation(case: Case, tariff: Tariff, layout: WindowLayout = WindowLayout(),
                    terminal_value: float = 0.10, fc_cfg: Optional[ForecastConfig] = None,
                    pool_days: int = 30) -> Simulation:
    fc_cfg = fc_cfg or ForecastConfig(horizon=layout.horizon_quarters)
    decisions = np.arange(0, case.eval_end, 4)
    fc = make_forecasts(case.load, case.pv, case.index, decisions, fc_cfg, pool_days)
    return Simulation(case.index, case.load, case.pv, tariff, case.battery, case.grid, fc,
                      case.eval_start, case.eval_end, terminal_value=terminal_value,
                      layout=layout)
