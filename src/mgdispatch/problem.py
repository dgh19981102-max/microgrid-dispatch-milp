"""Assemble a DispatchProblem from time-indexed data and a tariff."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .model import DispatchProblem
from .params import BatteryParams, GridParams
from .tariff import Tariff


def infer_dt_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("Need at least two time stamps to infer the step length.")
    # NOTE: do not use index.asi8 - its unit depends on the index resolution
    # (pandas >= 3 defaults to microseconds, older versions to nanoseconds).
    steps = np.diff(index.values).astype("timedelta64[s]").astype(float) / 3600.0
    if not np.allclose(steps, steps[0]):
        raise ValueError("Time index must be regularly spaced.")
    return float(steps[0])


def make_problem(
    index: pd.DatetimeIndex,
    load: np.ndarray,
    pv: np.ndarray,
    tariff: Tariff,
    battery: BatteryParams,
    grid: Optional[GridParams] = None,
    soc0: Optional[float] = None,
    peak_so_far: Optional[Dict[Tuple[str, str], float]] = None,
    **kwargs,
) -> DispatchProblem:
    """Build a DispatchProblem for the window `index`.

    `peak_so_far` is keyed by (charge name, billing month "YYYY-MM") so that the
    rolling-horizon controller does not need to know block numbering.
    """
    blocks = tariff.demand_blocks(index)
    psf_by_idx = {}
    if peak_so_far:
        for b, blk in enumerate(blocks):
            psf_by_idx[b] = float(peak_so_far.get((blk.name, blk.month), 0.0))
    return DispatchProblem(
        dt=infer_dt_hours(index),
        load=load,
        pv=pv,
        price_import=tariff.energy_price(index),
        price_export=tariff.export_price(index),
        battery=battery,
        grid=grid or GridParams(),
        soc0=soc0,
        demand_blocks=blocks,
        peak_so_far=psf_by_idx,
        **kwargs,
    )
