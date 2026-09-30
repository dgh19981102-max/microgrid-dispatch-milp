"""Ground-truth electricity bill, computed from realised power flows.

This is deliberately independent of the optimisation model: every policy
(rule-based, deterministic MPC, stochastic MPC, ...) is scored by feeding its
realised import/export/discharge series into `compute_bill`. Unit tests check
that the optimiser's own objective agrees with this function.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .tariff import Tariff


def compute_bill(
    index: pd.DatetimeIndex,
    imp: np.ndarray,
    exp: np.ndarray,
    dis: np.ndarray,
    tariff: Tariff,
    dt: float,
    deg_cost_per_kwh: float = 0.0,
) -> Dict[str, float]:
    """Return a cost breakdown in $ (demand charges are charged per calendar month).

    Months that are only partially covered by `index` are billed on the covered
    part only (no pro-rating) - fine for comparing policies over the same window.
    """
    imp = np.asarray(imp, dtype=float)
    exp = np.asarray(exp, dtype=float)
    dis = np.asarray(dis, dtype=float)
    price_imp = tariff.energy_price(index)
    price_exp = tariff.export_price(index)

    out: Dict[str, float] = {
        "energy_import": float(np.sum(dt * price_imp * imp)),
        "export_credit": float(np.sum(dt * price_exp * exp)),
        "degradation": float(np.sum(dt * deg_cost_per_kwh * dis)),
    }
    demand_total = 0.0
    for blk in tariff.demand_blocks(index):
        peak = float(imp[blk.mask].max())
        cost = blk.rate * peak
        key = f"demand:{blk.name}:{blk.month}"
        out[key] = cost
        demand_total += cost
    out["demand"] = demand_total
    # "total" is the dispatch-dependent part (what the optimiser minimises).
    out["total"] = (out["energy_import"] - out["export_credit"]
                    + out["degradation"] + demand_total)
    n_days = len(pd.DatetimeIndex(index).normalize().unique())
    out["customer_charge"] = tariff.customer_charge_per_day * n_days
    out["total_with_fixed"] = out["total"] + out["customer_charge"]
    return out
