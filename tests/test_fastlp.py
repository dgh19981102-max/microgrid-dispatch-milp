"""The direct highspy LP must reproduce the Pyomo LP relaxation exactly."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from mgdispatch import BatteryParams, GridParams, SolverConfig, solve
from mgdispatch.fastlp import solve_fast_lp
from mgdispatch.model import DispatchProblem
from mgdispatch.solve import solve_lp_first
from mgdispatch.tariff import DemandBlock

REF = SolverConfig(name="highs", mip_gap=1e-9)


def random_problem(seed, S=1, risk="expected", **kw):
    rng = np.random.default_rng(seed)
    T = 20
    dt = np.r_[np.full(8, 0.25), np.full(12, 1.0)]
    load = np.clip(50 + 30 * rng.standard_normal((S, T)).cumsum(axis=1) / 4, 0, None)
    pv = np.clip(40 * np.sin(np.linspace(0, np.pi, T)) + 5 * rng.standard_normal((S, T)), 0, None)
    price = rng.choice([0.06, 0.12, 0.19], size=T)
    m1 = np.ones(T, bool)
    m2 = np.zeros(T, bool)
    m2[10:15] = True
    blocks = [DemandBlock("max_demand", "2018-07", 37.37, m1),
              DemandBlock("peak_demand", "2018-07", 46.16, m2)]
    base = dict(dt=dt, load=load, pv=pv, price_import=price, price_export=np.full(T, 0.05),
                battery=BatteryParams(energy_kwh=60, power_kw=30), soc0=30.0,
                demand_blocks=blocks, peak_so_far={0: float(rng.uniform(0, 60)), 1: 0.0},
                risk=risk, n_nonanticipative=4, terminal_value_per_kwh=0.1, integer=False)
    base.update(kw)
    return DispatchProblem(**base)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("S,risk", [(1, "expected"), (5, "expected"), (5, "worst_case")])
def test_fast_lp_matches_pyomo(seed, S, risk):
    p = random_problem(seed, S=S, risk=risk)
    ref = solve(p, REF)
    fast = solve_fast_lp(p)
    assert fast["objective"] == pytest.approx(ref["objective"], rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(fast["cost"], fast["cost"])        # finite
    # the fast solution satisfies the SOC dynamics
    soc, ch, dis = fast["soc"], fast["ch"], fast["dis"]
    pred = soc[:, :-1] + p.dt * (0.95 * ch - dis / 0.95)
    np.testing.assert_allclose(soc[:, 1:], pred, atol=1e-6)


def test_fast_lp_with_export_limit_and_terminal_soc():
    p = random_problem(7, S=3, grid=GridParams(export_limit_kw=5.0), terminal_soc_min=40.0,
                       terminal_value_per_kwh=0.0)
    assert solve_fast_lp(p)["objective"] == pytest.approx(solve(p, REF)["objective"], rel=1e-6)


def test_lp_first_backends_agree():
    p = dataclasses.replace(random_problem(3, S=4), integer=True)
    a = solve_lp_first(p, SolverConfig(name="highs", mip_gap=1e-9, lp_backend="pyomo"))
    b = solve_lp_first(p, SolverConfig(name="highs", mip_gap=1e-9, lp_backend="highspy"))
    assert a["objective"] == pytest.approx(b["objective"], rel=1e-6)


def test_lp_first_falls_back_to_milp_when_needed():
    # negative prices make simultaneous charge/discharge attractive in the LP
    p = random_problem(1, price_import=np.full(20, -0.05), integer=True, demand_blocks=[],
                       peak_so_far={})
    sol = solve_lp_first(p, SolverConfig(name="highs", mip_gap=1e-9, lp_backend="highspy"))
    assert sol["milp_fallback"]
    assert np.minimum(sol["ch"], sol["dis"]).max() <= 1e-6
