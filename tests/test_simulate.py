"""Closed-loop simulation: causality, plant physics, guard, checkpoint/resume."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mgdispatch import BatteryParams, GridParams, Tariff
from mgdispatch.forecast import ForecastConfig, load_forecast_matrix, pv_forecast_matrix
from mgdispatch.simulate import (MPC, NoBattery, Plant, RuleBased, Simulation,
                                 WindowLayout)
from mgdispatch.forecast import make_forecasts
from mgdispatch.synthetic import synthetic_office_week

ROOT = Path(__file__).resolve().parents[1]
B19 = str(ROOT / "experiments" / "configs" / "tariff_pge_b19_secondary_2026.yaml")


def small_sim(days=21, eval_days=3):
    df = synthetic_office_week(start="2018-06-25", days=days, freq="15min", seed=1)
    idx = df.index.tz_localize("America/Los_Angeles")
    load, pv = df.load_kw.to_numpy(), df.pv_kw.to_numpy()
    lay = WindowLayout()
    end = len(idx) - lay.horizon_quarters
    start = end - eval_days * 96
    fc = make_forecasts(load, pv, idx, np.arange(0, end, 4), ForecastConfig(horizon=96))
    bat = BatteryParams(energy_kwh=100, power_kw=50)
    return Simulation(idx, load, pv, Tariff.from_yaml(B19), bat, GridParams(), fc, start, end)


def test_forecasts_do_not_use_the_future():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-03-01", periods=40 * 96, freq="15min", tz="America/Los_Angeles")
    x = rng.uniform(10, 50, len(idx))
    pv = np.clip(rng.uniform(-5, 30, len(idx)), 0, None)
    dec = np.array([20 * 96, 30 * 96 + 36])
    cfg = ForecastConfig(horizon=96)
    f1, p1 = load_forecast_matrix(x, idx, dec, cfg), pv_forecast_matrix(pv, dec, cfg)
    for n, d in enumerate(dec):          # scramble everything at/after the decision
        x2, pv2 = x.copy(), pv.copy()
        x2[d:], pv2[d:] = 999.0, 999.0
        f2 = load_forecast_matrix(x2, idx, dec[[n]], cfg)
        p2 = pv_forecast_matrix(pv2, dec[[n]], cfg)
        np.testing.assert_allclose(f2[0], f1[n])
        np.testing.assert_allclose(p2[0], p1[n])


def test_plant_never_discharges_into_export_and_respects_soc():
    bat = BatteryParams(energy_kwh=10, power_kw=100)
    pl = Plant(np.array([5.0, 5.0]), np.array([0.0, 20.0]), bat, GridParams(), soc0=10.0)
    pl.step(0, 100.0)                                  # huge discharge request
    assert pl.rec["dis"][0] <= 5.0 + 1e-9              # limited to the net load
    pl.step(1, 50.0)                                   # PV surplus: no discharge at all
    assert pl.rec["dis"][1] == 0.0
    assert pl.rec["soc"].min() >= bat.soc_min - 1e-9


def test_guard_only_lowers_import():
    bat = BatteryParams(energy_kwh=100, power_kw=50)
    a = Plant(np.array([100.0]), np.array([0.0]), bat, GridParams(), soc0=50.0)
    b = Plant(np.array([100.0]), np.array([0.0]), bat, GridParams(), soc0=50.0)
    a.step(0, -30.0)               # charging would push import to 130
    b.step(0, -30.0, cap=90.0)     # guard: stop charging, then discharge 10
    assert a.rec["imp"][0] == pytest.approx(130.0)
    assert b.rec["imp"][0] == pytest.approx(90.0)
    c = Plant(np.array([100.0]), np.array([0.0]), bat, GridParams(), soc0=50.0)
    c.step(0, 0.0, cap=150.0)      # cap above import: nothing changes
    assert c.rec["imp"][0] == pytest.approx(100.0)


def test_energy_feasible_threshold():
    fnet = np.array([50, 80, 100, 80, 50], float)
    T = RuleBased.energy_feasible_threshold(fnet, power=100, energy=10, dt=1.0)
    assert np.sum(np.clip(fnet - T, 0, None)) == pytest.approx(10, abs=1e-6)
    assert RuleBased.energy_feasible_threshold(fnet, power=30, energy=1e6, dt=1.0) == 70


@pytest.mark.parametrize("mode", ["forecast", "stochastic"])
def test_checkpoint_resume_reproduces_uninterrupted_run(tmp_path, mode):
    sim = small_sim()
    ref = sim.run(MPC(sim, mode, n_scen=3, seed=5))["rec"]
    ck = tmp_path / "ck.npz"

    class Crash(Exception):
        pass

    pol = MPC(sim, mode, n_scen=3, seed=5)
    calls = {"n": 0}
    orig = pol.setpoints

    def crashing(n, i0, st):
        calls["n"] += 1
        if calls["n"] == 40:                   # dies on day 2, after a checkpoint
            raise Crash
        return orig(n, i0, st)

    pol.setpoints = crashing
    with pytest.raises(Crash):
        sim.run(pol, checkpoint=ck, checkpoint_every_days=1)
    assert ck.exists()
    resumed = sim.run(MPC(sim, mode, n_scen=3, seed=5), checkpoint=ck)["rec"]
    for k in ("imp", "soc", "dis"):
        np.testing.assert_allclose(resumed[k], ref[k], atol=1e-6)


def test_no_battery_matches_net_load():
    sim = small_sim(eval_days=1)
    rec = sim.run(NoBattery())["rec"]
    sl = slice(sim.eval_start, sim.eval_end)
    net = sim.load[sl] - sim.pv[sl]
    np.testing.assert_allclose(rec["imp"] - rec["exp"], net, atol=1e-9)
