"""Correctness tests for the dispatch MILP on hand-solvable instances."""
import numpy as np
import pandas as pd
import pytest

from mgdispatch import (BatteryParams, DispatchProblem, GridParams, SolverConfig,
                        Tariff, compute_bill, make_problem, solve)
from mgdispatch.tariff import DemandBlock
from mgdispatch.synthetic import synthetic_office_week

from pathlib import Path

TARIFF_PATH = str(Path(__file__).resolve().parents[1] / "experiments" / "configs"
                  / "tariff_tou_demand_placeholder.yaml")
TOL = 1e-4
CFG = SolverConfig(name="highs", mip_gap=1e-9)


def ideal_battery(**kw):
    """Lossless battery with no degradation cost - makes hand calculations easy."""
    base = dict(energy_kwh=10.0, power_kw=10.0, eta_charge=1.0, eta_discharge=1.0,
                soc_min_frac=0.0, soc_max_frac=1.0, deg_cost_per_kwh=0.0)
    base.update(kw)
    return BatteryParams(**base)


def simple_problem(load, pv=None, price=None, **kw):
    load = np.asarray(load, float)
    T = load.size
    pv = np.zeros(T) if pv is None else np.asarray(pv, float)
    price = np.full(T, 0.1) if price is None else np.asarray(price, float)
    kw.setdefault("battery", ideal_battery())
    kw.setdefault("price_export", np.zeros(T))
    return DispatchProblem(dt=1.0, load=load, pv=pv, price_import=price, **kw)


# --------------------------------------------------------------- arbitrage
def test_energy_arbitrage_matches_hand_solution():
    # Buy 10 kWh at 0.1 in hour 0, avoid buying 10 kWh at 0.3 in hour 1.
    p = simple_problem([10, 10], price=[0.1, 0.3], soc0=0.0, terminal_soc_min=0.0)
    sol = solve(p, CFG)
    assert sol["objective"] == pytest.approx(20 * 0.1, abs=TOL)
    np.testing.assert_allclose(sol["imp"][0], [20, 0], atol=TOL)


def test_efficiency_losses_are_accounted():
    # Round-trip efficiency 0.81: buying x at 0.1 and delivering 0.81x at 0.3
    # is still profitable, but the battery (10 kWh) can only hold 10 kWh.
    bat = ideal_battery(eta_charge=0.9, eta_discharge=0.9)
    p = simple_problem([10, 10], price=[0.1, 0.3], battery=bat, soc0=0.0,
                       terminal_soc_min=0.0)
    sol = solve(p, CFG)
    # charge c kW for 1 h -> stored 0.9c <= 10 -> c <= 11.11 but power <= 10 -> c = 10
    # stored 9 kWh -> delivers 8.1 kWh in hour 1.
    expected = (10 + 10) * 0.1 + (10 - 8.1) * 0.3
    assert sol["objective"] == pytest.approx(expected, abs=TOL)


def test_soc_dynamics_and_power_balance_hold():
    df = synthetic_office_week(days=2, seed=3)
    tariff = Tariff.from_yaml(TARIFF_PATH)
    bat = BatteryParams()
    p = make_problem(df.index, df.load_kw.values, df.pv_kw.values, tariff, bat)
    sol = solve(p, CFG)
    soc, ch, dis = sol["soc"][0], sol["ch"][0], sol["dis"][0]
    pred = soc[:-1] + p.dt * (bat.eta_charge * ch - dis / bat.eta_discharge)
    np.testing.assert_allclose(soc[1:], pred, atol=1e-6)
    lhs = sol["imp"][0] - sol["exp"][0]
    rhs = p.load[0] - (p.pv[0] - sol["curt"][0]) + ch - dis
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)
    assert soc.min() >= bat.soc_min - 1e-6 and soc.max() <= bat.soc_max + 1e-6


# ----------------------------------------------------------- demand charge
def _block(T, rate, mask=None, name="max_demand", month="2023-07"):
    mask = np.ones(T, bool) if mask is None else np.asarray(mask, bool)
    return DemandBlock(name=name, month=month, rate=rate, mask=mask)


def test_peak_shaving_hand_solution():
    # Load [10, 50, 10], battery full with 20 kWh, 30 kW. Energy price 0.
    # Discharging 20 kWh in hour 1 lowers the peak to 30 kW -> cost 10 $/kW * 30.
    bat = ideal_battery(energy_kwh=20.0, power_kw=30.0)
    p = simple_problem([10, 50, 10], price=np.zeros(3), battery=bat, soc0=20.0,
                       demand_blocks=[_block(3, 10.0)])
    sol = solve(p, CFG)
    assert sol["objective"] == pytest.approx(300.0, abs=TOL)
    assert sol["peak"][0, 0] == pytest.approx(30.0, abs=TOL)


def test_peak_so_far_only_charges_the_increase():
    # A peak of 40 kW is already on this month's bill. The best we can reach is 30,
    # which is below 40, so the incremental demand cost must be zero.
    bat = ideal_battery(energy_kwh=20.0, power_kw=30.0)
    p = simple_problem([10, 50, 10], price=np.zeros(3), battery=bat, soc0=20.0,
                       demand_blocks=[_block(3, 10.0)], peak_so_far={0: 40.0})
    sol = solve(p, CFG)
    assert sol["objective"] == pytest.approx(0.0, abs=TOL)
    # And if the recorded peak is 25 kW, we pay only for 30 - 25 = 5 kW.
    p2 = simple_problem([10, 50, 10], price=np.zeros(3), battery=bat, soc0=20.0,
                        demand_blocks=[_block(3, 10.0)], peak_so_far={0: 25.0})
    assert solve(p2, CFG)["objective"] == pytest.approx(50.0, abs=TOL)


def test_month_boundary_starts_a_new_peak():
    tariff = Tariff.from_yaml(TARIFF_PATH)
    idx = pd.date_range("2023-06-30 20:00", periods=8, freq="1h")   # June -> July
    blocks = tariff.demand_blocks(idx)
    months = {b.month for b in blocks if b.name == "max_demand"}
    assert months == {"2023-06", "2023-07"}
    # The June block must only contain June hours.
    june = next(b for b in blocks if b.name == "max_demand" and b.month == "2023-06")
    assert june.mask.sum() == 4 and not june.mask[4:].any()


def test_objective_equals_independent_bill():
    """Model objective must equal the ground-truth bill of its own solution."""
    df = synthetic_office_week(days=7, seed=1)
    tariff = Tariff.from_yaml(TARIFF_PATH)
    bat = BatteryParams()
    p = make_problem(df.index, df.load_kw.values, df.pv_kw.values, tariff, bat,
                     terminal_soc_min=bat.soc_min)
    sol = solve(p, CFG)
    bill = compute_bill(df.index, sol["imp"][0], sol["exp"][0], sol["dis"][0], tariff,
                        dt=1.0, deg_cost_per_kwh=bat.deg_cost_per_kwh)
    assert sol["objective"] == pytest.approx(bill["total"], rel=1e-6)


# ------------------------------------------------------------ integrality
def test_milp_forbids_simultaneous_charge_and_discharge():
    # With a NEGATIVE import price, wasting energy is profitable. The LP relaxation
    # exploits this by charging and discharging at the same time (burning energy in
    # the efficiency losses); the MILP must not.
    bat = ideal_battery(eta_charge=0.9, eta_discharge=0.9)
    kw = dict(price=np.full(4, -0.05), battery=bat, soc0=5.0)
    milp = solve(simple_problem([5, 5, 5, 5], integer=True, **kw), CFG)
    lp = solve(simple_problem([5, 5, 5, 5], integer=False, **kw), CFG)
    assert np.all(np.minimum(milp["ch"][0], milp["dis"][0]) <= TOL)
    assert np.any(np.minimum(lp["ch"][0], lp["dis"][0]) > 1.0)
    assert lp["objective"] < milp["objective"] - 1e-3


def test_export_limit_forces_curtailment():
    p = simple_problem([0, 0], pv=[50, 50], price=np.zeros(2), soc0=10.0,
                       battery=ideal_battery(), grid=GridParams(export_limit_kw=20.0),
                       price_export=np.full(2, 0.05))
    sol = solve(p, CFG)
    # Curtailment itself is not unique here (a lossless, free-to-cycle battery can
    # discharge and curtail more at no cost), so check limit + revenue instead.
    assert sol["exp"][0].max() <= 20.0 + TOL
    assert sol["objective"] == pytest.approx(-2 * 20 * 0.05, abs=TOL)
    np.testing.assert_allclose(sol["exp"][0] + sol["ch"][0] - sol["dis"][0]
                               + sol["curt"][0], [50, 50], atol=TOL)


# --------------------------------------------------------- multi-scenario
def test_duplicated_scenarios_equal_deterministic():
    df = synthetic_office_week(days=2, seed=2)
    tariff = Tariff.from_yaml(TARIFF_PATH)
    bat = BatteryParams()
    L, P = df.load_kw.values, df.pv_kw.values
    det = solve(make_problem(df.index, L, P, tariff, bat), CFG)
    sto = solve(make_problem(df.index, np.vstack([L, L, L]), np.vstack([P, P, P]),
                             tariff, bat), CFG)
    assert sto["objective"] == pytest.approx(det["objective"], rel=1e-6)


def test_nonanticipativity_and_worst_case_properties():
    # start at noon so the first (non-anticipative) steps contain real decisions
    df = synthetic_office_week(days=2, seed=4).iloc[12:]
    tariff = Tariff.from_yaml(TARIFF_PATH)
    bat = BatteryParams()
    L, P = df.load_kw.values, df.pv_kw.values
    loads = np.vstack([L * 0.85, L, L * 1.2])
    pvs = np.vstack([P * 1.1, P, P * 0.6])
    kw = dict(n_nonanticipative=3)
    exp_sol = solve(make_problem(df.index, loads, pvs, tariff, bat, risk="expected", **kw), CFG)
    wc_sol = solve(make_problem(df.index, loads, pvs, tariff, bat, risk="worst_case", **kw), CFG)
    for sol in (exp_sol, wc_sol):
        for key in ("ch", "dis"):
            first = sol[key][:, :3]
            np.testing.assert_allclose(first, np.broadcast_to(first[0], first.shape), atol=1e-6)
    # min-max solution has a no-worse worst case; expected solution a no-worse mean
    assert wc_sol["cost"].max() <= exp_sol["cost"].max() + 1e-4
    w = np.full(3, 1 / 3)
    assert w @ exp_sol["cost"] <= w @ wc_sol["cost"] + 1e-4


# --------------------------------------------------------------- validation
def test_infer_dt_independent_of_timestamp_resolution():
    from mgdispatch.problem import infer_dt_hours
    for unit in ("ns", "us", "s"):
        idx = pd.date_range("2023-01-01", periods=5, freq="15min").as_unit(unit)
        assert infer_dt_hours(idx) == pytest.approx(0.25)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simple_problem([1, 2], pv=[1, 2, 3])
    with pytest.raises(ValueError):
        simple_problem([1, 2], soc0=50.0)          # above capacity
    with pytest.raises(ValueError):
        simple_problem([1, 2], risk="cvar")


def test_worst_case_tiebreak_improves_non_worst_scenarios():
    df = synthetic_office_week(days=2, seed=4).iloc[12:]
    tariff = Tariff.from_yaml(TARIFF_PATH)
    bat = BatteryParams()
    L, P = df.load_kw.values, df.pv_kw.values
    loads, pvs = np.vstack([L * 0.85, L, L * 1.2]), np.vstack([P * 1.1, P, P * 0.6])
    kw = dict(risk="worst_case", n_nonanticipative=3)
    pure = solve(make_problem(df.index, loads, pvs, tariff, bat, worst_case_tiebreak=0.0, **kw), CFG)
    tie = solve(make_problem(df.index, loads, pvs, tariff, bat, worst_case_tiebreak=1e-3, **kw), CFG)
    # worst case essentially unchanged, mean cost no worse
    assert tie["cost"].max() == pytest.approx(pure["cost"].max(), rel=1e-4)
    assert tie["cost"].mean() <= pure["cost"].mean() + 1e-4


# ------------------------------------------------------- solver switching
from mgdispatch import available_solvers
from mgdispatch.solve import SolverLicenseLimitError

AVAILABLE = available_solvers()


@pytest.mark.parametrize("name", ["highs", "cbc", "gurobi", "cplex"])
def test_solvers_agree_on_small_instance(name):
    if name not in AVAILABLE:
        pytest.skip(f"{name} not installed")
    # 1 day hourly: ~190 variables -> fits the free CPLEX/Gurobi size limits
    df = synthetic_office_week(days=1, seed=5)
    tariff = Tariff.from_yaml(TARIFF_PATH)
    p = make_problem(df.index, df.load_kw.values, df.pv_kw.values, tariff, BatteryParams(),
                     terminal_soc_min=100.0)
    ref = solve(p, SolverConfig(name="highs", mip_gap=1e-9))
    sol = solve(p, SolverConfig(name=name, mip_gap=1e-9))
    assert sol["objective"] == pytest.approx(ref["objective"], rel=1e-5)


def test_size_limited_licence_gives_clear_error():
    if "cplex" not in AVAILABLE:
        pytest.skip("cplex not installed")
    df = synthetic_office_week(days=7, seed=5)       # ~1350 variables > 1000
    tariff = Tariff.from_yaml(TARIFF_PATH)
    p = make_problem(df.index, df.load_kw.values, df.pv_kw.values, tariff, BatteryParams())
    try:
        solve(p, SolverConfig(name="cplex"))
    except SolverLicenseLimitError:
        return
    pytest.skip("cplex accepted the model (full licence installed)")
