"""Battery dispatch optimisation model (MILP) written in Pyomo.

One model covers three use cases:

* deterministic optimisation      -> 1 scenario
* two-stage stochastic programme  -> S scenarios, risk="expected"
* scenario-based robust (min-max) -> S scenarios, risk="worst_case"
  (plus a small expected-cost tie-breaker, see `worst_case_tiebreak`)

Decision variables for scenario s and step t (all >= 0, kW unless stated):
    ch[s,t]    battery charging power
    dis[s,t]   battery discharging power
    soc[s,k]   stored energy (kWh) at the START of step k, k = 0..T
    imp[s,t]   grid import
    exp[s,t]   grid export
    curt[s,t]  PV curtailment
    ub[s,t]    1 if the battery is (allowed to be) charging, 0 if discharging
    ug[s,t]    1 if the PCC is importing, 0 if exporting
    pk[s,b]    peak demand (kW) of demand block b (per billing month & charge)

Power balance (load and PV are scenario data):
    imp - exp = load - (pv - curt) + ch - dis

Battery:
    soc[k+1] = soc[k] + dt*(eta_c*ch - dis/eta_d)
    ch  <= P * ub,   dis <= P * (1 - ub)          (no simultaneous charge/discharge)
    imp <= M_imp*ug, exp <= M_exp*(1 - ug)         (no simultaneous import/export)

Variable step lengths dt[t] are allowed (e.g. 15-min steps first, hourly later in an
MPC window). A terminal value (credit per kWh left in the battery) can replace a hard
terminal SOC constraint so that a finite-horizon controller does not empty the battery
just because the window ends.

Demand charge with rolling horizons: `peak_so_far[b]` is the highest import already
recorded in this billing month BEFORE the window starts. Only the increase above it
is costed:
    pk[s,b] >= peak_so_far[b],   pk[s,b] >= imp[s,t]  for t in block b
    demand cost = rate_b * (pk[s,b] - peak_so_far[b])

Non-anticipativity: the first `n_nonanticipative` battery decisions (ch, dis, ub)
must be identical across scenarios - they are what we actually implement now.
Import/export/curtailment remain scenario-dependent "recourse", because in reality
the grid absorbs whatever mismatch occurs.

Setting integer=False relaxes ub, ug to [0, 1] (LP relaxation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyomo.environ as pyo

from .params import BatteryParams, GridParams
from .tariff import DemandBlock


@dataclass
class DispatchProblem:
    """All numerical data needed to build one optimisation window."""
    dt: object                                 # step length(s) in hours: scalar or [T]
    load: np.ndarray                           # [S, T] kW
    pv: np.ndarray                             # [S, T] kW (available PV power)
    price_import: np.ndarray                   # [T] $/kWh
    price_export: np.ndarray                   # [T] $/kWh
    battery: BatteryParams
    grid: GridParams = field(default_factory=GridParams)
    soc0: Optional[float] = None               # kWh; default = midpoint of band
    demand_blocks: List[DemandBlock] = field(default_factory=list)
    peak_so_far: Optional[Dict[int, float]] = None   # block index -> kW (default 0)
    weights: Optional[np.ndarray] = None       # [S] scenario probabilities
    terminal_soc_min: Optional[float] = None   # kWh; None -> no terminal constraint
    terminal_value_per_kwh: float = 0.0        # $/kWh credited for energy left at the end
    risk: str = "expected"                     # expected | worst_case
    worst_case_tiebreak: float = 1e-3          # weight of E[cost] added to min-max objective
    n_nonanticipative: int = 1
    integer: bool = True

    def __post_init__(self) -> None:
        self.load = np.atleast_2d(np.asarray(self.load, dtype=float))
        self.pv = np.atleast_2d(np.asarray(self.pv, dtype=float))
        self.price_import = np.asarray(self.price_import, dtype=float)
        self.price_export = np.asarray(self.price_export, dtype=float)
        S, T = self.load.shape
        self.dt = np.broadcast_to(np.asarray(self.dt, dtype=float), (T,)).copy()
        if np.any(self.dt <= 0):
            raise ValueError("Step lengths must be positive.")
        if self.pv.shape != (S, T):
            raise ValueError(f"pv shape {self.pv.shape} != load shape {(S, T)}")
        if self.price_import.shape != (T,) or self.price_export.shape != (T,):
            raise ValueError("Price vectors must have length T.")
        for b in self.demand_blocks:
            if b.mask.shape != (T,):
                raise ValueError("Demand block mask must have length T.")
        if self.weights is None:
            self.weights = np.full(S, 1.0 / S)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (S,) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must have shape [S] and sum to 1.")
        if self.risk not in ("expected", "worst_case"):
            raise ValueError("risk must be 'expected' or 'worst_case'.")
        if np.any(self.pv < 0) or np.any(self.load < 0):
            raise ValueError("Load and PV must be non-negative.")
        self.battery.validate()
        if self.soc0 is None:
            self.soc0 = 0.5 * (self.battery.soc_min + self.battery.soc_max)
        if not (self.battery.soc_min - 1e-9 <= self.soc0 <= self.battery.soc_max + 1e-9):
            raise ValueError("soc0 outside the allowed SOC band.")
        if self.peak_so_far is None:
            self.peak_so_far = {}
        self.n_nonanticipative = int(min(max(self.n_nonanticipative, 0), T))

    @property
    def n_scenarios(self) -> int:
        return self.load.shape[0]

    @property
    def n_steps(self) -> int:
        return self.load.shape[1]


def build_model(p: DispatchProblem) -> pyo.ConcreteModel:
    """Translate a DispatchProblem into a Pyomo ConcreteModel."""
    S, T = p.n_scenarios, p.n_steps
    bat, grid = p.battery, p.grid
    dt = [float(x) for x in p.dt]

    m = pyo.ConcreteModel(name="battery_dispatch")
    m.S = pyo.RangeSet(0, S - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.K = pyo.RangeSet(0, T)                  # SOC time points
    m.B = pyo.RangeSet(0, len(p.demand_blocks) - 1) if p.demand_blocks else pyo.Set(initialize=[])

    # Tight big-M values derived from physical limits (tighter M -> better LP bound).
    max_imp = float(p.load.max() + bat.power_kw)
    if grid.import_limit_kw is not None:
        max_imp = min(max_imp, float(grid.import_limit_kw))
    max_exp = float(p.pv.max() + bat.power_kw)
    if grid.export_limit_kw is not None:
        max_exp = min(max_exp, float(grid.export_limit_kw))

    bin_domain = pyo.Binary if p.integer else pyo.UnitInterval
    nn = pyo.NonNegativeReals

    m.ch = pyo.Var(m.S, m.T, domain=nn, bounds=(0, bat.power_kw))
    m.dis = pyo.Var(m.S, m.T, domain=nn, bounds=(0, bat.power_kw))
    m.soc = pyo.Var(m.S, m.K, domain=nn, bounds=(bat.soc_min, bat.soc_max))
    m.imp = pyo.Var(m.S, m.T, domain=nn, bounds=(0, max_imp))
    m.exp = pyo.Var(m.S, m.T, domain=nn, bounds=(0, max_exp))
    m.curt = pyo.Var(m.S, m.T, domain=nn)
    m.ub = pyo.Var(m.S, m.T, domain=bin_domain)
    m.ug = pyo.Var(m.S, m.T, domain=bin_domain)
    m.pk = pyo.Var(m.S, m.B, domain=nn)

    # ---- power balance and PV curtailment
    m.balance = pyo.Constraint(
        m.S, m.T,
        rule=lambda m, s, t: m.imp[s, t] - m.exp[s, t]
        == p.load[s, t] - (p.pv[s, t] - m.curt[s, t]) + m.ch[s, t] - m.dis[s, t],
    )
    m.curt_ub = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.curt[s, t] <= p.pv[s, t])

    # ---- battery dynamics
    m.soc_init = pyo.Constraint(m.S, rule=lambda m, s: m.soc[s, 0] == p.soc0)
    m.soc_dyn = pyo.Constraint(
        m.S, m.T,
        rule=lambda m, s, t: m.soc[s, t + 1]
        == m.soc[s, t] + dt[t] * (bat.eta_charge * m.ch[s, t] - m.dis[s, t] / bat.eta_discharge),
    )
    if p.terminal_soc_min is not None:
        tsoc = float(np.clip(p.terminal_soc_min, bat.soc_min, bat.soc_max))
        m.soc_term = pyo.Constraint(m.S, rule=lambda m, s: m.soc[s, T] >= tsoc)

    # ---- mutual exclusivity (big-M with physical limits)
    m.ch_mode = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.ch[s, t] <= bat.power_kw * m.ub[s, t])
    m.dis_mode = pyo.Constraint(
        m.S, m.T, rule=lambda m, s, t: m.dis[s, t] <= bat.power_kw * (1 - m.ub[s, t])
    )
    m.imp_mode = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.imp[s, t] <= max_imp * m.ug[s, t])
    m.exp_mode = pyo.Constraint(
        m.S, m.T, rule=lambda m, s, t: m.exp[s, t] <= max_exp * (1 - m.ug[s, t])
    )

    # ---- demand charges
    psf = {b: float(p.peak_so_far.get(b, 0.0)) for b in range(len(p.demand_blocks))}
    if p.demand_blocks:
        m.pk_floor = pyo.Constraint(m.S, m.B, rule=lambda m, s, b: m.pk[s, b] >= psf[b])
        pk_pairs = [
            (s, b, t)
            for s in range(S)
            for b, blk in enumerate(p.demand_blocks)
            for t in np.flatnonzero(blk.mask)
        ]
        m.PK = pyo.Set(initialize=pk_pairs, dimen=3)
        m.pk_def = pyo.Constraint(m.PK, rule=lambda m, s, b, t: m.pk[s, b] >= m.imp[s, t])

    # ---- non-anticipativity of the implemented battery decisions
    na = range(p.n_nonanticipative)
    if S > 1 and p.n_nonanticipative > 0:
        m.NA = pyo.Set(initialize=[(s, t) for s in range(1, S) for t in na], dimen=2)
        m.na_ch = pyo.Constraint(m.NA, rule=lambda m, s, t: m.ch[s, t] == m.ch[0, t])
        m.na_dis = pyo.Constraint(m.NA, rule=lambda m, s, t: m.dis[s, t] == m.dis[0, t])
        m.na_ub = pyo.Constraint(m.NA, rule=lambda m, s, t: m.ub[s, t] == m.ub[0, t])

    # ---- cost of each scenario
    def energy_cost(m, s):
        return sum(
            dt[t] * (p.price_import[t] * m.imp[s, t]
                     - p.price_export[t] * m.exp[s, t]
                     + bat.deg_cost_per_kwh * m.dis[s, t])
            for t in m.T
        ) - p.terminal_value_per_kwh * m.soc[s, T]

    def demand_cost(m, s):
        return sum(blk.rate * (m.pk[s, b] - psf[b]) for b, blk in enumerate(p.demand_blocks))

    m.cost_energy = pyo.Expression(m.S, rule=energy_cost)
    m.cost_demand = pyo.Expression(m.S, rule=demand_cost)
    m.cost = pyo.Expression(m.S, rule=lambda m, s: m.cost_energy[s] + m.cost_demand[s])

    if p.risk == "expected":
        m.obj = pyo.Objective(expr=sum(float(p.weights[s]) * m.cost[s] for s in m.S),
                              sense=pyo.minimize)
    else:  # worst_case: epigraph formulation of min max_s cost_s
        # A pure min-max objective is indifferent to every non-worst scenario, so
        # the solver may return needlessly bad costs for them. A small expected-cost
        # term breaks these ties (approximately lexicographic: worst case first).
        m.z = pyo.Var(domain=pyo.Reals)
        m.epi = pyo.Constraint(m.S, rule=lambda m, s: m.z >= m.cost[s])
        eps = float(p.worst_case_tiebreak)
        m.obj = pyo.Objective(
            expr=m.z + eps * sum(float(p.weights[s]) * m.cost[s] for s in m.S),
            sense=pyo.minimize,
        )

    m._problem = p          # keep a reference for result extraction
    m._peak_so_far = psf
    return m


def extract_solution(m: pyo.ConcreteModel) -> dict:
    """Pull variable values out of a solved model into numpy arrays."""
    p: DispatchProblem = m._problem
    S, T = p.n_scenarios, p.n_steps

    def arr(var, shape_t):
        return np.array([[pyo.value(var[s, t]) for t in range(shape_t)] for s in range(S)])

    out = {
        "ch": arr(m.ch, T),
        "dis": arr(m.dis, T),
        "soc": arr(m.soc, T + 1),
        "imp": arr(m.imp, T),
        "exp": arr(m.exp, T),
        "curt": arr(m.curt, T),
        "ub": arr(m.ub, T),
        "ug": arr(m.ug, T),
        "cost": np.array([pyo.value(m.cost[s]) for s in range(S)]),
        "cost_energy": np.array([pyo.value(m.cost_energy[s]) for s in range(S)]),
        "cost_demand": np.array([pyo.value(m.cost_demand[s]) for s in range(S)]),
        "objective": float(pyo.value(m.obj)),
    }
    if p.demand_blocks:
        out["peak"] = np.array(
            [[pyo.value(m.pk[s, b]) for b in range(len(p.demand_blocks))] for s in range(S)]
        )
    # net battery power (+ = discharging into the site), convenient for simulation
    out["p_bat"] = out["dis"] - out["ch"]
    return out
