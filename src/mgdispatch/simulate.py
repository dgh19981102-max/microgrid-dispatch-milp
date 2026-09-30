"""Closed-loop simulation of battery dispatch policies on the 15-min grid.

Every policy is called once per hour (decision step i0) and returns battery set-points
for the next 4 quarter-hours. The PLANT then applies them to the ACTUAL load and PV:

* set-points are clipped to power and SOC limits;
* local safeguard: the battery never discharges into export (dis <= net load);
* optional real-time DEMAND GUARD (as in commercial battery controllers): if the grid
  import in this quarter would exceed an import cap, the controller first reduces
  charging and then increases discharging to hold the cap (within power/SOC limits).
  It can only LOWER import, never raise it. The cap is supplied by the policy: for MPC
  the larger of the month-to-date peak and the planned import of each demand-charge
  block the quarter belongs to (cheapest-binding block wins); for the rule-based policy
  its peak-shaving target. The guard acts on the actual load of the current quarter,
  which is realistic for a local controller that responds within seconds;
* the grid absorbs the rest: import, or export up to the limit, PV curtailed beyond it.

The realised flows are billed afterwards with the independent `compute_bill`.

MPC window (all MPC variants share it, so differences come only from how they treat
uncertainty): 24 h ahead, first 4 h at 15-min resolution and the remaining 20 h as
hourly averages -> 36 optimisation steps instead of 96.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .forecast import STEPS_PER_HOUR, BuildingForecasts
from .model import DispatchProblem
from .params import BatteryParams, GridParams
from .solve import SolverConfig, solve_lp_first
from .tariff import DemandBlock, Tariff

DT_Q = 0.25   # hours per quarter


# ============================================================ window layout
@dataclass(frozen=True)
class WindowLayout:
    fine_steps: int = 16        # 4 h of 15-min steps
    coarse_hours: int = 20      # then 20 hourly steps

    @property
    def horizon_quarters(self) -> int:
        return self.fine_steps + 4 * self.coarse_hours

    @property
    def dt(self) -> np.ndarray:
        return np.r_[np.full(self.fine_steps, DT_Q), np.full(self.coarse_hours, 1.0)]

    def aggregate(self, x: np.ndarray) -> np.ndarray:
        """[..., 96 quarters] -> [..., 36 steps] (coarse part = hourly means)."""
        f = x[..., : self.fine_steps]
        c = x[..., self.fine_steps: self.horizon_quarters]
        c = c.reshape(*c.shape[:-1], self.coarse_hours, 4).mean(axis=-1)
        return np.concatenate([f, c], axis=-1)

    def step_starts(self, i0: int) -> np.ndarray:
        """Quarter positions at which each optimisation step starts."""
        return np.r_[i0 + np.arange(self.fine_steps),
                     i0 + self.fine_steps + 4 * np.arange(self.coarse_hours)]


# ============================================================ plant state
@dataclass
class PlantState:
    soc: float
    peaks: Dict[tuple, float] = field(default_factory=dict)   # (charge, month) -> kW


class Plant:
    """Applies set-points to reality and records realised flows."""

    def __init__(self, load, pv, battery: BatteryParams, grid: GridParams, soc0: float):
        self.load, self.pv, self.bat, self.grid = load, pv, battery, grid
        n = len(load)
        self.rec = {k: np.full(n, np.nan) for k in ("imp", "exp", "ch", "dis", "curt", "soc")}
        self.state = PlantState(soc=soc0)
        self.guard_actions = 0

    def step(self, i: int, p_set: float, cap: Optional[float] = None) -> None:
        b, s = self.bat, self.state
        net0 = self.load[i] - self.pv[i]
        dmax = max(min(b.power_kw, (s.soc - b.soc_min) * b.eta_discharge / DT_Q, max(net0, 0.0)), 0.0)
        if p_set >= 0:   # discharge
            dis, ch = float(np.clip(p_set, 0, dmax)), 0.0
        else:
            cmax = min(b.power_kw, (b.soc_max - s.soc) / (b.eta_charge * DT_Q))
            ch, dis = float(np.clip(-p_set, 0, max(cmax, 0.0))), 0.0
        if cap is not None and net0 + ch - dis > cap:          # real-time demand guard
            excess = net0 + ch - dis - cap
            red = min(ch, excess)
            ch -= red
            excess -= red
            if excess > 0 and ch == 0:
                dis = min(dmax, dis + excess)
            self.guard_actions += 1
        net = net0 + ch - dis
        imp = max(net, 0.0)
        surplus = max(-net, 0.0)
        lim = self.grid.export_limit_kw
        exp = surplus if lim is None else min(surplus, lim)
        curt = surplus - exp
        s.soc = float(np.clip(s.soc + DT_Q * (b.eta_charge * ch - dis / b.eta_discharge),
                              b.soc_min, b.soc_max))
        for k, v in (("imp", imp), ("exp", exp), ("ch", ch), ("dis", dis),
                     ("curt", curt), ("soc", s.soc)):
            self.rec[k][i] = v


def update_peaks(state: PlantState, blocks_at, i: int, imp: float) -> None:
    for name, month in blocks_at(i):
        key = (name, month)
        state.peaks[key] = max(state.peaks.get(key, 0.0), imp)


# ============================================================ policies
class Policy:
    name = "base"

    def setpoints(self, n: int, i0: int, state: PlantState) -> np.ndarray:
        raise NotImplementedError

    def caps(self, n: int, i0: int, state: PlantState):
        """Import caps for the demand guard (length 4) or None (no guard)."""
        return None


class NoBattery(Policy):
    name = "no_battery"

    def setpoints(self, n, i0, state):
        return np.zeros(STEPS_PER_HOUR)


class RuleBased(Policy):
    """Transparent heuristic of the kind often used in practice.

    Uses only the previous quarter's measured net load (causal) and the same load/PV
    forecasts as MPC for the daily peak-shaving target.
      1. Daily target T = max(month-to-date peak, T*), where T* is the lowest threshold
         whose forecast energy above it (next 24 h) fits in the usable battery energy
         and whose excess power fits in the battery power ("energy-feasible threshold").
      2. Shave: if net load > T, discharge the excess.
      3. Peak period (16-21 h): discharge spread evenly over the remaining peak hours.
      4. PV surplus: charge from it.
      5. Night off-peak (0-6 h) or super-off-peak: charge from the grid, keeping import
         at least `margin` below T (the rule reacts one quarter late).
    Note: a first version used T* = forecast max - 0.9 P (power only). It ignored the
    energy limit and emptied the battery before the real peak; see LEARNING_GUIDE.
    """
    name = "rule_based"
    margin = 0.05

    @staticmethod
    def energy_feasible_threshold(fnet: np.ndarray, power: float, energy: float,
                                  dt: float = DT_Q) -> float:
        hi = float(fnet.max())
        lo = max(hi - power, 0.0)
        if np.sum(np.clip(fnet - lo, 0, None)) * dt <= energy:
            return lo
        for _ in range(40):                       # bisection on the threshold
            mid = 0.5 * (lo + hi)
            if np.sum(np.clip(fnet - mid, 0, None)) * dt <= energy:
                hi = mid
            else:
                lo = mid
        return hi

    def __init__(self, sim: "Simulation"):
        self.sim = sim
        self.target = None
        self.day = None

    def setpoints(self, n, i0, state):
        sim, b = self.sim, self.sim.battery
        out = np.zeros(STEPS_PER_HOUR)
        t0 = sim.index[i0]
        if self.day != t0.date():                       # new day: set target
            self.day = t0.date()
            H = min(96, sim.fc.load_fc.shape[1])
            fnet = sim.fc.load_fc[n, :H] - sim.fc.pv_fc[n, :H]
            month = f"{t0.year:04d}-{t0.month:02d}"
            mtd = state.peaks.get(("max_demand", month), 0.0)
            usable = (b.soc_max - b.soc_min) * b.eta_discharge
            t_star = self.energy_feasible_threshold(fnet, b.power_kw, usable)
            self.target = max(mtd, t_star)
        # plan the hour quarter by quarter with a simple internal SOC estimate
        soc = state.soc
        meas = sim.load[i0 - 1] - sim.pv[i0 - 1] if i0 > 0 else sim.load[i0] - sim.pv[i0]
        for q in range(STEPS_PER_HOUR):
            i = i0 + q
            ts = sim.index[i]
            per = sim.period[i]
            net = meas
            p = 0.0
            if net > self.target:
                p = min(b.power_kw, net - self.target)
            elif per == "peak":
                hours_left = max(21 - (ts.hour + ts.minute / 60), 0.25)
                p = min(b.power_kw, max(net, 0.0),
                        (soc - b.soc_min) * b.eta_discharge / hours_left)
            elif net < 0:
                p = -min(b.power_kw, -net)
            elif ts.hour < 6 or per == "super_off_peak":
                p = -min(b.power_kw, max((1 - self.margin) * self.target - net, 0.0))
            # track SOC estimate
            if p >= 0:
                p = min(p, (soc - b.soc_min) * b.eta_discharge / DT_Q)
                soc -= DT_Q * p / b.eta_discharge
            else:
                p = -min(-p, (b.soc_max - soc) / (b.eta_charge * DT_Q))
                soc += DT_Q * b.eta_charge * (-p)
            out[q] = p
        return out

    def caps(self, n, i0, state):
        return np.full(STEPS_PER_HOUR, self.target)


class MPC(Policy):
    """Rolling-horizon optimisation; variants differ only in the scenarios they see.

    mode = "forecast"   : deterministic, point forecast
         = "quantile"   : deterministic, conservative forecast: load + q-quantile and
                          PV + (1-q)-quantile of recent errors per lead time (q = 0.9)
         = "perfect"    : deterministic, actual future (isolates the finite-horizon loss)
         = "stochastic" : expected cost over sampled error scenarios
         = "robust"     : worst case over the same scenarios (min-max)
    error_scale k: forecast = actual + k * (point forecast - actual)  (experiment 05)
    """

    def __init__(self, sim: "Simulation", mode: str = "forecast", n_scen: int = 10,
                 seed: int = 0, error_scale: float = 1.0, quantile: float = 0.9,
                 solver: Optional[SolverConfig] = None):
        self.q = quantile
        self.sim, self.mode, self.n_scen = sim, mode, n_scen
        self.k = error_scale
        self.rng = np.random.default_rng(seed)
        self.solver = solver or SolverConfig(name="highs", mip_gap=1e-4, time_limit=60,
                                             lp_backend="highspy")
        self.name = f"mpc_{mode}" + (f"_k{error_scale:g}" if mode == "forecast" and error_scale != 1 else "")
        self.stats = {"solves": 0, "milp_fallbacks": 0, "solve_time_s": 0.0}

    def _scenarios(self, n):
        fc, H = self.sim.fc, self.sim.layout.horizon_quarters
        la, pa = fc.load_act[n, :H], fc.pv_act[n, :H]
        lf, pf = fc.load_fc[n, :H], fc.pv_fc[n, :H]
        if self.mode == "perfect":
            return la[None], pa[None]
        L = np.clip(la + self.k * (lf - la), 0, None)     # k > 1 can overshoot below 0
        P = np.clip(pa + self.k * (pf - pa), 0, None)
        if self.mode == "forecast":
            return L[None], P[None]
        if self.mode == "quantile":
            cand = fc.sampler.pool(n)
            if len(cand) == 0:
                return lf[None], pf[None]
            el = np.quantile(fc.sampler.el[cand][:, :H], self.q, axis=0)
            ep = np.quantile(fc.sampler.ep[cand][:, :H], 1 - self.q, axis=0)
            return np.clip(lf + el, 0, None)[None], np.clip(pf + ep, 0, None)[None]
        el, ep = fc.sampler.sample(n, self.n_scen, self.rng)
        return np.clip(lf[None] + el[:, :H], 0, None), np.clip(pf[None] + ep[:, :H], 0, None)

    def setpoints(self, n, i0, state):
        sim, lay = self.sim, self.sim.layout
        L, P = self._scenarios(n)
        L, P = np.nan_to_num(L), np.nan_to_num(P)
        steps = lay.step_starts(i0)
        blocks, psf = sim.window_blocks(steps, state)
        prob = DispatchProblem(
            dt=lay.dt, load=lay.aggregate(L), pv=lay.aggregate(P),
            price_import=sim.price_imp[steps], price_export=sim.price_exp[steps],
            battery=sim.battery, grid=sim.grid, soc0=state.soc,
            demand_blocks=blocks, peak_so_far=psf,
            risk="worst_case" if self.mode == "robust" else "expected",
            n_nonanticipative=STEPS_PER_HOUR, terminal_value_per_kwh=sim.terminal_value,
        )
        t0 = time.perf_counter()
        sol = solve_lp_first(prob, self.solver)
        self.stats["solve_time_s"] += time.perf_counter() - t0
        self.stats["solves"] += 1
        self.stats["milp_fallbacks"] += int(sol["milp_fallback"])
        # planned import of the first hour (most permissive scenario) -> guard caps
        self._plan_imp = sol["imp"][:, :STEPS_PER_HOUR].max(axis=0)
        return sol["p_bat"][0, :STEPS_PER_HOUR]

    def caps(self, n, i0, state):
        caps = np.full(STEPS_PER_HOUR, np.inf)
        for q in range(STEPS_PER_HOUR):
            for key in self.sim.blocks_at(i0 + q):
                caps[q] = min(caps[q], max(state.peaks.get(key, 0.0), self._plan_imp[q]))
        return caps


# ============================================================ simulation
class Simulation:
    """Holds one building case and runs a policy over the evaluation window."""

    def __init__(self, index: pd.DatetimeIndex, load: np.ndarray, pv: np.ndarray,
                 tariff: Tariff, battery: BatteryParams, grid: GridParams,
                 fc: BuildingForecasts, eval_start: int, eval_end: int,
                 soc0_frac: float = 0.5, terminal_value: float = 0.10,
                 layout: WindowLayout = WindowLayout()):
        self.index, self.load, self.pv = index, load, pv
        self.tariff, self.battery, self.grid = tariff, battery, grid
        self.fc, self.layout = fc, layout
        self.eval_start, self.eval_end = eval_start, eval_end
        self.soc0 = battery.soc_min + soc0_frac * (battery.soc_max - battery.soc_min)
        self.terminal_value = terminal_value
        self.price_imp = tariff.energy_price(index)
        self.price_exp = tariff.export_price(index)
        self.period = tariff.period_of(index)
        # demand-charge membership of every quarter: list of (name, month)
        self._blocks = tariff.demand_blocks(index)
        self._member = [[] for _ in range(len(index))]
        self._rate = {}
        for blk in self._blocks:
            self._rate[(blk.name, blk.month)] = blk.rate
            for i in np.flatnonzero(blk.mask):
                self._member[i].append((blk.name, blk.month))
        self.dec_pos = {int(d): n for n, d in enumerate(fc.decisions)}

    def blocks_at(self, i: int):
        return self._member[i]

    def window_blocks(self, steps: np.ndarray, state: PlantState):
        keys = {}
        for t, i in enumerate(steps):
            for key in self._member[i]:
                keys.setdefault(key, []).append(t)
        blocks, psf = [], {}
        for b, (key, ts) in enumerate(keys.items()):
            mask = np.zeros(len(steps), dtype=bool)
            mask[ts] = True
            blocks.append(DemandBlock(key[0], key[1], self._rate[key], mask))
            psf[b] = state.peaks.get(key, 0.0)
        return blocks, psf

    def run(self, policy: Policy, checkpoint: Optional[Path] = None,
            checkpoint_every_days: int = 7, progress: bool = False,
            guard: bool = True) -> dict:
        plant = Plant(self.load, self.pv, self.battery, self.grid, self.soc0)
        start = self.eval_start
        if checkpoint is not None and checkpoint.exists():
            ck = np.load(checkpoint, allow_pickle=True)
            for k in plant.rec:
                plant.rec[k] = ck[k]
            plant.state = PlantState(float(ck["state_soc"]), dict(ck["peaks"].item()))
            plant.guard_actions = int(ck["guard_actions"])
            start = int(ck["next_i"])
            if hasattr(policy, "rng") and "rng_state" in ck:
                policy.rng.bit_generator.state = ck["rng_state"].item()
            if hasattr(policy, "stats") and "stats" in ck:
                policy.stats.update(ck["stats"].item())
        t_wall = time.perf_counter()
        last_ck_day = None
        for i0 in range(start, self.eval_end, STEPS_PER_HOUR):
            n = self.dec_pos[i0]
            p = policy.setpoints(n, i0, plant.state)
            caps = policy.caps(n, i0, plant.state) if guard else None
            for q in range(STEPS_PER_HOUR):
                i = i0 + q
                if i >= self.eval_end:
                    break
                cap = None if caps is None or not np.isfinite(caps[q]) else float(caps[q])
                plant.step(i, float(p[q]), cap)
                update_peaks(plant.state, self.blocks_at, i, plant.rec["imp"][i])
            day = self.index[i0].date()
            if (checkpoint is not None and self.index[i0].hour == 0
                    and (last_ck_day is None or (day - last_ck_day).days >= checkpoint_every_days)):
                last_ck_day = day
                self._save_ck(checkpoint, plant, i0 + STEPS_PER_HOUR, policy)
            if progress and self.index[i0].hour == 0 and self.index[i0].day == 1:
                print(f"    {policy.name}: {self.index[i0]:%Y-%m} "
                      f"({time.perf_counter() - t_wall:.0f}s)", flush=True)
        sl = slice(self.eval_start, self.eval_end)
        rec = {k: v[sl] for k, v in plant.rec.items()}
        return {"rec": rec, "stats": getattr(policy, "stats", {}),
                "guard_actions": plant.guard_actions,
                "wall_time_s": time.perf_counter() - t_wall}

    def _save_ck(self, path, plant, next_i, policy):
        extra = {}
        if hasattr(policy, "rng"):
            extra["rng_state"] = np.array(policy.rng.bit_generator.state, dtype=object)
        if hasattr(policy, "stats"):
            extra["stats"] = np.array(policy.stats, dtype=object)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **plant.rec, state_soc=plant.state.soc, next_i=next_i,
                 guard_actions=plant.guard_actions,
                 peaks=np.array(plant.state.peaks, dtype=object), **extra)
        tmp.replace(path)
