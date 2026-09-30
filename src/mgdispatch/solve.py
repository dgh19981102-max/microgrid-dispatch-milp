"""Solver-agnostic solve wrapper.

The same Pyomo model can be sent to any of the solvers below by changing one
string in the experiment config. Option names differ per solver, so common
options (relative MIP gap, time limit, threads) are translated here.

Licensing notes (see README): HiGHS and CBC are open source. The pip packages
`gurobipy` and `cplex` ship with size-limited free licences (Gurobi: 2000
variables/constraints, CPLEX Community: 1000 variables/constraints), which are
enough for unit tests but not for full stochastic models.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from .model import DispatchProblem, build_model, extract_solution

# user-facing name -> Pyomo solver plugin
SOLVER_PLUGINS = {
    "highs": "appsi_highs",
    "cbc": "cbc",
    "gurobi": "gurobi_direct",
    "cplex": "cplex_direct",
}

# user-facing option -> solver-specific option name
_OPTION_NAMES = {
    "highs": {"mip_gap": "mip_rel_gap", "time_limit": "time_limit", "threads": "threads"},
    "cbc": {"mip_gap": "ratioGap", "time_limit": "seconds", "threads": "threads"},
    "gurobi": {"mip_gap": "MIPGap", "time_limit": "TimeLimit", "threads": "Threads"},
    "cplex": {"mip_gap": "mip_tolerances_mipgap", "time_limit": "timelimit", "threads": "threads"},
}

_OK = {TerminationCondition.optimal, TerminationCondition.locallyOptimal,
       TerminationCondition.maxTimeLimit}


class SolverLicenseLimitError(RuntimeError):
    """Raised when a size-restricted commercial licence rejects the model."""


@dataclass
class SolverConfig:
    name: str = "highs"
    mip_gap: float = 1e-4
    time_limit: Optional[float] = 60.0
    threads: Optional[int] = 1
    tee: bool = False
    # LP relaxation backend used by solve_lp_first: "pyomo" (any solver above) or
    # "highspy" (direct sparse-matrix HiGHS model, ~10x faster; see fastlp.py)
    lp_backend: str = "pyomo"


def available_solvers() -> list:
    out = []
    for name, plugin in SOLVER_PLUGINS.items():
        try:
            if pyo.SolverFactory(plugin).available(exception_flag=False):
                out.append(name)
        except Exception:  # pragma: no cover - plugin import problems
            pass
    return out


def solve_model(m: pyo.ConcreteModel, cfg: SolverConfig) -> dict:
    """Solve a built model; return timing and termination info."""
    if cfg.name not in SOLVER_PLUGINS:
        raise ValueError(f"Unknown solver '{cfg.name}'. Choose from {list(SOLVER_PLUGINS)}.")
    solver = pyo.SolverFactory(SOLVER_PLUGINS[cfg.name])
    names = _OPTION_NAMES[cfg.name]
    for key in ("mip_gap", "time_limit", "threads"):
        val = getattr(cfg, key)
        if val is not None:
            solver.options[names[key]] = val
    t0 = time.perf_counter()
    try:
        res = solver.solve(m, tee=cfg.tee, load_solutions=False)
    except Exception as exc:
        msg = str(exc)
        if "1016" in msg or "size-limited" in msg.lower() or "Model too large" in msg:
            raise SolverLicenseLimitError(
                f"'{cfg.name}' refused the model ({m.nvariables()} variables, "
                f"{m.nconstraints()} constraints): the free licence is size-limited. "
                "Use 'highs' or 'cbc', shrink the model, or install a full licence."
            ) from exc
        raise
    wall = time.perf_counter() - t0
    tc = res.solver.termination_condition
    if tc not in _OK:
        raise RuntimeError(f"Solver '{cfg.name}' failed: termination={tc}")
    m.solutions.load_from(res)
    return {"solver": cfg.name, "termination": str(tc), "wall_time_s": wall}


def solve(problem: DispatchProblem, cfg: Optional[SolverConfig] = None) -> dict:
    """Build, solve and extract in one call."""
    cfg = cfg or SolverConfig()
    t0 = time.perf_counter()
    m = build_model(problem)
    build_s = time.perf_counter() - t0
    info = solve_model(m, cfg)
    sol = extract_solution(m)
    sol.update(info)
    sol["build_time_s"] = build_s
    sol["n_vars"] = m.nvariables()
    sol["n_cons"] = m.nconstraints()
    return sol


def complementarity_violation(sol: dict) -> float:
    """Largest simultaneous charge&discharge or import&export (kW) in a solution."""
    return float(max(np.minimum(sol["ch"], sol["dis"]).max(),
                     np.minimum(sol["imp"], sol["exp"]).max()))


def solve_lp_first(problem: DispatchProblem, cfg: Optional[SolverConfig] = None,
                   tol: float = 1e-6) -> dict:
    """Solve the LP relaxation; fall back to the MILP only if the LP solution uses
    simultaneous charge/discharge or import/export.

    If the LP optimum satisfies complementarity it is feasible for the MILP, and since
    the LP is a relaxation its objective is a lower bound -> it is MILP-optimal.
    The result therefore always equals a MILP optimum; only the run time differs.
    """
    cfg = cfg or SolverConfig()
    lp = dataclasses.replace(problem, integer=False)
    if cfg.lp_backend == "highspy":
        from .fastlp import solve_fast_lp
        sol = solve_fast_lp(lp)
    else:
        sol = solve(lp, cfg)
    viol = complementarity_violation(sol)
    if viol <= tol:
        sol["milp_fallback"] = False
        sol["lp_violation"] = viol
        return sol
    milp = solve(dataclasses.replace(problem, integer=True), cfg)
    milp["milp_fallback"] = True
    milp["lp_violation"] = viol
    return milp
