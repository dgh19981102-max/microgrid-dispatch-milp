"""Fast path: the LP relaxation of `model.build_model`, assembled with vectorised numpy
as a sparse matrix and solved with the HiGHS Python API (highspy).

Why: in the rolling-horizon simulation the problem is small (a few thousand variables)
but solved ~8,000 times per building and policy. Profiling showed that >85 % of the time
went into Pyomo model construction, not into HiGHS. The Pyomo model remains the
REFERENCE formulation (readable, solver-agnostic); `tests/test_fastlp.py` checks that
both give the same optimum on random problems. Only the LP relaxation is implemented:
if its solution violates complementarity, `solve.solve_lp_first` falls back to the
Pyomo MILP.

Variable layout per scenario s (T steps, B demand blocks):
    ch[T] dis[T] soc[T+1] imp[T] exp[T] curt[T] ub[T] ug[T] pk[B]
followed by one epigraph variable z for the worst-case objective.
"""
from __future__ import annotations

import time

import highspy
import numpy as np
from scipy import sparse

from .model import DispatchProblem

INF = highspy.kHighsInf


class _Rows:
    """Collects constraint blocks as COO triplets."""

    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []
        self.n = 0

    def add(self, cols, vals, lo, hi):
        """Element-wise block: every array in `cols` has the same shape; constraint k
        involves element k of each array. lo/hi: scalar or array of that shape."""
        shape = np.shape(cols[0])
        m = int(np.prod(shape))
        ridx = self.n + np.arange(m)
        for c, v in zip(cols, vals):
            self.r.append(ridx)
            self.c.append(np.asarray(c).ravel())
            self.v.append(np.broadcast_to(np.asarray(v, float), shape).ravel())
        self.lo.append(np.broadcast_to(np.asarray(lo, float), shape).ravel())
        self.hi.append(np.broadcast_to(np.asarray(hi, float), shape).ravel())
        self.n += m

    def add_row(self, cols, vals, lo, hi):
        """One constraint sum_j vals[j] * x[cols[j]] in [lo, hi]."""
        cols = np.asarray(cols).ravel()
        self.r.append(np.full(cols.size, self.n))
        self.c.append(cols)
        self.v.append(np.asarray(vals, float).ravel())
        self.lo.append(np.array([lo], float))
        self.hi.append(np.array([hi], float))
        self.n += 1


def solve_fast_lp(p: DispatchProblem) -> dict:
    S, T = p.n_scenarios, p.n_steps
    B = len(p.demand_blocks)
    bat, grid = p.battery, p.grid
    dt = np.asarray(p.dt, float)
    P = bat.power_kw

    max_imp = float(p.load.max() + P)
    if grid.import_limit_kw is not None:
        max_imp = min(max_imp, float(grid.import_limit_kw))
    max_exp = float(p.pv.max() + P)
    if grid.export_limit_kw is not None:
        max_exp = min(max_exp, float(grid.export_limit_kw))

    worst = p.risk == "worst_case"
    nv_s = 8 * T + 1 + B
    nv = S * nv_s + (1 if worst else 0)
    o = (np.arange(S) * nv_s)[:, None]
    t = np.arange(T)[None, :]
    CH, DIS = o + t, o + T + t
    SOC = o + 2 * T + np.arange(T + 1)[None, :]
    IMP, EXP = o + 3 * T + 1 + t, o + 4 * T + 1 + t
    CURT, UB, UG = o + 5 * T + 1 + t, o + 6 * T + 1 + t, o + 7 * T + 1 + t
    PK = o + 8 * T + 1 + np.arange(B)[None, :]

    lo, hi = np.zeros(nv), np.full(nv, INF)
    hi[CH], hi[DIS] = P, P
    lo[SOC], hi[SOC] = bat.soc_min, bat.soc_max
    lo[SOC[:, 0]] = hi[SOC[:, 0]] = p.soc0
    if p.terminal_soc_min is not None:
        tsoc = float(np.clip(p.terminal_soc_min, bat.soc_min, bat.soc_max))
        lo[SOC[:, T]] = np.maximum(lo[SOC[:, T]], tsoc)
    hi[IMP], hi[EXP] = max_imp, max_exp
    hi[CURT] = p.pv
    hi[UB], hi[UG] = 1.0, 1.0
    psf = np.array([float(p.peak_so_far.get(b, 0.0)) for b in range(B)])
    rates = np.array([blk.rate for blk in p.demand_blocks], float)
    if B:
        lo[PK] = psf[None, :]

    R = _Rows()
    one = np.ones((S, T))
    R.add([IMP, EXP, CURT, CH, DIS], [1, -1, -1, -1, 1], (p.load - p.pv), (p.load - p.pv))
    R.add([SOC[:, 1:], SOC[:, :-1], CH, DIS],
          [1, -1, -dt * bat.eta_charge * one, dt / bat.eta_discharge * one], 0.0, 0.0)
    R.add([CH, UB], [1, -P], -INF, 0.0)
    R.add([DIS, UB], [1, P], -INF, P)
    R.add([IMP, UG], [1, -max_imp], -INF, 0.0)
    R.add([EXP, UG], [1, max_exp], -INF, max_exp)
    for b, blk in enumerate(p.demand_blocks):
        ts = np.flatnonzero(blk.mask)
        if len(ts):
            R.add([np.repeat(PK[:, [b]], len(ts), axis=1), IMP[:, ts]], [1, -1], 0.0, INF)
    na = p.n_nonanticipative
    if S > 1 and na > 0:
        for X in (CH, DIS, UB):
            R.add([X[1:, :na], np.repeat(X[[0], :na], S - 1, axis=0)], [1, -1], 0.0, 0.0)

    # per-scenario cost: coefficient vector over that scenario's variables + constant
    def scen_terms(s):
        cols = np.r_[IMP[s], EXP[s], DIS[s], SOC[s, T], PK[s]]
        vals = np.r_[dt * p.price_import, -dt * p.price_export, dt * bat.deg_cost_per_kwh,
                     -p.terminal_value_per_kwh, rates]
        return cols, vals, -float((rates * psf).sum())

    w = np.asarray(p.weights, float)
    cost = np.zeros(nv)
    const = 0.0
    terms = [scen_terms(s) for s in range(S)]
    scale = 1.0
    if worst:
        z = nv - 1
        lo[z], hi[z] = -INF, INF
        cost[z] = 1.0
        scale = float(p.worst_case_tiebreak)
        for s, (cols, vals, c0) in enumerate(terms):
            R.add_row(np.r_[z, cols], np.r_[1.0, -vals], c0, INF)
    for s, (cols, vals, c0) in enumerate(terms):
        np.add.at(cost, cols, scale * w[s] * vals)
        const += scale * w[s] * c0

    rows = np.concatenate(R.r)
    A = sparse.csc_matrix((np.concatenate(R.v), (rows, np.concatenate(R.c))), shape=(R.n, nv))
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = nv, R.n
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = cost, lo, hi
    lp.row_lower_, lp.row_upper_ = np.concatenate(R.lo), np.concatenate(R.hi)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"highspy LP failed: {h.modelStatusToString(status)}")
    x = np.asarray(h.getSolution().col_value)

    out = {"ch": x[CH], "dis": x[DIS], "soc": x[SOC], "imp": x[IMP], "exp": x[EXP],
           "curt": x[CURT], "ub": x[UB], "ug": x[UG]}
    out["p_bat"] = out["dis"] - out["ch"]
    out["cost"] = np.array([vals @ x[cols] + c0 for cols, vals, c0 in terms])
    out["objective"] = float(cost @ x + const)
    if B:
        out["peak"] = x[PK]
    out.update({"solver": "highspy-direct", "termination": "optimal", "wall_time_s": wall,
                "n_vars": nv, "n_cons": R.n})
    return out
