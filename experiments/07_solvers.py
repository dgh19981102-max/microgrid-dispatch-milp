"""Experiment 07 - solver comparison and scaling.

(a) The same MPC windows (deterministic, 36 steps) solved through Pyomo with HiGHS,
    CBC, Gurobi and CPLEX (whichever are installed; the free size-limited licences are
    enough for these small models) and through the direct highspy path.
(b) Scaling of the stochastic LP with the number of scenarios (direct highspy).

    python experiments/07_solvers.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from mgdispatch.fastlp import solve_fast_lp                          # noqa: E402
from mgdispatch.runner import all_buildings, build, load_config, results_root  # noqa: E402
from mgdispatch.solve import SolverConfig, SolverLicenseLimitError, available_solvers, solve  # noqa: E402
from mgdispatch.simulate import PlantState                           # noqa: E402

exp06 = __import__("06_milp_vs_lp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out = results_root() / "07_solvers" / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    site, b = all_buildings(cfg)[9]
    _, sim, _ = build(cfg, site, b)
    rng = np.random.default_rng(1)
    probs = [p for p, _ in exp06.windows(sim, 5 if args.quick else 30, rng)]
    res = {"building": b, "available": available_solvers(), "solvers": {}, "scaling": []}

    def bench(name, fn):
        objs, times = [], []
        for p in probs:
            t0 = time.perf_counter()
            objs.append(fn(p))
            times.append(time.perf_counter() - t0)
        return objs, times

    ref, _ = bench("direct", lambda p: solve_fast_lp(dataclasses.replace(p, integer=False))["objective"])
    for s in res["available"]:
        for integer in (False, True):
            key = f"pyomo+{s}+{'milp' if integer else 'lp'}"
            try:
                objs, times = bench(key, lambda p: solve(dataclasses.replace(p, integer=integer),
                                                          SolverConfig(name=s, mip_gap=1e-6))["objective"])
            except SolverLicenseLimitError as e:
                res["solvers"][key] = {"error": str(e)}
                continue
            res["solvers"][key] = {"median_ms": 1000 * float(np.median(times)),
                                   "max_abs_obj_diff_vs_direct": float(np.max(np.abs(np.array(objs) - ref)))}
    _, times = bench("direct", lambda p: solve_fast_lp(dataclasses.replace(p, integer=False))["objective"])
    res["solvers"]["direct_highspy_lp"] = {"median_ms": 1000 * float(np.median(times)),
                                           "max_abs_obj_diff_vs_direct": 0.0}
    for k, v in res["solvers"].items():
        print(f"(a) {k:26s} {v}")

    # (b) scaling with scenarios
    base = probs[0]
    err = sim.fc.sampler
    for S in ([1, 5, 10] if args.quick else [1, 5, 10, 20, 40, 80]):
        L = np.repeat(base.load, S, axis=0) * rng.uniform(0.85, 1.15, (S, 1))
        P = np.repeat(base.pv, S, axis=0) * rng.uniform(0.5, 1.1, (S, 1))
        p = dataclasses.replace(base, load=L, pv=P, weights=None, n_nonanticipative=4, integer=False)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            sol = solve_fast_lp(p)
            ts.append(time.perf_counter() - t0)
        res["scaling"].append({"scenarios": S, "n_vars": sol["n_vars"], "median_ms": 1000 * float(np.median(ts))})
        print(f"(b) S={S:3d} vars={sol['n_vars']:6d} {1000 * np.median(ts):8.1f} ms")
    (out / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
