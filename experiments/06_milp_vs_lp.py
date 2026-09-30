"""Experiment 06 - when do the binary variables matter?

(a) Real tariff: count MILP fallbacks recorded in every MPC run of exp 03-05 (each
    run stores how often the LP relaxation violated complementarity).
(b) Stress tests on real MPC windows (fixed before running): the LP relaxation vs the
    MILP under tariffs where wasting energy or round-tripping through the grid pays:
      - B-19 as is                         (expected: identical)
      - negative import price 10:00-15:00  (e.g. oversupply hours in CAISO)
      - export price above import price during the peak period
    For each: objective gap and max simultaneous charge/discharge in the LP.

    python experiments/06_milp_vs_lp.py [--quick]
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
from mgdispatch.model import DispatchProblem                         # noqa: E402
from mgdispatch.runner import all_buildings, build, load_config, results_root  # noqa: E402
from mgdispatch.simulate import PlantState                          # noqa: E402
from mgdispatch.solve import SolverConfig, complementarity_violation, solve  # noqa: E402


def windows(sim, n, rng):
    pos = rng.choice(np.arange(sim.eval_start, sim.eval_end - 4, 4), size=n, replace=False)
    for i0 in pos:
        nidx = sim.dec_pos[int(i0)]
        lay = sim.layout
        steps = lay.step_starts(int(i0))
        blocks, psf = sim.window_blocks(steps, PlantState(soc=sim.soc0))
        L = lay.aggregate(sim.fc.load_fc[nidx, :lay.horizon_quarters])[None]
        P = lay.aggregate(sim.fc.pv_fc[nidx, :lay.horizon_quarters])[None]
        hours = np.array([sim.index[s].hour for s in steps])
        yield DispatchProblem(dt=lay.dt, load=L, pv=P, price_import=sim.price_imp[steps].copy(),
                              price_export=sim.price_exp[steps].copy(), battery=sim.battery,
                              grid=sim.grid, soc0=sim.soc0, demand_blocks=blocks,
                              peak_so_far=psf, terminal_value_per_kwh=0.10), hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--windows", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config()
    out = results_root() / "06_milp_vs_lp" / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    res = {"fallback_counts": {}, "stress": {}}

    # (a) fallbacks recorded in the MPC runs
    tot_s = tot_f = 0
    recs = [json.loads(f.read_text()) for f in results_root().glob("0[345]_*/raw/*/*.json")]
    for f in results_root().glob("0[345]_*/runs.jsonl"):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for r in recs:
        st = r.get("solver_stats", {})
        tot_s += st.get("solves", 0)
        tot_f += st.get("milp_fallbacks", 0)
    res["fallback_counts"] = {"solves": tot_s, "milp_fallbacks": tot_f}
    print(f"(a) MPC solves in exp 03-05: {tot_s}, MILP fallbacks: {tot_f}")

    # (b) stress tests
    rng = np.random.default_rng(0)
    blds = all_buildings(cfg)
    blds = blds[:1] if args.quick else [blds[k] for k in (0, 5, 9, 14)]
    nwin = 5 if args.quick else args.windows
    scen = {
        "b19_as_is": lambda p, h: p,
        "negative_price_10_15h": lambda p, h: dataclasses.replace(
            p, price_import=np.where((h >= 10) & (h < 15), -0.03, p.price_import)),
        "export_above_import_in_peak": lambda p, h: dataclasses.replace(
            p, price_export=np.where((h >= 16) & (h < 21), 0.30, p.price_export)),
    }
    solver = SolverConfig(name="highs", mip_gap=1e-6)
    for name, f in scen.items():
        rows = []
        for site, b in blds:
            _, sim, _ = build(cfg, site, b)
            for prob, hours in windows(sim, nwin, rng):
                p = f(prob, hours)
                t0 = time.perf_counter()
                lp = solve(dataclasses.replace(p, integer=False), solver)
                t1 = time.perf_counter()
                mi = solve(dataclasses.replace(p, integer=True), solver)
                t2 = time.perf_counter()
                rows.append({"gap": mi["objective"] - lp["objective"],
                             "rel_gap": (mi["objective"] - lp["objective"]) / max(abs(mi["objective"]), 1e-9),
                             "lp_violation_kw": complementarity_violation(lp),
                             "lp_s": t1 - t0, "milp_s": t2 - t1})
        g = np.array([r["gap"] for r in rows])
        v = np.array([r["lp_violation_kw"] for r in rows])
        res["stress"][name] = {
            "windows": len(rows), "share_lp_invalid": float((v > 1e-6).mean()),
            "mean_gap_usd": float(g.mean()), "max_gap_usd": float(g.max()),
            "max_lp_violation_kw": float(v.max()),
            "median_lp_s": float(np.median([r["lp_s"] for r in rows])),
            "median_milp_s": float(np.median([r["milp_s"] for r in rows]))}
        print(f"(b) {name:28s} LP invalid in {100 * (v > 1e-6).mean():5.1f}% of windows, "
              f"mean gap ${g.mean():.3f}, max violation {v.max():.1f} kW")
    (out / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
