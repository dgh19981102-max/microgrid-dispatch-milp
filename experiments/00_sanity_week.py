"""Experiment 00 - model sanity check on SYNTHETIC one-week profiles.

Purpose: verify that the MILP behaves sensibly before any real data is used.
Not a reported result. For each seed it solves the week with perfect foresight and
compares against "no battery", and it compares MILP vs LP relaxation and solvers.

Usage:
    python experiments/00_sanity_week.py                 # full (5 seeds)
    python experiments/00_sanity_week.py --quick         # 1 seed, 2 days
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

from mgdispatch import (BatteryParams, GridParams, SolverConfig, Tariff,  # noqa: E402
                        available_solvers, compute_bill, make_problem, solve)
from mgdispatch.solve import SolverLicenseLimitError          # noqa: E402
from mgdispatch.synthetic import synthetic_office_week        # noqa: E402


def no_battery_flows(load, pv, export_limit):
    net = load - pv
    imp = np.clip(net, 0, None)
    exp = np.clip(-net, 0, None)
    if export_limit is not None:
        exp = np.minimum(exp, export_limit)
    return imp, exp


def plot_week(df, sol, tariff, path):
    idx = df.index
    price = tariff.energy_price(idx)
    imp0, _ = no_battery_flows(df.load_kw.values, df.pv_kw.values, None)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1]})
    ax = axes[0]
    ax.plot(idx, df.load_kw, color="#444", lw=1.2, label="Building load")
    ax.fill_between(idx, 0, df.pv_kw, color="#f2b134", alpha=0.35, label="PV available")
    ax.plot(idx, imp0, color="#c0392b", lw=1, ls="--", label="Grid import, no battery")
    ax.plot(idx, sol["imp"][0], color="#154f58", lw=1.6, label="Grid import, optimised")
    ax.set_ylabel("Power (kW)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title("Synthetic office week - perfect-foresight MILP dispatch (sanity check)")

    ax = axes[1]
    ax.bar(idx, sol["p_bat"][0], width=1 / 24, color=np.where(sol["p_bat"][0] >= 0,
                                                              "#2e86ab", "#8e44ad"))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Battery (kW)\n+dis / -ch")
    ax2 = ax.twinx()
    ax2.plot(idx, sol["soc"][0][:-1], color="#e67e22", lw=1.2)
    ax2.set_ylabel("SOC (kWh)", color="#e67e22")

    ax = axes[2]
    ax.step(idx, price, where="post", color="#555")
    ax.set_ylabel("Energy price\n($/kWh)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/00_sanity.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["days"] = 2

    tariff = Tariff.from_yaml(str(ROOT / cfg["tariff"]))
    bat = BatteryParams(**cfg["battery"])
    grid = GridParams(**cfg.get("grid", {}))
    out_dir = ROOT / cfg["out_dir"] / ("quick" if args.quick else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    solvers = [s for s in cfg["solvers"] if s in available_solvers()]
    print(f"Solvers available for comparison: {solvers}")

    records = []
    for seed in cfg["seeds"]:
        df = synthetic_office_week(start=cfg["start"], days=cfg["days"], seed=seed,
                                   **cfg["profile"])
        L, P = df.load_kw.values, df.pv_kw.values
        kw = dict(battery=bat, grid=grid, soc0=cfg["soc0_frac"] * bat.energy_kwh,
                  terminal_soc_min=cfg["soc0_frac"] * bat.energy_kwh)
        # baseline: no battery
        imp0, exp0 = no_battery_flows(L, P, grid.export_limit_kw)
        base = compute_bill(df.index, imp0, exp0, np.zeros_like(L), tariff, dt=1.0)

        rec = {"seed": seed, "no_battery": base, "solvers": {}}
        milp = None
        for name in solvers:
            try:
                sol = solve(make_problem(df.index, L, P, tariff, **kw),
                            SolverConfig(name=name, mip_gap=cfg["mip_gap"]))
            except SolverLicenseLimitError as e:
                rec["solvers"][name] = {"error": "size-limited licence"}
                print(f"  seed {seed} {name}: skipped ({e})")
                continue
            rec["solvers"][name] = {k: sol[k] for k in
                                    ("objective", "wall_time_s", "n_vars", "n_cons")}
            if name == "highs":
                milp = sol
        lp = solve(make_problem(df.index, L, P, tariff, integer=False, **kw),
                   SolverConfig(name="highs"))
        bill = compute_bill(df.index, milp["imp"][0], milp["exp"][0], milp["dis"][0],
                            tariff, dt=1.0, deg_cost_per_kwh=bat.deg_cost_per_kwh)
        both = np.minimum(lp["ch"][0], lp["dis"][0])
        rec.update({
            "milp_bill": bill,
            "lp_objective": lp["objective"],
            "milp_objective": milp["objective"],
            "lp_simultaneous_ch_dis_kwh": float(both.sum()),
            "equivalent_full_cycles": float(milp["dis"][0].sum() / bat.energy_kwh),
            "peak_no_battery_kw": float(imp0.max()),
            "peak_milp_kw": float(milp["imp"][0].max()),
        })
        records.append(rec)
        with open(out_dir / f"seed_{seed}.json", "w") as f:
            json.dump(rec, f, indent=2)
        if seed == cfg["seeds"][0]:
            plot_week(df, milp, tariff, out_dir / "dispatch_week.png")
        sav = base["total"] - bill["total"]
        print(f"seed {seed}: no battery ${base['total']:.0f} -> MILP ${bill['total']:.0f} "
              f"(save ${sav:.0f}, {100 * sav / base['total']:.1f}%); "
              f"peak {imp0.max():.0f} -> {milp['imp'][0].max():.0f} kW; "
              f"LP-MILP gap ${milp['objective'] - lp['objective']:.4f}")

    # summary
    sav = np.array([r["no_battery"]["total"] - r["milp_bill"]["total"] for r in records])
    dem = np.array([r["no_battery"]["demand"] - r["milp_bill"]["demand"] for r in records])
    summary = {
        "n_seeds": len(records),
        "saving_mean": float(sav.mean()), "saving_std": float(sav.std(ddof=1)) if len(sav) > 1 else 0.0,
        "share_of_saving_from_demand_charges": float(dem.sum() / sav.sum()),
        "max_lp_milp_gap": float(max(r["milp_objective"] - r["lp_objective"] for r in records)),
        "solver_objectives_seed0": {k: v.get("objective") for k, v in records[0]["solvers"].items()},
        "solver_times_seed0_s": {k: v.get("wall_time_s") for k, v in records[0]["solvers"].items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
