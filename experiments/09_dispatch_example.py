"""Experiment 09 - illustrate the demand-charge 'ratchet' on one building and week.

Re-simulates July 2018 (from July 1, so month-to-date peaks are right) for one medium
office and plots the grid import of four policies in the week containing the month's
no-battery peak. Cheap (~1 min); illustration only, not a statistic.

    python experiments/09_dispatch_example.py [--building fresno_MediumOffice_75080]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from mgdispatch.runner import build, load_config, results_root       # noqa: E402
from mgdispatch.simulate import MPC, NoBattery, RuleBased            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", default="fresno_MediumOffice_75080")
    args = ap.parse_args()
    cfg = load_config()
    site = args.building.split("_")[0]
    case, sim, _ = build(cfg, site, args.building)
    tz = case.index.tz
    s = int(np.searchsorted(case.index, pd.Timestamp("2018-07-01", tz=tz)))
    e = int(np.searchsorted(case.index, pd.Timestamp("2018-08-01", tz=tz)))
    sim.eval_start, sim.eval_end = s, e
    idx = case.index[s:e]
    runs = {}
    for label, pol, guard in [("No battery", NoBattery(), True),
                              ("Deterministic MPC, open-loop", MPC(sim, "forecast"), False),
                              ("Deterministic MPC + guard", MPC(sim, "forecast"), True),
                              ("Stochastic MPC + guard", MPC(sim, "stochastic", n_scen=10), True),
                              ("MPC, perfect forecast", MPC(sim, "perfect"), True)]:
        runs[label] = pd.Series(sim.run(pol, guard=guard)["rec"]["imp"], index=idx)
        print(f"{label:30s} July peak {runs[label].max():6.1f} kW", flush=True)
    t_peak = runs["No battery"].idxmax()
    w0 = t_peak.normalize() - pd.Timedelta(days=3)
    w = slice(w0, w0 + pd.Timedelta(days=5))
    colors = ["#999999", "#c0392b", "#e0a526", "#8e44ad", "#154f58"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for (label, ser), c in zip(runs.items(), colors):
        ax.plot(ser.loc[w].index, ser.loc[w], color=c, lw=1.3 if label != "No battery" else 2,
                label=f"{label} (July peak {ser.max():.0f} kW)")
        ax.axhline(ser.max(), color=c, lw=0.8, ls="--")
    ax.set_ylabel("Grid import (kW, 15-min)")
    ax.set_title(f"{args.building}: grid import around the July peak "
                 "(dashed = month's billed peak)")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    out = results_root() / "summary"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_dispatch_example.png", dpi=150)
    print("saved", out / "fig_dispatch_example.png")


if __name__ == "__main__":
    main()
