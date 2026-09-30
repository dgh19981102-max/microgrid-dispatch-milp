"""Upwork cover image (1800 x 1150 px): title bar + core result."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results/summary/all_runs.csv")
core = df[df.exp.isin(["03_policies", "04_uncertainty"]) & df.guard]
base = core[core.policy == "no_battery"].set_index("building")["total"]
per = core[core.policy != "no_battery"].groupby(["policy", "building"])["total"].mean().unstack(0)
sav = 100 * (per.rsub(base, axis=0)).div(base, axis=0)
order = ["mpc_forecast", "mpc_robust", "rule_based", "mpc_quantile", "mpc_stochastic",
         "mpc_perfect", "perfect_foresight_lp"]
names = ["Deterministic MPC", "Robust min-max MPC", "Rule-based", "Quantile MPC",
         "Stochastic MPC", "MPC, perfect forecast", "Perfect foresight"]
rng = np.random.default_rng(0)
ci = []
for p in order:
    x = sav[p].to_numpy()
    b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    ci.append(np.percentile(b, [2.5, 97.5]))
means = [sav[p].mean() for p in order]

fig = plt.figure(figsize=(18, 11.5), dpi=100)
fig.patch.set_facecolor("#f5f7f8")
bar = fig.add_axes([0, 0.80, 1, 0.20])
bar.set_facecolor("#154f58"); bar.set_xticks([]); bar.set_yticks([])
for s in bar.spines.values(): s.set_visible(False)
bar.text(0.035, 0.62, "Battery Dispatch Under Demand Charges", color="white", fontsize=46,
         fontweight="bold", family="serif", va="center")
bar.text(0.035, 0.24, "Pyomo MILP  ·  Stochastic & robust MPC  ·  PG&E B-19  ·  18 buildings, 15-min, 2.3 M solves",
         color="#d6e6ea", fontsize=23, va="center")

ax = fig.add_axes([0.20, 0.10, 0.50, 0.62])
colors = ["#c0392b", "#b07aa1", "#7f8c8d", "#e0a526", "#8e44ad", "#2e86ab", "#154f58"]
y = np.arange(len(order))
ax.barh(y, means, color=colors, height=0.62)
for i, (m, (lo, hi)) in enumerate(zip(means, ci)):
    ax.plot([lo, hi], [i, i], color="black", lw=2)
    ax.text(hi + 0.5, i, f"{m:.1f} %", va="center", fontsize=21, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=21)
ax.set_xlabel("Bill saving vs PV-only (mean, 95 % CI)", fontsize=20)
ax.tick_params(axis="x", labelsize=17)
ax.set_xlim(0, 27)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_facecolor("#f5f7f8")

side = fig.add_axes([0.74, 0.10, 0.24, 0.62]); side.axis("off")
facts = [("97.7 %", "of the bound reached by MPC\nwith perfect forecasts"),
         ("18 / 18", "buildings: stochastic beats\ndeterministic MPC"),
         ("11 / 18", "buildings where open-loop\ndeterministic MPC loses money"),
         ("0", "MILP fallbacks in 2.3 M\nLP-first solves")]
for k, (big, small) in enumerate(facts):
    yy = 0.93 - k * 0.25
    side.text(0.02, yy, big, fontsize=40, fontweight="bold", color="#154f58", family="serif", va="top")
    side.text(0.02, yy - 0.10, small, fontsize=17, color="#44555a", va="top")
out = ROOT / "docs/upwork/cover_1800x1150.png"
fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
print(out)
