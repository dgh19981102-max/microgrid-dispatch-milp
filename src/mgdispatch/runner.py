"""Experiment runner: one (building, policy, seed) run -> one JSON file.

Runs are skipped if their JSON exists, and long runs checkpoint their state every
simulated week, so an interrupted session (e.g. a Colab disconnect) resumes where it
stopped. Everything needed to recompute the bill is stored in the JSON.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .billing import compute_bill
from .cases import SizingRule, list_buildings, load_case, make_simulation
from .forecast import forecast_skill
from .simulate import MPC, NoBattery, RuleBased, WindowLayout
from .solve import SolverConfig
from .tariff import Tariff

ROOT = Path(__file__).resolve().parents[2]


def results_root() -> Path:
    """Results (and checkpoints) go here; set MG_RESULTS_DIR to e.g. a Google Drive
    folder so that an interrupted Colab session can resume."""
    return Path(os.environ.get("MG_RESULTS_DIR", ROOT / "results"))


def load_config(path: Optional[str] = None) -> dict:
    path = path or ROOT / "experiments" / "configs" / "base.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_buildings(cfg: dict) -> list:
    pdir = ROOT / cfg["processed_dir"]
    return [(site, b) for site in cfg["sites"] for b in list_buildings(pdir, site)]


def sizing_from(cfg: dict, **override) -> SizingRule:
    sz = dict(cfg["sizing"])
    sz.update(override)
    return SizingRule(**sz, battery_template=dict(cfg["battery_template"]))


def build(cfg: dict, site: str, building: str, quick: bool = False, **size_override):
    """Case + simulation. size_override: fields of SizingRule (e.g. bat_power_fraction)."""
    tariff = Tariff.from_yaml(str(ROOT / cfg["tariff"]))
    layout = WindowLayout(cfg["mpc"]["fine_steps"], cfg["mpc"]["coarse_hours"])
    case = load_case(ROOT / cfg["processed_dir"], site, building, sizing_from(cfg, **size_override),
                     eval_start=cfg["eval_start"], horizon_quarters=layout.horizon_quarters)
    if quick:
        q = cfg["quick"]
        tz = case.index.tz
        case.eval_start = int(np.searchsorted(case.index, pd.Timestamp(q["eval_start"], tz=tz)))
        case.eval_end = int(np.searchsorted(case.index, pd.Timestamp(q["eval_end"], tz=tz)))
    sim = make_simulation(case, tariff, layout, cfg["mpc"]["terminal_value_per_kwh"],
                          pool_days=cfg["mpc"]["pool_days"])
    return case, sim, tariff


def make_policy(spec: dict, sim, cfg: dict):
    kind = spec["policy"]
    if kind == "no_battery":
        return NoBattery()
    if kind == "rule_based":
        return RuleBased(sim)
    if kind.startswith("mpc_"):
        return MPC(sim, mode=kind[4:], n_scen=spec.get("n_scen", cfg["mpc"]["n_scenarios"]),
                   seed=spec.get("seed", 0), error_scale=spec.get("error_scale", 1.0),
                   quantile=cfg["mpc"]["quantile"], solver=SolverConfig(**cfg["solver"]))
    raise ValueError(kind)


def run_label(spec: dict) -> str:
    lab = spec["policy"]
    for k, v in sorted(spec.get("sizing", {}).items()):
        lab += f"_{k}{v:g}"
    if spec.get("error_scale", 1.0) != 1.0:
        lab += f"_k{spec['error_scale']:g}"
    if spec.get("guard") is False:
        lab += "_openloop"
    if spec["policy"] in ("mpc_stochastic", "mpc_robust"):
        lab += f"_seed{spec.get('seed', 0)}"
    return lab


def perfect_foresight_lp(case, sim, tariff):
    """Whole evaluation period solved at once with the actual data (cost lower bound).

    Terminal SOC >= initial SOC so the benchmark cannot 'borrow' stored energy.
    """
    from .fastlp import solve_fast_lp
    from .model import DispatchProblem
    from .solve import complementarity_violation, solve_lp_first
    sl = slice(case.eval_start, case.eval_end)
    idx = case.index[sl]
    prob = DispatchProblem(
        dt=0.25, load=case.load[sl], pv=case.pv[sl],
        price_import=tariff.energy_price(idx), price_export=tariff.export_price(idx),
        battery=case.battery, grid=case.grid, soc0=sim.soc0,
        demand_blocks=tariff.demand_blocks(idx), terminal_soc_min=sim.soc0, integer=False)
    t0 = time.perf_counter()
    sol = solve_fast_lp(prob)
    viol = complementarity_violation(sol)
    if viol > 1e-6:
        raise RuntimeError(f"Perfect-foresight LP violates complementarity ({viol:.2e}); "
                           "the MILP would be needed here.")
    rec = {k: sol[k][0] for k in ("imp", "exp", "ch", "dis", "curt")}
    rec["soc"] = sol["soc"][0][1:]
    return {"rec": rec, "stats": {"lp_vars": sol["n_vars"], "lp_rows": sol["n_cons"],
                                  "lp_violation": viol, "solve_time_s": sol["wall_time_s"]},
            "guard_actions": 0, "wall_time_s": time.perf_counter() - t0}


_PACKED = {}


def packed_result(exp: str, building: str, label: str) -> Optional[dict]:
    """Look a run up in results/<exp>/runs.jsonl (see experiments/pack_results.py)."""
    f = results_root() / exp / "runs.jsonl"
    if not f.exists():
        return None
    key = str(f)
    if key not in _PACKED:
        _PACKED[key] = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                _PACKED[key][(r["building"], r["label"])] = r
    return _PACKED[key].get((building, label))


def run_one(cfg: dict, exp: str, site: str, building: str, spec: dict,
            quick: bool = False, cache: Optional[dict] = None, verbose: bool = True) -> dict:
    out_dir = results_root() / exp / ("quick" if quick else "raw") / building
    out_dir.mkdir(parents=True, exist_ok=True)
    label = run_label(spec)
    out = out_dir / f"{label}.json"
    if out.exists():
        return json.loads(out.read_text())
    if not quick:
        packed = packed_result(exp, building, label)
        if packed is not None:
            return packed
    size_override = spec.get("sizing", {})
    key = (site, building, quick, json.dumps(size_override, sort_keys=True))
    if cache is not None and key in cache:
        case, sim, tariff = cache[key]
    else:
        case, sim, tariff = build(cfg, site, building, quick, **size_override)
        if cache is not None:
            cache.clear()
            cache[key] = (case, sim, tariff)
    ck = out_dir / f"{label}.ckpt.npz"
    t0 = time.perf_counter()
    if spec["policy"] == "perfect_foresight_lp":
        res = perfect_foresight_lp(case, sim, tariff)
    else:
        pol = make_policy(spec, sim, cfg)
        res = sim.run(pol, checkpoint=None if quick else ck, guard=spec.get("guard", cfg["guard"]))
    sl = slice(case.eval_start, case.eval_end)
    rec = res["rec"]
    bill = compute_bill(case.index[sl], rec["imp"], rec["exp"], rec["dis"], tariff, 0.25,
                        case.battery.deg_cost_per_kwh)
    monthly_peak = pd.Series(rec["imp"], index=case.index[sl]).resample("MS").max()
    result = {
        "experiment": exp, "site": site, "building": building, "type": case.btype,
        "label": label, "spec": spec, "quick": quick,
        "eval_start_local": str(case.index[case.eval_start]),
        "eval_end_local": str(case.index[case.eval_end - 1]),
        "pv_kwp": case.pv_kwp, "battery": case.battery.to_dict(),
        "bill": bill,
        "monthly_peak_kw": {k.strftime("%Y-%m"): float(v) for k, v in monthly_peak.items()},
        "energy_discharged_kwh": float(rec["dis"].sum() * 0.25),
        "equivalent_cycles": float(rec["dis"].sum() * 0.25 / case.battery.energy_kwh),
        "pv_curtailed_kwh": float(rec["curt"].sum() * 0.25),
        "guard_actions": int(res.get("guard_actions", 0)),
        "solver_stats": res["stats"], "wall_time_s": time.perf_counter() - t0,
    }
    if spec["policy"] == "mpc_forecast" and spec.get("error_scale", 1.0) == 1.0:
        fc = sim.fc
        in_eval = (fc.decisions >= case.eval_start) & (fc.decisions < case.eval_end)
        result["forecast_skill"] = forecast_skill(fc, in_eval)
    out.write_text(json.dumps(result, indent=1, default=float))
    if ck.exists():
        ck.unlink()
    if verbose:
        print(f"  {building:32s} {label:28s} total ${bill['total']:10.0f}  "
              f"({result['wall_time_s']:.0f}s)", flush=True)
    return result
