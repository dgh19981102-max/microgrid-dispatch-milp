"""Experiment 01 - build the processed hourly dataset and check its quality.

Outputs
  data/processed/<site>.csv            hourly load per building + PV per kWp + weather
  results/01_data/data_summary.json    per-building statistics and validation numbers
  results/01_data/data_overview.png    figure for the README / showcase page

Usage:
    python experiments/01_prepare_data.py            # full
    python experiments/01_prepare_data.py --quick    # one site only (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

from mgdispatch.checks import lighting_offset_by_month      # noqa: E402
from mgdispatch.data import (build_site_frame, load_comstock_kw,  # noqa: E402
                             write_processed)

COLORS = {"MediumOffice": "#154f58", "RetailStandalone": "#c0392b", "Warehouse": "#7f8c8d"}


def building_files(raw: Path, site: str, btype: str):
    return sorted((raw / "comstock").glob(f"{site}_{btype}_*.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/01_data.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    raw, out = ROOT / cfg["raw_dir"], ROOT / cfg["out_dir"]
    res = ROOT / cfg["results_dir"] / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    sites = dict(list(cfg["sites"].items())[:1]) if args.quick else cfg["sites"]

    summary = {"sites": {}, "buildings": {}}
    frames = {}
    for site, sc in sites.items():
        loads = {f.stem: f for bt in cfg["building_types"] for f in building_files(raw, site, bt)}
        if not loads:
            raise FileNotFoundError(f"No ComStock files for {site} in {raw/'comstock'}")
        wx_file = raw / "weather" / f"{site}_{sc['county']}_2018.csv"
        pvgis = raw / "pvgis" / f"{site}_2018.json"
        frame = build_site_frame(loads, wx_file, sc["lat"], sc["lon"],
                                 pvgis if pvgis.exists() else None)
        write_processed(frame, out / f"{site}_hourly.csv")
        frame15 = build_site_frame(loads, wx_file, sc["lat"], sc["lon"], None, freq="15min")
        write_processed(frame15.drop(columns=["temp_air", "ghi"]), out / f"{site}_15min.csv")
        frames[site] = frame

        # --- PV validation against PVGIS on common hours
        pv_info = {"yield_nsrdb_pvlib_kwh_per_kwp": float(frame.pv_kw_per_kwp.sum())}
        if "pv_pvgis_kw_per_kwp" in frame:
            m = frame.dropna(subset=["pv_pvgis_kw_per_kwp"])
            a, p = m.pv_kw_per_kwp, m.pv_pvgis_kw_per_kwp
            da, dp = a.resample("D").sum(), p.resample("D").sum()
            pv_info.update({
                "common_hours": int(len(m)),
                "yield_nsrdb_pvlib_common": float(a.sum()),
                "yield_pvgis_era5_common": float(p.sum()),
                "relative_difference": float(a.sum() / p.sum() - 1),
                "hourly_correlation": float(np.corrcoef(a, p)[0, 1]),
                "daily_energy_cv_nsrdb": float(da.std() / da.mean()),
                "daily_energy_cv_era5": float(dp.std() / dp.mean()),
            })
        summary["sites"][site] = {"name": sc["name"], "hours": int(len(frame)),
                                  "start_local": str(frame.index[0]),
                                  "end_local": str(frame.index[-1]), "pv": pv_info}

        # --- per-building statistics + time-zone evidence
        for name, path in loads.items():
            kw15 = load_comstock_kw(path)
            naive = pd.Series(kw15.to_numpy(), index=kw15.index.tz_convert("Etc/GMT+5").tz_localize(None))
            offs = lighting_offset_by_month(naive, sc["lat"], sc["lon"])
            h = frame[name]
            summary["buildings"][name] = {
                "site": site, "type": name.split("_")[1],
                "annual_mwh": float(h.sum() / 1000),
                "peak_kw_15min": float(kw15.max()), "peak_kw_hourly": float(h.max()),
                "load_factor": float(h.mean() / h.max()),
                "lighting_minus_pst_sunset_h_by_month": {int(k): round(float(v), 2)
                                                         for k, v in offs.items()},
            }
        print(f"[{site}] {len(frame)} hours, {len(loads)} buildings, "
              f"PV yield {pv_info['yield_nsrdb_pvlib_kwh_per_kwp']:.0f} kWh/kWp")

    with open(res / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------------------------------------------------------- figure
    ns = len(frames)
    fig, axes = plt.subplots(2, ns, figsize=(6.2 * ns, 7.2), squeeze=False)
    for j, (site, frame) in enumerate(frames.items()):
        week = frame.loc["2018-07-16":"2018-07-22"]
        ax = axes[0, j]
        for bt, c in COLORS.items():
            cols = [k for k in frame.columns if f"_{bt}_" in k]
            for i, col in enumerate(cols):
                ax.plot(week.index, week[col] / frame[col].max(), color=c, lw=1.1 if i == 0 else 0.6,
                        alpha=1 if i == 0 else 0.45, label=bt if i == 0 else None)
        ax.fill_between(week.index, 0, week.pv_kw_per_kwp / frame.pv_kw_per_kwp.max(),
                        color="#f2b134", alpha=0.3, label="PV (normalised)")
        ax.set_title(f"{summary['sites'][site]['name']} - week of 16 Jul 2018", fontsize=10)
        ax.set_ylabel("Power / annual peak (-)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)

        ax = axes[1, j]
        for bt, c in COLORS.items():
            cols = [k for k in frame.columns if f"_{bt}_" in k]
            mp = pd.concat([frame[k].resample("MS").max() / frame[k].max() for k in cols], axis=1)
            ax.plot(range(1, 13), mp.mean(axis=1).to_numpy()[:12], "o-", color=c, label=bt)
            ax.fill_between(range(1, 13), mp.min(axis=1).to_numpy()[:12],
                            mp.max(axis=1).to_numpy()[:12], color=c, alpha=0.15)
        ax.axvspan(5.5, 9.5, color="#f2b134", alpha=0.12, label="B-19 summer")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly peak / annual peak (-)")
        ax.set_title("Monthly peak demand (band = 3 buildings)", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res / "data_overview.png", dpi=130)
    print(f"Saved {res/'data_overview.png'}")


if __name__ == "__main__":
    main()
