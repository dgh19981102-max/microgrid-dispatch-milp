"""Download the raw public data used by this project (run once, e.g. in Google Colab).

What it downloads (all public, no API key needed):
  1. NREL ComStock AMY2018 (2025 release 3) building metadata for two California
     counties, then 15-min electricity time series for a few candidate buildings
     of three building types per county.
     Source: OEDI data lake, https://data.openei.org/ (bucket oedi-data-lake)
  2. The AMY2018 weather file that ComStock used for each county (if published),
     so PV can be modelled from the SAME weather that drove the building loads.
  3. PVGIS hourly PV output for 2018 at each site (1 kWp, fixed rooftop array).
     Source: EU JRC PVGIS API v5.3, https://re.jrc.ec.europa.eu/
     The US radiation database available for 2018 is not known in advance, so
     several are tried in order and the one used is recorded in the manifest.

Only raw data is saved; all cleaning (time zones, resampling, unit conversion)
happens in src/ so that it is unit-tested and reproducible.

Output: ./mg_data_raw/ and ./mg_data_raw.zip (a few MB).

Usage:
    python download_data.py            # full download
    python download_data.py --dry-run  # only list what would be downloaded
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------- config
BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
EULP = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RELEASE = f"{EULP}/2025/comstock_amy2018_release_3/"
OLD_RELEASE = f"{EULP}/2021/comstock_amy2018_release_1/"      # fallback for weather only
META_PREFIX = RELEASE + "metadata_and_annual_results/by_state_and_county/full/parquet/"
TS_PREFIX = RELEASE + "timeseries_individual_buildings/by_state/upgrade=0/"

STATE = "CA"
# Two PG&E-territory counties with contrasting climates.
# County id = GISJOIN code: "G" + state FIPS (06) + "0" + county FIPS + "0".
SITES = {
    "alameda": {"county": "G0600010", "name": "Alameda County (mild, coastal)",
                "lat": 37.80, "lon": -122.27},
    "fresno": {"county": "G0600190", "name": "Fresno County (hot, inland)",
               "lat": 36.74, "lon": -119.79},
}
BUILDING_TYPES = ["MediumOffice", "RetailStandalone", "Warehouse"]
N_CANDIDATES = 3            # buildings per (site, type) closest to the median annual kWh
YEAR = 2018                 # ComStock AMY2018 weather year

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
PVGIS_DATABASES = ["PVGIS-NSRDB", "PVGIS-ERA5", None]      # None = PVGIS default
PVGIS_PARAMS = dict(pvcalculation=1, peakpower=1.0, loss=14, angle=10, aspect=0,
                    mountingplace="building", components=1, outputformat="json")

OUT = Path("mg_data_raw")
TIMEOUT = 120


# ------------------------------------------------------------------ helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def http_get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}: {r.text[:300]}")
            if r.status_code in (400, 403, 404):       # not worth retrying
                break
        except requests.RequestException as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise last


def s3_list(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """Anonymous S3 ListObjectsV2 over HTTPS. Returns (sub-folders, object keys)."""
    folders, keys, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(http_get(BUCKET_URL + "/", params=params).content)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        q = (lambda t: f"s3:{t}") if ns else (lambda t: t)
        folders += [e.text for e in root.findall(f"{q('CommonPrefixes')}/{q('Prefix')}", ns)]
        keys += [e.text for e in root.findall(f"{q('Contents')}/{q('Key')}", ns)]
        trunc = root.find(q("IsTruncated"), ns)
        if trunc is not None and trunc.text == "true":
            token = root.find(q("NextContinuationToken"), ns).text
        else:
            return folders, keys


def pick_candidates(meta: pd.DataFrame, btype: str, n: int) -> pd.DataFrame:
    """Deterministic, documented rule: the n buildings of this type whose annual
    electricity use is closest to the county median for that type."""
    kwh_col = "out.electricity.total.energy_consumption..kwh"
    df = meta[meta["in.comstock_building_type"] == btype].drop_duplicates("bldg_id")
    if df.empty:
        return df
    med = df[kwh_col].median()
    df = df.assign(_dist=(df[kwh_col] - med).abs()).sort_values(["_dist", "bldg_id"])
    return df.head(n).drop(columns="_dist")


# ---------------------------------------------------------------- downloads
def download_metadata(site: str, cfg: dict, manifest: dict) -> pd.DataFrame:
    county = cfg["county"]
    url = f"{BUCKET_URL}/{META_PREFIX}state={STATE}/county={county}/{STATE}_{county}_upgrade0.parquet"
    log(f"[{site}] metadata: {url}")
    meta = pd.read_parquet(io.BytesIO(http_get(url).content))
    keep = [c for c in meta.columns if c in (
        "bldg_id", "in.state", "in.county_name", "in.comstock_building_type",
        "in.sqft..ft2", "in.ashrae_iecc_climate_zone_2006", "weight",
        "out.electricity.total.energy_consumption..kwh",
        "out.electricity.total.peak_demand..kw")]
    manifest["metadata"][site] = {"url": url, "rows": int(len(meta)),
                                  "types": meta["in.comstock_building_type"].value_counts().to_dict()}
    return meta[keep]


def download_timeseries(site: str, row: pd.Series, manifest: dict) -> None:
    bid = int(row["bldg_id"])
    url = f"{BUCKET_URL}/{TS_PREFIX}state={STATE}/{bid}-0.parquet"
    df = pd.read_parquet(io.BytesIO(http_get(url).content))
    time_cols = [c for c in df.columns if "time" in c.lower()]
    elec_cols = [c for c in df.columns if c.startswith("out.electricity.total")
                 and "energy_consumption" in c]
    if not elec_cols:
        raise RuntimeError(f"No total-electricity column in {url}; columns: {list(df.columns)[:40]}")
    if df.index.name and "time" in str(df.index.name).lower():
        df = df.reset_index()
        time_cols = [c for c in df.columns if "time" in c.lower()]
    out = OUT / "comstock" / f"{site}_{row['in.comstock_building_type']}_{bid}.csv"
    df[time_cols + elec_cols].round(4).to_csv(out, index=False)
    manifest["timeseries"].append({
        "site": site, "building_type": row["in.comstock_building_type"], "bldg_id": bid,
        "url": url, "file": str(out.relative_to(OUT)), "rows": int(len(df)),
        "all_columns": list(map(str, df.columns))[:200],
        "time_columns": time_cols, "electricity_columns": elec_cols,
        "first_timestamps": df[time_cols[0]].astype(str).head(3).tolist() if time_cols else [],
    })
    log(f"    saved {out.name} ({len(df)} rows)")


def download_weather(site: str, cfg: dict, manifest: dict) -> None:
    county = cfg["county"]
    for release in (RELEASE, OLD_RELEASE):
        try:
            folders, _ = s3_list(release + "weather/")
        except Exception as e:                                    # noqa: BLE001
            manifest["weather"][site + "_list_error_" + release[-30:]] = str(e)
            continue
        # e.g. weather/amy2018/... ; search (recursively) for this county's csv
        search = [f for f in folders if "2018" in f] or folders or [release + "weather/"]
        for folder in search:
            _, keys = s3_list(folder, delimiter=None)
            keys = [k for k in keys if county in k and k.endswith(".csv")]
            if keys:
                key = keys[0]
                r = http_get(f"{BUCKET_URL}/{key}")
                out = OUT / "weather" / f"{site}_{Path(key).name}"
                out.write_bytes(r.content)
                manifest["weather"][site] = {"url": f"{BUCKET_URL}/{key}", "file": str(out.relative_to(OUT))}
                log(f"[{site}] weather: {key}")
                return
    manifest["weather"][site] = "NOT FOUND"
    log(f"[{site}] weather file not found (not fatal; PVGIS will be used)")


def download_pvgis(site: str, cfg: dict, manifest: dict) -> None:
    errors = {}
    for db in PVGIS_DATABASES:
        params = dict(lat=cfg["lat"], lon=cfg["lon"], startyear=YEAR, endyear=YEAR, **PVGIS_PARAMS)
        if db:
            params["raddatabase"] = db
        try:
            r = http_get(PVGIS_URL, params=params, retries=2)
            js = r.json()
            n = len(js["outputs"]["hourly"])
            if n < 8000:
                raise RuntimeError(f"only {n} hourly rows")
            out = OUT / "pvgis" / f"{site}_{YEAR}.json"
            out.write_text(json.dumps(js))
            manifest["pvgis"][site] = {"database": db or "default", "url": r.url, "rows": n,
                                       "file": str(out.relative_to(OUT))}
            log(f"[{site}] PVGIS ok with {db or 'default'} database ({n} rows)")
            return
        except Exception as e:                                    # noqa: BLE001
            errors[db or "default"] = str(e)[:300]
            log(f"[{site}] PVGIS with {db or 'default'} failed: {str(e)[:120]}")
    manifest["pvgis"][site] = {"error": errors}


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args, _ = ap.parse_known_args()        # tolerate extra args injected by notebooks

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("comstock", "weather", "pvgis"):
        (OUT / sub).mkdir(parents=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "release": RELEASE,
                "sites": SITES, "building_types": BUILDING_TYPES, "metadata": {},
                "candidates": [], "timeseries": [], "weather": {}, "pvgis": {},
                "errors": []}

    for site, cfg in SITES.items():
        log(f"\n===== {site}: {cfg['name']} =====")
        try:
            meta = download_metadata(site, cfg, manifest)
            meta.to_csv(OUT / "comstock" / f"{site}_metadata.csv", index=False)
        except Exception:                                          # noqa: BLE001
            manifest["errors"].append({"step": f"metadata {site}", "trace": traceback.format_exc()})
            log(traceback.format_exc())
            continue
        for btype in BUILDING_TYPES:
            cands = pick_candidates(meta, btype, N_CANDIDATES)
            log(f"[{site}] {btype}: {len(cands)} candidate(s) "
                f"{cands['bldg_id'].tolist()}")
            for _, row in cands.iterrows():
                manifest["candidates"].append({k: (v.item() if hasattr(v, "item") else v)
                                               for k, v in row.items()})
                if args.dry_run:
                    continue
                try:
                    download_timeseries(site, row, manifest)
                except Exception:                                  # noqa: BLE001
                    manifest["errors"].append({"step": f"timeseries {site} {row['bldg_id']}",
                                               "trace": traceback.format_exc()})
                    log(traceback.format_exc())
        if args.dry_run:
            continue
        for fn in (download_weather, download_pvgis):
            try:
                fn(site, cfg, manifest)
            except Exception:                                      # noqa: BLE001
                manifest["errors"].append({"step": f"{fn.__name__} {site}",
                                           "trace": traceback.format_exc()})
                log(traceback.format_exc())

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    archive = shutil.make_archive("mg_data_raw", "zip", root_dir=".", base_dir=str(OUT))

    n_ts = len(manifest["timeseries"])
    n_expected = len(SITES) * len(BUILDING_TYPES) * N_CANDIDATES
    pv_ok = sum("error" not in v for v in manifest["pvgis"].values())
    wx_ok = sum(isinstance(v, dict) for v in manifest["weather"].values())
    log("\n================ SUMMARY ================")
    log(f"building time series : {n_ts} / {n_expected}")
    log(f"weather files        : {wx_ok} / {len(SITES)}")
    log(f"PVGIS series         : {pv_ok} / {len(SITES)}")
    log(f"errors recorded      : {len(manifest['errors'])}")
    log(f"archive              : {archive}")
    ok = (n_ts >= len(SITES) * len(BUILDING_TYPES)) and (pv_ok + wx_ok >= len(SITES))
    log("RESULT: OK" if ok else "RESULT: INCOMPLETE - please send me mg_data_raw.zip anyway")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    if "google.colab" in sys.modules:
        try:
            from google.colab import files
            files.download("mg_data_raw.zip")
        except Exception as e:                                     # noqa: BLE001
            print("Automatic download failed, use the Files panel instead:", e)
    elif code:
        sys.exit(code)
