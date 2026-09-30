"""Load and align the public input data.

Time conventions (verified in data/README and LEARNING_GUIDE):

* ComStock AMY2018 individual-building time series: `timestamp` marks the END of a
  15-minute interval, in fixed Eastern Standard Time (UTC-5, no DST), even for
  California buildings. Evidence: the evening exterior-lighting switch-on lags the
  astronomically computed Pacific-standard sunset by 2.7-2.9 h in every month of the
  year (no DST jump), i.e. ~3 h minus a lighting lead of ~15 min.
* ComStock weather files: hour-ENDING stamps in local standard time (PST, UTC-8).
* PVGIS: hourly stamps "YYYYMMDD:HH30" in UTC; each value represents hour HH.
* Output: every series is indexed by the START of its interval, tz-aware in
  America/Los_Angeles (clock time incl. DST), because PG&E's TOU periods are defined
  in local clock time.

Aggregation is done on the UTC axis (PST/PDT offsets are whole hours, so hourly bins
coincide) to avoid ambiguous wall-clock hours on DST change days.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Los_Angeles"
COMSTOCK_TZ = "Etc/GMT+5"      # fixed UTC-5 (POSIX sign convention is inverted)
WEATHER_TZ = "Etc/GMT+8"       # fixed UTC-8
ENERGY_COL = "out.electricity.total.energy_consumption"   # kWh per 15 min

WEATHER_COLS = {
    "Dry Bulb Temperature [°C]": "temp_air",
    "Wind Speed [m/s]": "wind_speed",
    "Global Horizontal Radiation [W/m2]": "ghi",
    "Direct Normal Radiation [W/m2]": "dni",
    "Diffuse Horizontal Radiation [W/m2]": "dhi",
}


# ------------------------------------------------------------------ ComStock
def load_comstock_kw(path: str | Path, source_tz: str = COMSTOCK_TZ) -> pd.Series:
    """15-min average power (kW), indexed by interval START, tz-aware UTC."""
    df = pd.read_csv(path)
    if ENERGY_COL not in df.columns:
        raise ValueError(f"{path}: missing column {ENERGY_COL}")
    end = pd.to_datetime(df["timestamp"]).dt.tz_localize(source_tz)
    start = end - pd.Timedelta(minutes=15)
    kw = df[ENERGY_COL].to_numpy(dtype=float) / 0.25
    s = pd.Series(kw, index=pd.DatetimeIndex(start).tz_convert("UTC"), name="load_kw")
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps not strictly increasing")
    return s


def to_hourly(s: pd.Series) -> pd.Series:
    """Hourly mean of a regular sub-hourly series (keeps only complete hours)."""
    if s.index.tz is None:
        raise ValueError("Expected a tz-aware index.")
    utc = s.tz_convert("UTC")
    counts = utc.resample("1h").count()
    per_hour = int(round(pd.Timedelta("1h") / (utc.index[1] - utc.index[0])))
    mean = utc.resample("1h").mean()
    return mean[counts == per_hour]


# ------------------------------------------------------------------ weather
def load_weather(path: str | Path, source_tz: str = WEATHER_TZ) -> pd.DataFrame:
    """Hourly weather indexed by hour START (tz-aware UTC)."""
    df = pd.read_csv(path)
    end = pd.to_datetime(df["date_time"]).dt.tz_localize(source_tz)
    start = pd.DatetimeIndex(end - pd.Timedelta(hours=1)).tz_convert("UTC")
    out = df.rename(columns=WEATHER_COLS)[list(WEATHER_COLS.values())]
    out.index = start
    for c in ("ghi", "dni", "dhi"):
        out[c] = out[c].clip(lower=0)
    return out


def pv_from_weather(
    wx: pd.DataFrame,
    lat: float,
    lon: float,
    tilt: float = 10.0,
    azimuth: float = 180.0,       # pvlib convention: 180 = south
    system_loss: float = 0.14,    # same lumped loss as the PVGIS request
    gamma_pdc: float = -0.0037,   # 1/K, typical c-Si
    dc_ac_ratio: float = 1.2,
) -> pd.Series:
    """AC power per kWp (DC) from hourly irradiance, using pvlib.

    Chain: solar position at mid-hour -> Hay-Davies transposition -> incidence-angle
    (Fresnel/physical IAM) loss on the beam component, as in PVWatts -> Faiman cell
    temperature -> PVWatts DC model -> lumped system losses -> AC clipping at
    1/dc_ac_ratio. Deliberately simple and transparent; it is validated against PVGIS.
    """
    import pvlib  # local import: only this function needs it

    mid = wx.index + pd.Timedelta(minutes=30)
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
    dni_extra = pvlib.irradiance.get_extra_radiation(mid)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azimuth,
        solar_zenith=sp["apparent_zenith"].to_numpy(), solar_azimuth=sp["azimuth"].to_numpy(),
        dni=wx["dni"].to_numpy(), ghi=wx["ghi"].to_numpy(), dhi=wx["dhi"].to_numpy(),
        dni_extra=dni_extra.to_numpy(), model="haydavies",
    )
    aoi = pvlib.irradiance.aoi(tilt, azimuth, sp["apparent_zenith"].to_numpy(),
                               sp["azimuth"].to_numpy())
    iam = np.nan_to_num(np.asarray(pvlib.iam.physical(aoi), dtype=float), nan=0.0)
    beam = np.nan_to_num(np.asarray(poa["poa_direct"], dtype=float), nan=0.0)
    diffuse = np.nan_to_num(np.asarray(poa["poa_diffuse"], dtype=float), nan=0.0)
    poa_global = np.clip(beam + diffuse, 0, None)
    poa_eff = np.clip(beam * iam + diffuse, 0, None)
    t_cell = pvlib.temperature.faiman(poa_global, wx["temp_air"].to_numpy(),
                                      wx["wind_speed"].to_numpy())
    pdc = pvlib.pvsystem.pvwatts_dc(poa_eff, t_cell, pdc0=1.0, gamma_pdc=gamma_pdc)
    pac = np.minimum(np.asarray(pdc) * (1 - system_loss), 1.0 / dc_ac_ratio)
    return pd.Series(np.clip(pac, 0, None), index=wx.index, name="pv_kw_per_kwp")


# -------------------------------------------------------------------- PVGIS
def load_pvgis_kw_per_kwp(path: str | Path) -> pd.Series:
    js = json.loads(Path(path).read_text())
    h = pd.DataFrame(js["outputs"]["hourly"])
    t = pd.to_datetime(h["time"], format="%Y%m%d:%H%M").dt.floor("h").dt.tz_localize("UTC")
    peak = float(js["inputs"]["pv_module"]["peak_power"])
    return pd.Series(h["P"].to_numpy(float) / 1000.0 / peak, index=pd.DatetimeIndex(t),
                     name="pv_pvgis_kw_per_kwp")


# ------------------------------------------------------------------ assembly
def build_site_frame(
    load_csvs: dict,
    weather_csv: str | Path,
    lat: float,
    lon: float,
    pvgis_json: Optional[str | Path] = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """One frame per site: a load column per building + PV per kWp + weather.

    freq="1h": hourly means. freq="15min": native 15-min loads; hourly PV and weather
    are held constant within each hour (energy-preserving; intra-hour PV variability is
    not represented - a documented limitation).
    The index is the intersection of all inputs, converted to America/Los_Angeles.
    """
    if freq == "1h":
        cols = {name: to_hourly(load_comstock_kw(p)) for name, p in load_csvs.items()}
    elif freq == "15min":
        cols = {name: load_comstock_kw(p) for name, p in load_csvs.items()}
    else:
        raise ValueError("freq must be '1h' or '15min'")
    wx = load_weather(weather_csv)
    hourly = wx[["temp_air", "ghi"]].copy()
    hourly["pv_kw_per_kwp"] = pv_from_weather(wx, lat, lon)
    if pvgis_json is not None:
        hourly["pv_pvgis_kw_per_kwp"] = load_pvgis_kw_per_kwp(pvgis_json)
    frame = pd.DataFrame(cols)
    # map every (sub-)hourly stamp to the hour it belongs to
    hour_of = frame.index.floor("h")
    frame = frame.join(hourly.reindex(hour_of).set_axis(frame.index))
    frame = frame.dropna(subset=list(cols) + ["pv_kw_per_kwp"])
    frame.index = frame.index.tz_convert(LOCAL_TZ)
    frame.index.name = "time_local_start"
    return frame


def read_processed(path: str | Path) -> pd.DataFrame:
    """Read a processed site CSV back with a tz-aware local index."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df.pop("time_utc_start"), utc=True).dt.tz_convert(LOCAL_TZ)
    df.index = pd.DatetimeIndex(idx, name="time_local_start")
    return df


def write_processed(frame: pd.DataFrame, path: str | Path) -> None:
    """Write with an unambiguous UTC time column (local times repeat at DST end)."""
    out = frame.copy()
    out.insert(0, "time_utc_start", out.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.round(5).to_csv(path, index=False)
