"""Tests for time-zone handling, aggregation and the PV model."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mgdispatch import Tariff
from mgdispatch.data import (LOCAL_TZ, load_comstock_kw, load_weather, pv_from_weather,
                             to_hourly, read_processed, write_processed)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
B19 = ROOT / "experiments" / "configs" / "tariff_pge_b19_secondary_2026.yaml"


def _write_comstock(tmp_path, stamps, kwh):
    p = tmp_path / "b.csv"
    pd.DataFrame({"timestamp": stamps,
                  "out.electricity.total.energy_consumption": kwh}).to_csv(p, index=False)
    return p


def test_comstock_est_interval_end_to_local_start(tmp_path):
    # 12:15 EST marks the END of 12:00-12:15 EST = 09:00 PST = 10:00 PDT (July).
    p = _write_comstock(tmp_path, ["2018-07-01 12:15:00", "2018-07-01 12:30:00"], [5.0, 5.0])
    s = load_comstock_kw(p)
    local = s.index.tz_convert(LOCAL_TZ)
    assert local[0] == pd.Timestamp("2018-07-01 10:00", tz=LOCAL_TZ)
    assert s.iloc[0] == pytest.approx(20.0)          # 5 kWh in 15 min = 20 kW


def test_hourly_mean_preserves_energy_and_drops_partial_hours(tmp_path):
    stamps = pd.date_range("2018-01-01 00:15", periods=10, freq="15min").strftime("%Y-%m-%d %H:%M:%S")
    kwh = np.arange(1, 11, dtype=float)
    s = load_comstock_kw(_write_comstock(tmp_path, stamps, kwh))
    h = to_hourly(s)
    assert len(h) == 2                                # 10 quarters -> 2 complete hours
    assert h.iloc[0] == pytest.approx(kwh[:4].sum())  # kWh in 1 h == mean kW


def test_weather_hour_ending_to_start():
    if not RAW.exists():
        pytest.skip("raw data not present")
    wx = load_weather(RAW / "weather" / "fresno_G0600190_2018.csv")
    # first stamp 2018-01-01 01:00 PST (hour ending) -> starts 00:00 PST = 08:00 UTC
    assert wx.index[0] == pd.Timestamp("2018-01-01 08:00", tz="UTC")


def test_pv_model_physics():
    idx = pd.date_range("2018-06-21 00:00", periods=24, freq="1h", tz="Etc/GMT+8").tz_convert("UTC")
    hour = np.arange(24)
    ghi = np.clip(900 * np.sin(np.pi * (hour - 5) / 14), 0, None)
    wx = pd.DataFrame({"ghi": ghi, "dni": ghi * 0.9, "dhi": ghi * 0.15,
                       "temp_air": 25.0, "wind_speed": 2.0}, index=idx)
    pv = pv_from_weather(wx, 36.74, -119.79)
    assert pv.iloc[:4].max() == 0 and pv.iloc[21:].max() == 0      # night
    assert 0.5 < pv.max() <= 1 / 1.2 + 1e-9                         # clipped at AC limit
    hotter = pv_from_weather(wx.assign(temp_air=40.0), 36.74, -119.79)
    assert hotter.sum() < pv.sum()                                   # temperature derate


def test_processed_roundtrip_keeps_dst(tmp_path):
    idx = pd.date_range("2018-11-04 00:00", periods=5, freq="1h", tz=LOCAL_TZ)  # fall-back day
    f = pd.DataFrame({"x": np.arange(5.0)}, index=idx)
    write_processed(f, tmp_path / "p.csv")
    back = read_processed(tmp_path / "p.csv")
    assert back.index.equals(idx) and back.x.tolist() == f.x.tolist()


# ---------------------------------------------------------------- B-19 tariff
@pytest.fixture(scope="module")
def b19():
    return Tariff.from_yaml(str(B19))


@pytest.mark.parametrize("local, period, price", [
    ("2018-07-02 16:30", "peak", 0.18648),        # summer Monday, PDT clock time
    ("2018-07-07 17:00", "peak", 0.18648),        # summer SATURDAY: peak every day
    ("2018-07-02 15:00", "part_peak", 0.14775),
    ("2018-07-02 22:00", "part_peak", 0.14775),
    ("2018-07-02 23:00", "off_peak", 0.12037),
    ("2018-04-10 10:00", "super_off_peak", 0.06442),   # spring midday
    ("2018-01-10 10:00", "off_peak", 0.12026),         # January midday: NOT super-off-peak
    ("2018-12-01 18:00", "peak", 0.16188),
])
def test_b19_periods_in_local_clock_time(b19, local, period, price):
    idx = pd.DatetimeIndex([pd.Timestamp(local, tz=LOCAL_TZ)])
    assert b19.period_of(idx)[0] == period
    assert b19.energy_price(idx)[0] == pytest.approx(price)


def test_b19_demand_blocks_summer_month(b19):
    idx = pd.date_range("2018-07-01", periods=31 * 24, freq="1h", tz=LOCAL_TZ)
    blocks = {b.name: b for b in b19.demand_blocks(idx)}
    assert blocks["max_demand"].rate == pytest.approx(37.37)
    assert blocks["peak_demand"].rate == pytest.approx(46.16)
    assert blocks["part_peak_demand"].rate == pytest.approx(10.52)
    assert blocks["peak_demand"].mask.sum() == 31 * 5          # 16-21 h every day
    assert blocks["part_peak_demand"].mask.sum() == 31 * 4
    winter = b19.demand_blocks(pd.date_range("2018-01-01", periods=48, freq="1h", tz=LOCAL_TZ))
    assert "part_peak_demand" not in {b.name for b in winter}
