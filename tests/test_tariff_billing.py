"""Tests for TOU period labelling, demand blocks and the independent bill."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mgdispatch import Tariff, compute_bill

TARIFF_PATH = str(Path(__file__).resolve().parents[1] / "experiments" / "configs"
                  / "tariff_tou_demand_placeholder.yaml")


@pytest.fixture(scope="module")
def tariff():
    return Tariff.from_yaml(TARIFF_PATH)


@pytest.mark.parametrize("stamp, period, price", [
    ("2023-07-12 17:00", "peak", 0.20),            # summer Wednesday
    ("2023-07-12 15:00", "part_peak", 0.15),
    ("2023-07-12 22:00", "part_peak", 0.15),
    ("2023-07-12 03:00", "off_peak", 0.12),
    ("2023-07-15 17:00", "off_peak", 0.12),        # summer Saturday: no peak
    ("2023-01-11 17:00", "peak", 0.17),            # winter Wednesday
    ("2023-01-14 10:00", "super_off_peak", 0.09),  # winter Saturday, midday
    ("2023-01-11 21:00", "off_peak", 0.13),        # end hour is exclusive
])
def test_period_and_price(tariff, stamp, period, price):
    idx = pd.DatetimeIndex([stamp])
    assert tariff.period_of(idx)[0] == period
    assert tariff.energy_price(idx)[0] == pytest.approx(price)


def test_invalid_season_cover_rejected():
    bad = {"seasons": {"a": [1, 2]}, "default_period": "x", "energy_rates": {"a": {"x": 1}}}
    with pytest.raises(ValueError):
        Tariff.from_dict(bad)


def test_bill_by_hand(tariff):
    # Two summer-weekday hours: 16:00 (peak) and 17:00 (peak).
    idx = pd.date_range("2023-07-12 16:00", periods=2, freq="1h")
    imp = np.array([100.0, 50.0])
    exp = np.array([0.0, 10.0])
    dis = np.array([5.0, 0.0])
    bill = compute_bill(idx, imp, exp, dis, tariff, dt=1.0, deg_cost_per_kwh=0.02)
    assert bill["energy_import"] == pytest.approx(150 * 0.20)
    assert bill["export_credit"] == pytest.approx(10 * 0.05)
    assert bill["degradation"] == pytest.approx(5 * 0.02)
    # max demand 100 kW * 30 + peak-period demand 100 kW * 20 (summer)
    assert bill["demand"] == pytest.approx(100 * 30 + 100 * 20)
    assert bill["total"] == pytest.approx(30 - 0.5 + 0.1 + 5000)


def test_demand_charged_per_month(tariff):
    idx = pd.date_range("2023-06-30 12:00", periods=24, freq="1h")  # spans June/July
    imp = np.where(idx.month == 6, 80.0, 40.0)
    bill = compute_bill(idx, imp, np.zeros(24), np.zeros(24), tariff, dt=1.0)
    assert bill["demand:max_demand:2023-06"] == pytest.approx(80 * 30)
    assert bill["demand:max_demand:2023-07"] == pytest.approx(40 * 30)
