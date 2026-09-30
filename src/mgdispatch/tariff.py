"""Commercial electricity tariff: time-of-use (TOU) energy rates + monthly demand charges.

A tariff is described declaratively (see experiments/configs/*.yaml):

    seasons:   {summer: [6,7,8,9], winter: [1,2,3,4,5,10,11,12]}
    periods:   per season, an ordered list of rules; the FIRST matching rule wins,
               unmatched hours fall back to `default_period`.
               rule = {name, hours: [start, end), days: all|weekday|weekend}
    energy_rates: $/kWh per (season, period)
    export_rate:  $/kWh credited for exported energy (flat)
    demand_charges: list of {name, rates: {season: $/kW}, periods: [...] or null}
               `periods: null` means the charge applies to the max demand over ALL hours
               of the billing month ("facility-related" / "max demand" charge).

Billing cycle = calendar month. Public holidays are ignored (treated as weekdays);
this is a documented simplification.

TIME ZONE: TOU periods are defined in LOCAL CLOCK TIME (including daylight saving
time). Pass a tz-aware index in the utility's time zone (e.g. America/Los_Angeles);
`index.hour` then gives the clock hour the utility bills by.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class PeriodRule:
    name: str
    hours: List[int]              # [start, end) in local clock hours, 0..24
    days: str = "all"             # all | weekday | weekend
    months: Optional[List[int]] = None   # restrict the rule to these months (None = all)

    def matches(self, hour: np.ndarray, is_weekend: np.ndarray,
                month: Optional[np.ndarray] = None) -> np.ndarray:
        start, end = self.hours
        if start <= end:
            in_hours = (hour >= start) & (hour < end)
        else:                                   # wraps midnight, e.g. [22, 6)
            in_hours = (hour >= start) | (hour < end)
        if self.days == "all":
            in_days = np.ones_like(in_hours)
        elif self.days == "weekday":
            in_days = ~is_weekend
        elif self.days == "weekend":
            in_days = is_weekend
        else:
            raise ValueError(f"Unknown days spec: {self.days}")
        out = in_hours & in_days
        if self.months is not None:
            if month is None:
                raise ValueError("Rule restricted to months needs the month array.")
            out &= np.isin(month, self.months)
        return out


@dataclass
class DemandCharge:
    name: str
    rates: Dict[str, float]               # season -> $/kW
    periods: Optional[List[str]] = None   # None -> all hours


@dataclass
class DemandBlock:
    """One demand-charge component restricted to one billing month.

    The optimisation model creates one peak variable per block.
    """
    name: str                  # e.g. "max_demand"
    month: str                 # billing month key "YYYY-MM"
    rate: float                # $/kW
    mask: np.ndarray           # bool[T]: time steps whose import counts towards this peak


@dataclass
class Tariff:
    seasons: Dict[str, List[int]]
    periods: Dict[str, List[PeriodRule]]
    default_period: str
    energy_rates: Dict[str, Dict[str, float]]
    export_rate: float
    demand_charges: List[DemandCharge] = field(default_factory=list)
    name: str = "tariff"
    source_note: str = ""
    customer_charge_per_day: float = 0.0     # fixed; does not depend on dispatch

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "Tariff":
        periods = {
            season: [PeriodRule(**r) for r in rules]
            for season, rules in d.get("periods", {}).items()
        }
        dcs = [DemandCharge(**dc) for dc in d.get("demand_charges", [])]
        t = cls(
            seasons={k: list(v) for k, v in d["seasons"].items()},
            periods=periods,
            default_period=d["default_period"],
            energy_rates=d["energy_rates"],
            export_rate=float(d.get("export_rate", 0.0)),
            demand_charges=dcs,
            name=d.get("name", "tariff"),
            source_note=d.get("source_note", ""),
            customer_charge_per_day=float(d.get("customer_charge_per_day", 0.0)),
        )
        t.validate()
        return t

    @classmethod
    def from_yaml(cls, path: str) -> "Tariff":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def validate(self) -> None:
        months = sorted(m for ms in self.seasons.values() for m in ms)
        if months != list(range(1, 13)):
            raise ValueError("Seasons must cover each month 1..12 exactly once.")
        for season in self.seasons:
            if season not in self.energy_rates:
                raise ValueError(f"No energy rates for season '{season}'.")
            names = {r.name for r in self.periods.get(season, [])} | {self.default_period}
            missing = names - set(self.energy_rates[season])
            if missing:
                raise ValueError(f"Season '{season}' lacks energy rates for {missing}.")

    # ------------------------------------------------------------ labeling
    def season_of(self, index: pd.DatetimeIndex) -> np.ndarray:
        month_to_season = {m: s for s, ms in self.seasons.items() for m in ms}
        return np.array([month_to_season[m] for m in index.month], dtype=object)

    def period_of(self, index: pd.DatetimeIndex) -> np.ndarray:
        """TOU period label of every time stamp (time stamp = START of the interval)."""
        season = self.season_of(index)
        hour = np.asarray(index.hour)
        month = np.asarray(index.month)
        is_weekend = np.asarray(index.dayofweek >= 5)
        out = np.full(len(index), self.default_period, dtype=object)
        assigned = np.zeros(len(index), dtype=bool)
        for s, rules in self.periods.items():
            in_season = season == s
            for rule in rules:
                hit = in_season & ~assigned & rule.matches(hour, is_weekend, month)
                out[hit] = rule.name
                assigned |= hit
        return out

    # -------------------------------------------------------------- prices
    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        season = self.season_of(index)
        period = self.period_of(index)
        return np.array(
            [self.energy_rates[s][p] for s, p in zip(season, period)], dtype=float
        )

    def export_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(index), self.export_rate, dtype=float)

    def demand_blocks(self, index: pd.DatetimeIndex) -> List[DemandBlock]:
        """Split every demand charge into per-billing-month blocks over `index`.

        Blocks whose mask is empty inside `index` are dropped (they cannot be
        affected by decisions in this window).
        """
        season = self.season_of(index)
        period = self.period_of(index)
        month_key = np.array([f"{y:04d}-{m:02d}" for y, m in zip(index.year, index.month)])
        blocks: List[DemandBlock] = []
        for dc in self.demand_charges:
            in_period = (
                np.ones(len(index), dtype=bool)
                if dc.periods is None
                else np.isin(period, dc.periods)
            )
            for mk in dict.fromkeys(month_key):          # preserves order
                in_month = month_key == mk
                # the season of a month is unique, so take it from any stamp
                s = season[np.argmax(in_month)]
                rate = float(dc.rates.get(s, 0.0))
                mask = in_month & in_period
                if rate > 0 and mask.any():
                    blocks.append(DemandBlock(dc.name, mk, rate, mask))
        return blocks
