"""Physical parameters of the battery and the grid connection."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BatteryParams:
    """Battery energy storage system (BESS).

    Units: energy in kWh, power in kW, costs in $.
    """
    energy_kwh: float = 200.0          # usable nameplate energy capacity
    power_kw: float = 100.0            # max charge / discharge power
    eta_charge: float = 0.95           # one-way charging efficiency
    eta_discharge: float = 0.95        # one-way discharging efficiency
    soc_min_frac: float = 0.10         # lower SOC bound (fraction of energy_kwh)
    soc_max_frac: float = 1.00         # upper SOC bound (fraction of energy_kwh)
    deg_cost_per_kwh: float = 0.02     # degradation cost per kWh DISCHARGED (assumption)

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_kwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_kwh

    def validate(self) -> None:
        if self.energy_kwh < 0 or self.power_kw < 0:
            raise ValueError("Battery energy and power must be non-negative.")
        if not (0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1):
            raise ValueError("Efficiencies must lie in (0, 1].")
        if not (0 <= self.soc_min_frac <= self.soc_max_frac <= 1):
            raise ValueError("Require 0 <= soc_min_frac <= soc_max_frac <= 1.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GridParams:
    """Limits of the point of common coupling (PCC)."""
    import_limit_kw: Optional[float] = None   # None = no contractual limit
    export_limit_kw: Optional[float] = None   # None = unlimited; 0 = export not allowed

    def to_dict(self) -> dict:
        return asdict(self)
