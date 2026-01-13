from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .schema import ShockConfig


@dataclass(frozen=True)
class ShockSignals:
    export_restriction: float = 0.0  # fraction of supply blocked
    demand_surge: float = 0.0        # fractional demand boost
    capex_shock: float = 0.0         # fractional reduction in investment target
    stockpile_release: float = 0.0   # one-time inventory delta (tons)


def _active(shock: ShockConfig, year: int) -> bool:
    return shock.start_year <= year <= shock.end_year


def shocks_for_year(shocks: List[ShockConfig], year: int) -> ShockSignals:
    export_restriction = 0.0
    demand_surge = 0.0
    capex_shock = 0.0
    stockpile_release = 0.0

    for s in shocks:
        if not _active(s, year):
            continue
        if s.type == "export_restriction":
            export_restriction += s.magnitude
        elif s.type == "demand_surge":
            demand_surge += s.magnitude
        elif s.type == "capex_shock":
            capex_shock += s.magnitude
        elif s.type == "stockpile_release":
            stockpile_release += s.magnitude

    # clamp to sane ranges
    export_restriction = min(max(export_restriction, 0.0), 0.95)
    capex_shock = min(max(capex_shock, 0.0), 0.95)
    # demand_surge can be >1 but keep it reasonable
    demand_surge = max(demand_surge, -0.95)
    # stockpile_release is a one-time delta, can be positive or negative but keep reasonable
    stockpile_release = max(stockpile_release, -1000.0)  # allow negative (stockpile drawdown)

    return ShockSignals(
        export_restriction=export_restriction,
        demand_surge=demand_surge,
        capex_shock=capex_shock,
        stockpile_release=stockpile_release,
    )
