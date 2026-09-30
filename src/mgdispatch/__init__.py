"""mgdispatch: battery / PV microgrid economic dispatch with Pyomo MILP models."""
from .params import BatteryParams, GridParams
from .tariff import Tariff, DemandBlock
from .model import DispatchProblem, build_model, extract_solution
from .problem import make_problem
from .solve import SolverConfig, solve, available_solvers
from .billing import compute_bill

__all__ = [
    "BatteryParams", "GridParams", "Tariff", "DemandBlock", "DispatchProblem",
    "build_model", "extract_solution", "make_problem", "SolverConfig", "solve",
    "available_solvers", "compute_bill",
]
__version__ = "0.1.0"
