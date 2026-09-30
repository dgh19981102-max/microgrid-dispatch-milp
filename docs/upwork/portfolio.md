# Upwork portfolio text

**Title** (58/70 chars)

Battery Dispatch Optimization: Pyomo MILP & Stochastic MPC

**Description** (453/600 chars)

Cut commercial electricity bills with PV + battery under TOU and demand charges (PG&E B-19). Solver-agnostic Pyomo MILP (HiGHS/CBC/Gurobi/CPLEX), rolling-horizon MPC with deterministic, quantile, stochastic and min-max variants, tested on 18 buildings at 15-min resolution. Stochastic MPC beat deterministic MPC in 18/18 buildings; forecast error, not the optimizer, is the main loss. Includes battery sizing, 66 unit tests and a 10x faster solver loop.

**Images and captions** (max 140 chars each)

- `cover_1800x1150.png` (124): Bill savings of 7 dispatch strategies across 18 commercial buildings on the PG&E B-19 tariff, with 95% confidence intervals.
- `fig_policy_savings.png` (122): Per-building savings: with perfect forecasts MPC reaches 98% of the optimum; real forecasts leave about half on the table.
- `fig_dispatch_example.png` (127): The demand-charge ratchet: open-loop deterministic MPC pushes the monthly peak from 130 to 159 kW; stochastic MPC holds 115 kW.
- `fig_forecast_value.png` (128): What forecast accuracy is worth: savings drop steeply with the first errors, so handling uncertainty beats small forecast gains.
- `fig_sizing.png` (124): Battery sizing sweep: savings grow with size while value per kWh falls; break-even capex from $3,354 down to $1,610 per kWh.

**Skills** (5)

Mathematical Optimization, Pyomo, Energy Modeling, Model Predictive Control, Python

Images: `docs/upwork/cover_1800x1150.png` (cover) and `results/summary/*.png`.