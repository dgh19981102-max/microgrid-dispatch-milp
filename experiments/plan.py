"""The complete run plan (experiments 02-05), in priority order.

Fixed before any full run. Each entry is (experiment, spec). The plan is executed
building by building within each PASS, so that if the time budget runs out the most
important results exist for ALL buildings first.

PASS 1  core comparison (exp 03) + sizing sweep (exp 02, cheap)
PASS 2  stochastic / robust MPC, seed 0 (exp 04)
PASS 3  forecast-error scaling (exp 05)
PASS 4  scenario-sampling seeds 1 and 2 (exp 04)
PASS 5  open-loop ablation without the real-time demand guard (exp 04)
"""
SIZES = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]     # battery power / annual peak, 2 h
ERROR_SCALES = [0.0, 0.5, 1.5, 2.0]              # k = 1 is the plain forecast (exp 03)

PASSES = [
    [("03_policies", {"policy": p}) for p in
     ("no_battery", "perfect_foresight_lp", "rule_based", "mpc_perfect", "mpc_forecast",
      "mpc_quantile")]
    + [("02_sizing", {"policy": "perfect_foresight_lp", "sizing": {"bat_power_fraction": f}})
       for f in SIZES],
    [("04_uncertainty", {"policy": p, "seed": 0}) for p in ("mpc_stochastic", "mpc_robust")],
    [("05_forecast_value", {"policy": "mpc_forecast", "error_scale": k}) for k in ERROR_SCALES],
    [("04_uncertainty", {"policy": p, "seed": s}) for s in (1, 2)
     for p in ("mpc_stochastic", "mpc_robust")],
    [("04_uncertainty", {"policy": p, "guard": False}) for p in
     ("rule_based", "mpc_forecast", "mpc_quantile")]
    + [("04_uncertainty", {"policy": "mpc_stochastic", "seed": 0, "guard": False})],
]
