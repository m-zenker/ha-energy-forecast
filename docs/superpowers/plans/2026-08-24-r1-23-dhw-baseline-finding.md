# R1-#23 finding: no dhw_idle_setpoint_c-equivalent baseline drift to fix

hef's `_dhw_kwh_series` baseline branch does not model a static "resting setpoint"
at all — it's a continuous ODE (UA-loss decay + draw-profile depletion) bounded by
`T_dhw_lower`/`T_legionella`, reheating whenever the simulated tank drops below
`T_dhw_lower` (45°C default). EM's `dhw_idle_setpoint_c` (25°C) is a *physical
actuation* concept (the commanded HP setpoint when no boost is wanted) with no
equivalent in hef's forecast model — there is nothing "stale" to correct here.

Confirmed via `grep -rn "dhw_idle_setpoint_c\|idle_setpoint"` across
`ha-energy-forecast` (zero matches) on 2026-08-24.

Decision: no prerequisite fix needed. R1-#23 is resolved as "not applicable" —
the spec's concern was based on an assumption (hef modeling a static resting
setpoint analogous to EM's) that doesn't hold in the current physics model.
Proceeding with the override-blind baseline exactly as `_dhw_kwh_series` computes
it today (Task 6 onward), with no independent baseline correction.
