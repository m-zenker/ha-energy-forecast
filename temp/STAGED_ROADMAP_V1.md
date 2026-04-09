# Staged Development Roadmap: ha-energy-forecast v1.x

This roadmap outlines the implementation of the Accuracy Improvement Proposals, divided into four logical stages. Each stage is designed to be a stable "checkpoint" with measurable accuracy gains.

---

## Stage 1: Foundational Decomposition (The "Passive House" Baseline)

**Goal**: Isolate the predictable household baseline from stochastic appliance "noise."

### Tasks:
1.  **[1.1] Baseline Target Logic**: Update `_aggregate` in `energy_forecast.py` to calculate `passive_baseline = total_consumption - sum(controllable_sub_sensors)`.
2.  **[1.2] Config Flag**: Add a `baseline_mode: true` flag to `apps.yaml` to enable this behavior (default to `false` for backward compatibility).
3.  **[1.3] Baseline Model Training**: Ensure `model.py` trains on the new `passive_baseline` target when the flag is enabled.
4.  **[1.4] Validation**: Verify that the Passive Baseline MAE is significantly lower than the previous Gross Import MAE (due to removed appliance noise).

---

## Stage 2: Intent-Driven Thermal & DHW Modeling

**Goal**: Use "Thermal Pressure" and buffer states to predict HVAC and DHW timing.

### Tasks:
1.  **[2.1] Generic Climate Integration**: Update `ha_data.py` to fetch `current_temperature` and `temperature` (setpoint) from any configured Home Assistant `climate` entities.
2.  **[2.2] Thermal Pressure Feature**: Implement the `Setpoint - Current_Temp` delta in `model.py`'s feature engineering.
3.  **[2.3] DHW Buffer Modeling**: Add the DHW Buffer Temperature (°C) as a feature; implement a non-linear "Trigger Probability" feature based on distance-to-floor.
4.  **[2.4] Verified Passive Decay (Tau)**: Implement a calibration step that observes indoor temp drop rate during verified "off" periods (e.g., using a `heating_system_active_entity` like a Summer Mode switch) to guarantee 100% passive behavior during Tau estimation.

---

## Stage 3: Automated Load Signature Discovery

**Goal**: Automatically learn the energy "shape" of dishwashers, washing machines, etc.

### Tasks:
1.  **[3.1] Cycle Detection Algorithm**: Create a utility to scan `sub_energy_sensors` history for 0 → >0 transitions.
2.  **[3.2] Signature Averaging**: Extract a 4-hour window following each transition and compute the "Mean Load Profile" (kWh per hour) for that appliance.
3.  **[3.3] Profile Persistence**: Save these learned profiles to a new `appliance_signatures.json` file.
4.  **[3.4] CLI/Logging**: Add a log output during retraining that shows the "Learned Signature" for each appliance (e.g., "Dishwasher: 2.1 kWh total, Peak at Hour 2").

---

## Stage 4: Scenario Mode & Energy Manager API

**Goal**: Turn the forecast into a "What-If" engine for the Energy Manager.

### Tasks:
1.  **[4.1] Composite Forecast Engine**: Implement a logic that can overlay multiple Learned Signatures (from Stage 3) onto the Passive Baseline (from Stage 1).
2.  **[4.2] AppDaemon Service (`get_scenario`)**: Create a new service that accepts a schedule (e.g., `{"dishwasher": "14:00"}`) and returns the composite forecast.
3.  **[4.3] Multi-Forecast Entities**: Publish a "Scenario Forecast" sensor that reflects the Energy Manager's current tentative schedule.
4.  **[4.4] Integration Tests**: Verify that the Energy Manager can simulate 3 different schedules and receive 3 distinct, accurate import forecasts.

---

## Success Criteria per Stage

| Stage | Status | Branch | Primary Metric | Target Outcome |
|---|---|---|---|---|
| **Stage 1** | **Completed** | `feat/stage1-passive-baseline` | Baseline MAE | < 0.25 kWh/h (Passive House stability) |
| **Stage 2** | **Completed** | `feat/stage2-thermal-dhw` | HVAC/DHW Residuals | 15%+ reduction in error during heating hours |
| **Stage 3** | Pending | - | Profile Consistency | Standard deviation of cycles < 0.3 kWh |
| **Stage 4** | Pending | - | Integration Health | Energy Manager receives scenario forecasts in < 500ms |
