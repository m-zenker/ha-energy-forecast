**Title:** I built an ML energy consumption forecaster to schedule loads around solar surplus

**Category:** Projects & Integration Showcase

---

With solar panels it's not enough to know when the sun will shine — you also need to know when your household will *consume*. I built a Home Assistant app that predicts hourly energy consumption 48 hours ahead, trained entirely on your own HA meter history. No cloud service, no generic averages — it learns your household's patterns (daily routines, heat pump cycles, seasonal swings) and publishes them as standard HA sensors.

My main use case is feeding the forecast into a companion app, [ha-energy-manager](https://github.com/m-zenker/ha-energy-manager), which combines it with a solar production forecast to compute expected surplus and automatically schedule deferrable loads. But the sensors also work directly in HA automations — `energy_forecast_tomorrow` and the 3-hour block sensors are enough for a threshold-based scheduling rule.

A few things that make it more than a basic forecast:
- Scenario API: "what if I run the dishwasher at 14:00?" returns the delta against the baseline
- SHAP explainability: know *why* the model predicted high consumption today
- Rolling 7d/30d MAE sensors to track real-world accuracy over time
- Physics-informed features: thermal pressure, DHW tank pressure, infiltration load, defrost risk
- Works on Raspberry Pi (automatic LightGBM → scikit-learn fallback)

[screenshot: assets/dashboard_overview.png]

GitHub: https://github.com/m-zenker/ha-energy-forecast

I've set up GitHub Discussions if you want to compare accuracy or share how you're using it: [What's your MAE?](https://github.com/m-zenker/ha-energy-forecast/discussions/11) · [How are you using it?](https://github.com/m-zenker/ha-energy-forecast/discussions/12)
