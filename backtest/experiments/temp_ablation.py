"""How much does outdoor temperature contribute? (2026-11-07, "temperature has low importance").

Every arm uses ``regime_count=3`` to reproduce the original prototype; live uses 0 (auto).
Lean frame: the -T / -regime results do NOT transfer to live, where physics inputs overlap temperature.

-T is not temperature-free: dropping columns leaves temperature inside lag_*_tgated (discounted by temp deltas) and
the regime_kwh temperature gate, so it understates temperature's value. A truly temperature-free arm needs a
weather transform (e.g. constant temp_c), not a drop.
"""

from backtest.arms import Arm, climatology

TEMP_GROUP = (
    "temp_c",
    "heating_degree",
    "cooling_degree",
    "hp_heating_degree",
    "temp_in_neutral_zone",
    "temp_rolling_3d",
    "temp_ewma_24h",
    "temp_ewma_72h",
    "heating_deg_sum_24h",
    "heating_deg_sum_168h",
    "temp_delta_1h",
    "temp_delta_24h",
    "temp_lag_24h",
    "temp_lag_168h",
    "defrost_risk",  # Gaussian of temp_c (model.py), not a sun/wind feature
)
SUN_GROUP = (  # control: non-temperature weather features
    "precipitation_mm",
    "sunshine_min",
    "wind_kmh",
    "humidity",
    "cloud_cover_pct",
    "direct_radiation_wm2",
    "weighted_solar_gain",
    "infiltration_pressure",
    "thermal_pressure_net",
)
_KW = {"regime_count": 3}

ARMS = [
    Arm("base", train_kwargs=_KW),
    Arm("-T", drop=TEMP_GROUP, train_kwargs=_KW),
    Arm("-T-regime", drop=(*TEMP_GROUP, "regime_kwh"), train_kwargs=_KW),
    Arm("-regime", drop=("regime_kwh",), train_kwargs=_KW),
    Arm("clim", weather=climatology, train_kwargs=_KW),
    Arm("-sun", drop=SUN_GROUP, train_kwargs=_KW),
]
