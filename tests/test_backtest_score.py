"""Tests for backtest/score.py — week-block paired bootstrap, segment tables, summary text."""

import numpy as np
import pandas as pd
import pytest

from backtest.score import format_summary, paired_bootstrap, summarize

FOLD_STARTS = {"f1": "2026-01-05", "f2": "2026-03-02"}  # both Mondays → 3 full ISO weeks each


def _frame(x_mae=lambda base, day, fold: base + 0.01, cold=lambda day: True, x_bias=None, extra_arms=()):
    rng = np.random.default_rng(3)
    rows = []
    for fold, start in FOLD_STARTS.items():
        for day in pd.date_range(start, periods=21, freq="D"):
            for origin in (6, 18):
                base = 0.5 + rng.uniform(-0.1, 0.1)
                common = dict(fold=fold, day=day.date(), origin=origin, cold=cold(day), ts0=day)
                rows.append(dict(arm="base", mae=base, bias=1.0, **common))
                rows.append(
                    dict(arm="x", mae=x_mae(base, day, fold), bias=x_bias if x_bias is not None else 1.0, **common)
                )
                for name, shift in extra_arms:
                    rows.append(dict(arm=name, mae=base + shift, bias=1.0, **common))
    return pd.DataFrame(rows)


DF = _frame()


def _width(r):
    return r["hi"] - r["lo"]


def _day_level_width(df, arm, n_boot=4000, seed=0):
    b = df[df.arm == "base"].set_index(["fold", "day", "origin"])["mae"]
    x = df[df.arm == arm].set_index(["fold", "day", "origin"])["mae"]
    d = (x - b).dropna().groupby(level=["fold", "day"]).mean().to_numpy()
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return np.quantile(bs, 0.975) - np.quantile(bs, 0.025)


def test_constant_shift_gives_exact_delta():
    r = paired_bootstrap(DF, "x")
    assert r["d"] == pytest.approx(0.01) and r["lo"] == pytest.approx(0.01) and r["hi"] == pytest.approx(0.01)
    assert r["worse_days"] == 42 and r["n_days"] == 42 and r["fold_sign"] == "2/2"
    assert r["pct"] == pytest.approx(100 * 0.01 / DF[DF.arm == "base"].mae.mean())


def test_blocks_not_days_are_resampled():
    alt = _frame(x_mae=lambda base, day, fold: base + (0.05 if day.isocalendar().week % 2 else -0.03))
    assert _width(paired_bootstrap(alt, "x")) > 1.5 * _day_level_width(alt, "x")


def test_seeded_ci_is_deterministic():
    noisy = _frame(x_mae=lambda base, day, fold: base + 0.01 + 0.02 * np.sin(day.dayofyear))
    assert paired_bootstrap(noisy, "x") == paired_bootstrap(noisy, "x")


def test_abs_bias_column():
    r = paired_bootstrap(_frame(x_bias=-2.0), "x", col="abs_bias")
    assert r["d"] == pytest.approx(1.0)


def test_missing_ref_raises():
    with pytest.raises(ValueError, match="base"):
        summarize(DF[DF.arm == "x"])


def test_fewer_than_three_blocks_gives_nan_ci():
    one_week = DF[(DF.fold == "f1") & (pd.to_datetime(DF.day) < pd.Timestamp("2026-01-12"))]
    r = paired_bootstrap(one_week, "x")
    assert np.isnan(r["lo"]) and np.isnan(r["hi"]) and r["d"] == pytest.approx(0.01)


def test_cold_warm_partition():
    t = summarize(_frame(cold=lambda day: day.day % 2 == 0))
    assert t["COLD"].loc["base", "n_days"] + t["WARM"].loc["base", "n_days"] == t["ALL"].loc["base", "n_days"] == 42
    assert np.isnan(t["ALL"].loc["base", "pct"]) and t["ALL"].loc["x", "pct"] > 0


def test_format_has_caveats_noise_floor_and_empty_segment():
    df = _frame(extra_arms=[("base@s1", 0.02), ("y", 0.005)])
    s = format_summary(summarize(df), header=["custom header line"])
    assert "archive (perfect)" in s and "frame: lean" in s and "custom header line" in s
    assert "seed noise floor" in s and "WARM: no scored days" in s
    y_line = next(line for line in s.splitlines() if line[2:].startswith("y "))  # col 0 = mark, col 2 = arm
    assert "~" in y_line  # y's +1 % is inside the base@s1 floor (~4 %)


def test_noise_floor_is_per_segment_and_ci_caveat():
    df = _frame(cold=lambda day: day.day % 2 == 0, extra_arms=[("base@s1", 0.0), ("y", 0.02)])
    warm_seed = (df.arm == "base@s1") & ~df.cold
    df.loc[warm_seed, "mae"] += 0.03  # seed spread ~6 % in WARM, 0 % in COLD
    s = format_summary(summarize(df))
    assert "training randomness" in s
    sections = dict(
        zip(
            ["ALL", "COLD", "WARM"],
            s.split("\nALL")[1].replace("\nCOLD", "\x00").replace("\nWARM", "\x00").split("\x00"),
        )
    )
    y_line = {seg: next(ln for ln in txt.splitlines() if ln[2:].startswith("y ")) for seg, txt in sections.items()}
    assert "~" in y_line["WARM"] and "~" not in y_line["COLD"] and "~" not in y_line["ALL"]
    assert all("seed noise floor" in txt for txt in sections.values())
