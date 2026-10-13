"""
normalize_scores.py
-------------------
Combines heat and precipitation processed CSVs into unified risk scores.
Converts all raw metrics to 0–10 scale using percentile-based normalization.

Risk scores computed:
- heat_score     : based on mean_temp_c + extreme_heat_days + temp_rise_c
- drought_score  : based on dry_months + precip_deficit + low annual precip
- flood_score    : based on heavy_precip_months + precip excess + monsoon intensity
- overall_score  : weighted average (heat 40%, drought 35%, flood 25%)

Output:
    processed/district_risk_scores.csv  — all districts, all periods, all scenarios

Usage:
    python scripts/normalize_scores.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "processed")

SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]
PERIODS   = ["baseline", "2030", "2040", "2050"]

# Overall score weights
WEIGHTS = {"heat": 0.40, "drought": 0.35, "flood": 0.25}


def percentile_scale(series, low_is_bad=False):
    """
    Scale a pandas Series to 0–10 using percentile rank.
    low_is_bad=True  → low values = high risk (e.g. low rainfall = high drought risk)
    low_is_bad=False → high values = high risk (e.g. high temp = high heat risk)
    """
    # Use rank-based percentile to avoid outlier sensitivity
    pct = series.rank(pct=True)  # 0–1
    if low_is_bad:
        pct = 1 - pct
    return np.round(pct * 10, 2)


def compute_heat_score(heat_df):
    """
    Heat score from 3 components:
    - mean_temp_c        (40%) — absolute temperature
    - extreme_heat_days  (40%) — days above 40°C
    - temp_rise_c        (20%) — projected rise vs baseline
    """
    s1 = percentile_scale(heat_df["mean_temp_c"])
    s2 = percentile_scale(heat_df["extreme_heat_days"])
    s3 = percentile_scale(heat_df["temp_rise_c"])
    return np.round(0.4 * s1 + 0.4 * s2 + 0.2 * s3, 2)


def compute_drought_score(precip_df):
    """
    Drought score from 3 components:
    - dry_months          (40%) — more dry months = higher risk
    - mean_annual_precip  (35%) — lower precip = higher risk
    - precip_change_pct   (25%) — larger deficit vs baseline = higher risk
    """
    s1 = percentile_scale(precip_df["dry_months"])
    s2 = percentile_scale(precip_df["mean_annual_precip_mm"], low_is_bad=True)
    # For precip change: more negative = more drought risk
    s3 = percentile_scale(-precip_df["precip_change_pct"])
    return np.round(0.4 * s1 + 0.35 * s2 + 0.25 * s3, 2)


def compute_flood_score(precip_df):
    """
    Flood score from 3 components:
    - heavy_precip_months  (45%) — more heavy rain months = higher flood risk
    - monsoon_precip_mm    (35%) — intense monsoon = higher flood risk
    - precip_change_pct    (20%) — positive change = more rain = more flood risk
    """
    s1 = percentile_scale(precip_df["heavy_precip_months"])
    s2 = percentile_scale(precip_df["monsoon_precip_mm"])
    s3 = percentile_scale(precip_df["precip_change_pct"])
    return np.round(0.45 * s1 + 0.35 * s2 + 0.20 * s3, 2)


def risk_label(score):
    if score >= 8.0:
        return "Very High"
    elif score >= 6.0:
        return "High"
    elif score >= 4.0:
        return "Moderate"
    elif score >= 2.0:
        return "Low"
    else:
        return "Very Low"


def process(scenario, period):
    heat_path   = os.path.join(OUT_DIR, f"heat_{scenario}_{period}.csv")
    precip_path = os.path.join(OUT_DIR, f"precipitation_{scenario}_{period}.csv")

    if not os.path.exists(heat_path) or not os.path.exists(precip_path):
        print(f"[SKIP] Missing files for {scenario} / {period}")
        return None

    heat_df   = pd.read_csv(heat_path)
    precip_df = pd.read_csv(precip_path)

    # Merge on district_id
    df = heat_df[["district_id", "district_name", "state",
                  "mean_temp_c", "max_temp_c", "extreme_heat_days", "temp_rise_c"]].copy()
    df = df.merge(
        precip_df[["district_id", "mean_annual_precip_mm", "dry_months",
                   "precip_change_pct", "monsoon_precip_mm", "heavy_precip_months"]],
        on="district_id", how="left"
    )

    df["scenario"] = scenario
    df["period"]   = period

    # Compute scores
    df["heat_score"]    = compute_heat_score(df)
    df["drought_score"] = compute_drought_score(df)
    df["flood_score"]   = compute_flood_score(df)

    # Overall weighted score
    df["overall_score"] = np.round(
        WEIGHTS["heat"]    * df["heat_score"] +
        WEIGHTS["drought"] * df["drought_score"] +
        WEIGHTS["flood"]   * df["flood_score"],
        2
    )

    # Risk labels
    df["heat_label"]    = df["heat_score"].apply(risk_label)
    df["drought_label"] = df["drought_score"].apply(risk_label)
    df["flood_label"]   = df["flood_score"].apply(risk_label)
    df["overall_label"] = df["overall_score"].apply(risk_label)

    print(f"[DONE] {scenario} / {period} — {len(df)} districts scored")
    return df


def main():
    print("=== ClimateWatch Score Normalizer ===\n")

    all_dfs = []

    for scenario in SCENARIOS:
        for period in PERIODS:
            df = process(scenario, period)
            if df is not None:
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns cleanly
    combined = combined[[
        "district_id", "district_name", "state",
        "scenario", "period",
        "heat_score", "drought_score", "flood_score", "overall_score",
        "heat_label", "drought_label", "flood_label", "overall_label",
        "mean_temp_c", "max_temp_c", "extreme_heat_days", "temp_rise_c",
        "mean_annual_precip_mm", "dry_months", "precip_change_pct",
        "monsoon_precip_mm", "heavy_precip_months",
    ]]

    out_path = os.path.join(OUT_DIR, "district_risk_scores.csv")
    combined.to_csv(out_path, index=False)

    print(f"\n[SAVED] {out_path}")
    print(f"Total rows: {len(combined)} ({len(combined) // (len(SCENARIOS) * len(PERIODS))} districts × {len(SCENARIOS)} scenarios × {len(PERIODS)} periods)")

    # Quick sanity check
    print("\n=== Score Distribution (overall) ===")
    print(combined.groupby("period")["overall_score"].describe().round(2))

    print("\n=== Top 10 Highest Risk Districts (SSP5 2050) ===")
    top = combined[(combined.scenario == "ssp5_8_5") & (combined.period == "2050")]
    top = top.nlargest(10, "overall_score")[["district_name", "state", "overall_score", "overall_label", "heat_score", "drought_score", "flood_score"]]
    print(top.to_string(index=False))

    print("\n=== Normalize complete ===")
    print("Next step: run load_to_neon.py")


if __name__ == "__main__":
    main()