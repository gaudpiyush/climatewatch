"""
download_cmip6.py
-----------------
Downloads CMIP6 climate projection data for India from Copernicus CDS.
Downloads temperature and precipitation for SSP2 and SSP5 scenarios.

Output: raw/ directory with zip files containing NetCDF
Runtime: can take 30–60 mins depending on CDS queue. Run once, never again.

Usage:
    python scripts/download_cmip6.py
"""

import cdsapi
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

client = cdsapi.Client()


def download(variable, scenario, period_label, years):
    filename = f"{variable}_{scenario}_{period_label}.zip"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        print(f"[SKIP] Already exists: {filename}")
        return filepath

    print(f"[DOWNLOAD] {filename} ...")

    # Build month list — all 12 months
    months = [f"{m:02d}" for m in range(1, 13)]

    client.retrieve(
        "projections-cmip6",
        {
            "format": "zip",
            "temporal_resolution": "monthly",
            "experiment": scenario,
            "variable": variable,
            "model": "access_cm2",
            "year": years,
            "month": months,
            "area": [37.5, 67.0, 6.5, 97.5],  # India: [N, W, S, E]
        },
        filepath
    )

    print(f"[DONE] Saved to {filepath}")
    return filepath


def main():
    print("=== ClimateWatch CMIP6 Downloader ===")
    print(f"Saving raw data to: {os.path.abspath(RAW_DIR)}\n")

    # Variables to download
    variables = [
        "near_surface_air_temperature",
        "precipitation",
    ]

    # Scenarios
    scenarios = ["ssp2_4_5", "ssp5_8_5"]

    # Time slices — smaller year ranges to avoid timeout
    periods = {
        "baseline": [str(y) for y in range(2015, 2021)],  # 2015–2020
        "2030":     [str(y) for y in range(2028, 2034)],  # centered on 2030
        "2040":     [str(y) for y in range(2038, 2044)],  # centered on 2040
        "2050":     [str(y) for y in range(2048, 2054)],  # centered on 2050
    }

    total = len(variables) * len(scenarios) * len(periods)
    count = 0

    for variable in variables:
        for scenario in scenarios:
            for period_label, years in periods.items():
                count += 1
                print(f"[{count}/{total}]", end=" ")
                try:
                    download(variable, scenario, period_label, years)
                except Exception as e:
                    print(f"[ERROR] {variable} / {scenario} / {period_label}: {e}")
                    print("Continuing with next download...\n")

    print("\n=== Downloads complete ===")
    print(f"Files saved in: {os.path.abspath(RAW_DIR)}")
    print("Next step: run process_heat.py")


if __name__ == "__main__":
    main()