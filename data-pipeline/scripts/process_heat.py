"""
process_heat.py
---------------
Processes CMIP6 near-surface air temperature data for India.
Interpolates grid to district centroids and computes heat metrics.

Metrics computed per district per scenario per period:
- mean_temp_c        : Mean annual temperature (°C)
- max_temp_c         : Mean of monthly maximums (°C)
- temp_rise_c        : Temperature rise vs baseline (°C) — 0 for baseline itself
- extreme_heat_days  : Estimated days per year above 40°C
- heat_score         : Raw score before normalization (0–10 will be done in normalize_scores.py)

Output: processed/heat_{scenario}_{period}.csv

Usage:
    python scripts/process_heat.py
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.interpolate import griddata

# Paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(BASE_DIR, "extracted")
BOUNDARY = os.path.join(BASE_DIR, "boundaries", "india_districts_clean.geojson")
OUT_DIR     = os.path.join(BASE_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = ["ssp2_4_5", "ssp5_8_5"]
PERIODS   = ["baseline", "2030", "2040", "2050"]

# Kelvin to Celsius
K_TO_C = 273.15

# Threshold for extreme heat day estimate (°C)
EXTREME_HEAT_THRESHOLD = 40.0


def load_districts():
    print("[INFO] Loading district boundaries...")
    gdf = gpd.read_file(BOUNDARY)
    gdf = gdf.to_crs("EPSG:4326")

    # Compute centroids for interpolation
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y

    print(f"[INFO] Loaded {len(gdf)} districts")
    return gdf


def load_netcdf(scenario, period):
    """Find and load the temperature NetCDF for a given scenario + period."""
    folder = os.path.join(EXTRACT_DIR, f"near_surface_air_temperature_{scenario}_{period}")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Extracted folder not found: {folder}")

    nc_files = glob.glob(os.path.join(folder, "*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in: {folder}")

    print(f"[INFO] Loading: {os.path.basename(nc_files[0])}")
    ds = xr.open_dataset(nc_files[0])
    return ds


def compute_monthly_stats(ds):
    """
    From monthly temperature data compute:
    - Monthly mean temperature grid (time, lat, lon)
    - Annual mean per grid point
    - Monthly max per grid point (proxy for hottest months)
    """
    tas = ds["tas"]  # shape: (time, lat, lon)

    # Convert Kelvin to Celsius
    tas_c = tas - K_TO_C

    # Monthly means (averaged across all years in the period)
    monthly_mean = tas_c.groupby("time.month").mean(dim="time")  # (12, lat, lon)

    # Hottest month mean per grid point
    hottest_month = monthly_mean.max(dim="month")  # (lat, lon)

    return tas_c, monthly_mean, hottest_month


def estimate_extreme_heat_days(monthly_mean_grid, lats, lons):
    """
    Estimate annual extreme heat days (>40°C) per grid point.
    Uses a simple threshold approach on the hottest month mean.
    A more accurate approach would use daily data — this is a reasonable proxy.

    Returns a 2D array (lat, lon) of estimated extreme heat days per year.
    """
    # Assume temperature follows a normal distribution within each month
    # If monthly mean is T, std ~3°C, then days above threshold can be estimated
    std_assumption = 3.0
    days_per_month = 30

    extreme_days = np.zeros((len(lats), len(lons)))

    for i in range(len(lats)):
        for j in range(len(lons)):
            annual_days = 0
            for m in range(12):
                mean_m = float(monthly_mean_grid[m, i, j])
                # Fraction of days above threshold using normal CDF approximation
                z = (EXTREME_HEAT_THRESHOLD - mean_m) / std_assumption
                # Probability of exceeding threshold
                prob = 0.5 * (1 - np.tanh(z * 0.7978846))  # approx erfc
                annual_days += prob * days_per_month

            extreme_days[i, j] = annual_days

    return extreme_days


def interpolate_to_districts(data_2d, lats, lons, district_lats, district_lons):
    """
    Bilinear interpolation from CMIP6 grid to district centroids.
    data_2d: numpy array (lat, lon)
    Returns: array of values at each district centroid
    """
    # Build grid of points
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    points = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])
    values = data_2d.ravel()

    # Interpolate to district centroids
    district_points = np.column_stack([district_lons, district_lats])
    interpolated = griddata(points, values, district_points, method="linear")

    # Fill any NaN at edges with nearest neighbour
    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(
            points, values, district_points[nan_mask], method="nearest"
        )

    return interpolated


def process(scenario, period, gdf, baseline_means=None):
    print(f"\n[PROCESS] scenario={scenario} period={period}")

    ds = load_netcdf(scenario, period)

    lats = ds.lat.values
    lons = ds.lon.values

    tas_c, monthly_mean, hottest_month = compute_monthly_stats(ds)

    # Annual mean per grid point
    annual_mean_grid = tas_c.mean(dim="time").values  # (lat, lon)

    # Extreme heat days grid
    monthly_mean_np = monthly_mean.values  # (12, lat, lon)
    extreme_days_grid = estimate_extreme_heat_days(monthly_mean_np, lats, lons)

    # Hottest month grid
    hottest_month_grid = hottest_month.values  # (lat, lon)

    # Interpolate all grids to district centroids
    dist_lats = gdf["centroid_lat"].values
    dist_lons = gdf["centroid_lon"].values

    mean_temp    = interpolate_to_districts(annual_mean_grid,  lats, lons, dist_lats, dist_lons)
    extreme_days = interpolate_to_districts(extreme_days_grid, lats, lons, dist_lats, dist_lons)
    max_temp     = interpolate_to_districts(hottest_month_grid, lats, lons, dist_lats, dist_lons)

    # Build output dataframe
    df = pd.DataFrame({
        "district_id":       gdf["district_id"].values,
        "district_name":     gdf["district"].values,
        "state":             gdf["st_nm"].values,
        "scenario":          scenario,
        "period":            period,
        "mean_temp_c":       np.round(mean_temp, 2),
        "max_temp_c":        np.round(max_temp, 2),
        "extreme_heat_days": np.round(extreme_days, 1),
    })

    # Compute temperature rise vs baseline
    if baseline_means is not None:
        df["temp_rise_c"] = np.round(df["mean_temp_c"].values - baseline_means, 2)
    else:
        df["temp_rise_c"] = 0.0  # baseline has 0 rise by definition

    out_path = os.path.join(OUT_DIR, f"heat_{scenario}_{period}.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path} ({len(df)} districts)")

    return df


def main():
    print("=== ClimateWatch Heat Processor ===\n")

    gdf = load_districts()

    for scenario in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print('='*50)

        # Process baseline first to get reference temperatures
        baseline_df = process(scenario, "baseline", gdf, baseline_means=None)
        baseline_means = baseline_df["mean_temp_c"].values

        # Process future periods using baseline as reference
        for period in ["2030", "2040", "2050"]:
            process(scenario, period, gdf, baseline_means=baseline_means)

    print("\n=== Heat processing complete ===")
    print(f"Output files in: {OUT_DIR}")
    print("Next step: run process_precipitation.py")


if __name__ == "__main__":
    main()