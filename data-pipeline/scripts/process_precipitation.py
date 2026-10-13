"""
process_precipitation.py
------------------------
Processes CMIP6 precipitation data for India.
Computes both drought and flood metrics per district per scenario per period.

Metrics computed:
- mean_precip_mm      : Mean annual precipitation (mm)
- dry_months          : Average months per year with precip < 10mm (drought proxy)
- precip_deficit_pct  : % change in annual precip vs baseline (negative = deficit)
- heavy_precip_days   : Estimated days/year with heavy rainfall (flood proxy)
- monsoon_intensity   : Mean precip in Jun–Sep (monsoon months)
- drought_score_raw   : Raw drought score before normalization
- flood_score_raw     : Raw flood score before normalization

Output: 
    processed/precipitation_{scenario}_{period}.csv

Usage:
    python scripts/process_precipitation.py
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

# CMIP6 precipitation unit is kg/m²/s — convert to mm/day
KG_TO_MM_DAY = 86400.0

# Thresholds
DRY_MONTH_THRESHOLD_MM   = 10.0   # months with < 10mm = dry
HEAVY_PRECIP_THRESHOLD   = 20.0   # mm/day = heavy rainfall day

# Monsoon months (June=6 to September=9)
MONSOON_MONTHS = [6, 7, 8, 9]


def load_districts():
    print("[INFO] Loading district boundaries...")
    gdf = gpd.read_file(BOUNDARY)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    print(f"[INFO] Loaded {len(gdf)} districts")
    return gdf


def load_netcdf(scenario, period):
    folder = os.path.join(EXTRACT_DIR, f"precipitation_{scenario}_{period}")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Extracted folder not found: {folder}")

    nc_files = glob.glob(os.path.join(folder, "*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in: {folder}")

    print(f"[INFO] Loading: {os.path.basename(nc_files[0])}")
    ds = xr.open_dataset(nc_files[0])
    return ds


def compute_precip_stats(ds):
    """
    Compute precipitation statistics from monthly CMIP6 data.
    Input unit: kg/m²/s → convert to mm/month for usable values.
    """
    pr = ds["pr"]  # (time, lat, lon) in kg/m²/s

    # Convert to mm/month (kg/m²/s × 86400 s/day × ~30 days/month)
    pr_mm_month = pr * KG_TO_MM_DAY * 30.0

    # Annual total (mm/year) — mean across all years in period
    annual_total = pr_mm_month.groupby("time.year").sum(dim="time").mean(dim="year")  # (lat, lon)

    # Monthly climatology (mm/month averaged across all years)
    monthly_clim = pr_mm_month.groupby("time.month").mean(dim="time")  # (12, lat, lon)

    # Dry months per year (months with < 10mm)
    dry_months = (monthly_clim < DRY_MONTH_THRESHOLD_MM).sum(dim="month").astype(float)  # (lat, lon)

    # Monsoon total (Jun–Sep)
    monsoon = monthly_clim.sel(month=MONSOON_MONTHS).sum(dim="month")  # (lat, lon)

    # Heavy precip proxy: months with > HEAVY_PRECIP_THRESHOLD * 30mm total
    # (proxy for high-intensity rainfall months → flood risk)
    heavy_months = (monthly_clim > HEAVY_PRECIP_THRESHOLD * 30).sum(dim="month").astype(float)

    return (
        annual_total.values,
        monthly_clim.values,
        dry_months.values,
        monsoon.values,
        heavy_months.values,
    )


def interpolate_to_districts(data_2d, lats, lons, dist_lats, dist_lons):
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    points = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])
    values = data_2d.ravel()
    district_points = np.column_stack([dist_lons, dist_lats])

    interpolated = griddata(points, values, district_points, method="linear")

    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(
            points, values, district_points[nan_mask], method="nearest"
        )

    return interpolated


def process(scenario, period, gdf, baseline_annual=None):
    print(f"\n[PROCESS] scenario={scenario} period={period}")

    ds = load_netcdf(scenario, period)
    lats = ds.lat.values
    lons = ds.lon.values

    annual_grid, monthly_grid, dry_months_grid, monsoon_grid, heavy_grid = compute_precip_stats(ds)

    dist_lats = gdf["centroid_lat"].values
    dist_lons = gdf["centroid_lon"].values

    annual_precip  = interpolate_to_districts(annual_grid,     lats, lons, dist_lats, dist_lons)
    dry_months     = interpolate_to_districts(dry_months_grid, lats, lons, dist_lats, dist_lons)
    monsoon_precip = interpolate_to_districts(monsoon_grid,    lats, lons, dist_lats, dist_lons)
    heavy_months   = interpolate_to_districts(heavy_grid,      lats, lons, dist_lats, dist_lons)

    # Precipitation deficit vs baseline
    if baseline_annual is not None:
        # Negative = deficit (drought), positive = excess (flood risk)
        precip_change_pct = np.round(
            ((annual_precip - baseline_annual) / (baseline_annual + 1e-6)) * 100, 2
        )
    else:
        precip_change_pct = np.zeros(len(gdf))

    df = pd.DataFrame({
        "district_id":        gdf["district_id"].values,
        "district_name":      gdf["district"].values,
        "state":              gdf["st_nm"].values,
        "scenario":           scenario,
        "period":             period,
        "mean_annual_precip_mm": np.round(annual_precip, 1),
        "dry_months":         np.round(dry_months, 1),
        "precip_change_pct":  precip_change_pct,
        "monsoon_precip_mm":  np.round(monsoon_precip, 1),
        "heavy_precip_months": np.round(heavy_months, 1),
    })

    out_path = os.path.join(OUT_DIR, f"precipitation_{scenario}_{period}.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path} ({len(df)} districts)")

    return df, annual_precip


def main():
    print("=== ClimateWatch Precipitation Processor ===\n")

    gdf = load_districts()

    for scenario in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print('='*50)

        # Baseline first — no deficit calculation
        baseline_df, baseline_annual = process(scenario, "baseline", gdf, baseline_annual=None)

        # Future periods — compute deficit vs baseline
        for period in ["2030", "2040", "2050"]:
            process(scenario, period, gdf, baseline_annual=baseline_annual)

    print("\n=== Precipitation processing complete ===")
    print(f"Output files in: {OUT_DIR}")
    print("Next step: run normalize_scores.py")


if __name__ == "__main__":
    main()