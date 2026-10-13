"""
load_to_neon.py
---------------
Loads processed district boundaries and risk scores into Neon PostgreSQL.

Input:
    - boundaries/india_districts_clean.geojson  (719 districts)
    - processed/district_risk_scores.csv        (5752 rows)

Creates:
    - districts      table (district metadata + geometry)
    - district_risk  table (all risk scores by district/year/scenario)

Usage:
    python scripts/load_to_neon.py

Requirements:
    - .env file with DATABASE_URL=postgresql://...
    - pip install psycopg2-binary python-dotenv
"""

import os
import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY   = os.path.join(BASE_DIR, "boundaries", "india_districts_clean.geojson")
SCORES_CSV = os.path.join(BASE_DIR, "processed", "district_risk_scores.csv")

# Map period labels to year integers
PERIOD_TO_YEAR = {
    "baseline": 2025,
    "2030":     2030,
    "2040":     2040,
    "2050":     2050,
}


def get_connection():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env file")
    return psycopg2.connect(db_url)


def enable_postgis(cur):
    print("[SETUP] Enabling PostGIS extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")


def create_tables(cur):
    print("[SETUP] Creating tables...")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS districts (
            district_id   TEXT PRIMARY KEY,
            district_name TEXT,
            state         TEXT,
            primary_crop  TEXT,
            geometry      GEOMETRY(GEOMETRY, 4326)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS district_risk (
            district_id           TEXT REFERENCES districts(district_id),
            year                  INTEGER,
            scenario              TEXT,
            heat_score            FLOAT,
            drought_score         FLOAT,
            flood_score           FLOAT,
            overall_score         FLOAT,
            heat_label            TEXT,
            drought_label         TEXT,
            flood_label           TEXT,
            overall_label         TEXT,
            mean_temp_c           FLOAT,
            max_temp_c            FLOAT,
            extreme_heat_days     FLOAT,
            temp_rise_c           FLOAT,
            mean_annual_precip_mm FLOAT,
            dry_months            FLOAT,
            precip_change_pct     FLOAT,
            monsoon_precip_mm     FLOAT,
            heavy_precip_months   FLOAT,
            PRIMARY KEY (district_id, year, scenario)
        );
    """)

    print("[SETUP] Tables ready")


def load_districts(cur):
    print("\n[LOAD] Loading district boundaries...")

    gdf = gpd.read_file(BOUNDARY)
    gdf = gdf.to_crs("EPSG:4326")

    # Recreate district_id in case it's missing
    if "district_id" not in gdf.columns:
        gdf["district_id"] = (
            gdf["st_code"].astype(str).str.zfill(2) + "_" +
            gdf["dt_code"].astype(str).str.zfill(3)
        )

    # Convert geometry to WKT for PostGIS
    gdf["wkt"] = gdf.geometry.to_wkt()

    rows = [
        (
            row["district_id"],
            row["district"],
            row["st_nm"],
            None,  # primary_crop — filled later
            row["wkt"],
        )
        for _, row in gdf.iterrows()
    ]

    execute_batch(cur, """
        INSERT INTO districts (district_id, district_name, state, primary_crop, geometry)
        VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326))
        ON CONFLICT DO NOTHING;
    """, rows, page_size=100)

    print(f"[LOAD] Inserted {len(rows)} districts")


def load_risk_scores(cur):
    print("\n[LOAD] Loading risk scores...")

    df = pd.read_csv(SCORES_CSV)

    # Map period → year
    df["year"] = df["period"].map(PERIOD_TO_YEAR)

    # Drop rows where year mapping failed
    missing = df["year"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} rows with unknown period — skipping")
        df = df[df["year"].notna()]

    df["year"] = df["year"].astype(int)

    rows = [
        (
            row.district_id,
            row.year,
            row.scenario,
            row.heat_score,
            row.drought_score,
            row.flood_score,
            row.overall_score,
            row.heat_label,
            row.drought_label,
            row.flood_label,
            row.overall_label,
            row.mean_temp_c,
            row.max_temp_c,
            row.extreme_heat_days,
            row.temp_rise_c,
            row.mean_annual_precip_mm,
            row.dry_months,
            row.precip_change_pct,
            row.monsoon_precip_mm,
            row.heavy_precip_months,
        )
        for row in df.itertuples(index=False)
    ]

    execute_batch(cur, """
        INSERT INTO district_risk (
            district_id, year, scenario,
            heat_score, drought_score, flood_score, overall_score,
            heat_label, drought_label, flood_label, overall_label,
            mean_temp_c, max_temp_c, extreme_heat_days, temp_rise_c,
            mean_annual_precip_mm, dry_months, precip_change_pct,
            monsoon_precip_mm, heavy_precip_months
        )
        VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s
        )
        ON CONFLICT DO NOTHING;
    """, rows, page_size=500)

    print(f"[LOAD] Inserted {len(rows)} risk score rows")


def verify(cur):
    print("\n[VERIFY] Checking row counts...")

    cur.execute("SELECT COUNT(*) FROM districts;")
    print(f"  districts     : {cur.fetchone()[0]} rows")

    cur.execute("SELECT COUNT(*) FROM district_risk;")
    print(f"  district_risk : {cur.fetchone()[0]} rows")

    cur.execute("""
        SELECT year, scenario, COUNT(*) 
        FROM district_risk 
        GROUP BY year, scenario 
        ORDER BY year, scenario;
    """)
    print("\n  Breakdown by year + scenario:")
    for row in cur.fetchall():
        print(f"    year={row[0]}  scenario={row[1]}  districts={row[2]}")


def main():
    print("=== ClimateWatch Neon Loader ===\n")

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        enable_postgis(cur)
        create_tables(cur)
        load_districts(cur)
        load_risk_scores(cur)

        conn.commit()
        print("\n[COMMIT] All data committed successfully")

        verify(cur)

        cur.close()
        print("\n=== Load complete ===")
        print("Next step: set up FastAPI backend")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        if conn:
            conn.rollback()
            print("[ROLLBACK] Transaction rolled back")
        raise
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()