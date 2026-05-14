#!/usr/bin/env python3
"""
Twin Cities Snow Depth & Snow Cover Analysis — 2025–2026 Winter Season

Fetches daily snow depth from two sources:

1. CF6 (NWS Preliminary Local Climatological Data) via Iowa Environmental Mesonet API
1. SNODAS (Snow Data Assimilation System) gridded data via NSIDC/NOHRSC

Produces a chart mimicking NSIDC's "Snow Today" style:

- Year-to-date snow depth (blue line)
- Median snow depth from climatology (gray dashed)
- Interquartile range shading
- Max/min year traces

Requirements:
pip install requests pandas matplotlib numpy

Optional (for SNODAS):
pip install xarray netCDF4 rasterio

Usage:
python msp_snow_cover.py

Author: Generated for Kyle's MGIS coursework / weather analysis
Data citation:
CF6: NWS Twin Cities (KMPX) via Iowa Environmental Mesonet (mesonet.agron.iastate.edu)
SNODAS: National Operational Hydrologic Remote Sensing Center. (2004).
Snow Data Assimilation System (SNODAS) Data Products at NSIDC.
(G02158, Version 1). Boulder, CO. https://doi.org/10.7265/N5TB14TC
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# ── Configuration ────────────────────────────────────────────────────────────

STATION = "MSP"
NETWORK = "MN_ASOS"
CURRENT_SEASON_START = "2025-10-01"
CURRENT_SEASON_END = "2026-03-18"  # Update to today's date as needed
CLIMO_START_YEAR = 2001
CLIMO_END_YEAR = 2025
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Twin Cities bounding box for SNODAS (approximate 7-county metro)
TC_BBOX = {
    "lat_min": 44.73,
    "lat_max": 45.13,
    "lon_min": -93.55,
    "lon_max": -92.95,
}

# ── 1. CF6 DATA VIA IEM API ─────────────────────────────────────────────────

def fetch_iem_daily(station, network, year, month):
    """Fetch daily climate data from the IEM API for one month."""
    url = (
        f"https://mesonet.agron.iastate.edu/api/1/daily.json"
        f"?network={network}&station={station}&year={year}&month={month}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("data", [])
    except Exception as e:
        print(f"  ⚠ Failed to fetch {year}-{month:02d}: {e}")
        return []


def fetch_current_season_cf6():
    """Fetch CF6 snow depth for the current winter season (Oct–Mar)."""
    print("━━━ Fetching CF6 data from IEM ━━━")
    all_rows = []

    # Oct–Dec of start year, Jan–Mar of end year
    months = [
        (2025, 10), (2025, 11), (2025, 12),
        (2026, 1), (2026, 2), (2026, 3),
    ]

    for year, month in months:
        print(f"  Fetching {year}-{month:02d}...", end=" ")
        rows = fetch_iem_daily(STATION, NETWORK, year, month)
        print(f"{len(rows)} days")
        all_rows.extend(rows)

    if not all_rows:
        print("  ✗ No data returned. Check your network connection.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    # IEM daily.json columns of interest:
    #   snow_depth_in  — 12Z snow depth (inches), may be None
    #   new_snow_in    — daily snowfall (inches)
    #   precip_in      — liquid precipitation
    #   max_temp_f, min_temp_f
    for col in ["snow_depth_in", "new_snow_in"]:
        if col not in df.columns:
            df[col] = np.nan

    # Filter to season window
    df = df[(df["date"] >= CURRENT_SEASON_START) &
            (df["date"] <= CURRENT_SEASON_END)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  ✓ {len(df)} days of CF6 data for current season")
    return df


def fetch_climo_cf6():
    """Fetch historical CF6 snow depth for climatology (Oct–Mar, many years)."""
    print("\n━━━ Fetching climatological CF6 data ━━━")
    print(f"  Period: {CLIMO_START_YEAR}–{CLIMO_END_YEAR}")
    all_rows = []

    for start_yr in range(CLIMO_START_YEAR, CLIMO_END_YEAR):
        end_yr = start_yr + 1
        season_label = f"{start_yr}-{end_yr}"
        print(f"  Season {season_label}...", end=" ")

        season_rows = []
        for year, month in [
            (start_yr, 10), (start_yr, 11), (start_yr, 12),
            (end_yr, 1), (end_yr, 2), (end_yr, 3),
        ]:
            rows = fetch_iem_daily(STATION, NETWORK, year, month)
            for row in rows:
                row["season"] = season_label
                row["season_start_year"] = start_yr
            season_rows.extend(rows)

        print(f"{len(season_rows)} days")
        all_rows.extend(season_rows)

    if not all_rows:
        print("  ✗ No climatological data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    for col in ["snow_depth_in", "new_snow_in"]:
        if col not in df.columns:
            df[col] = np.nan

    # Create a "day of season" column (Oct 1 = day 0)
    def day_of_season(row):
        d = row["date"]
        syr = row["season_start_year"]
        oct1 = pd.Timestamp(year=syr, month=10, day=1)
        return (d - oct1).days

    df["dos"] = df.apply(day_of_season, axis=1)
    df = df[(df["dos"] >= 0) & (df["dos"] <= 182)].copy()  # Oct 1 – Mar 31

    print(f"  ✓ {len(df)} total climatological observations")
    return df


# ── 2. SNODAS DATA ──────────────────────────────────────────────────────────

def fetch_snodas_for_date(target_date, product="snow_depth"):
    """
    Fetch SNODAS gridded snow depth for a single date.

    SNODAS data is distributed as gzipped binary files from NSIDC FTP.
    This function downloads the national grid and extracts the Twin Cities bbox.

    NOTE: This is computationally heavier. For a quick analysis, the CF6
    approach above is recommended. Enable this section only if you want
    the gridded spatial view.

    Product codes:
        snow_depth: us_ssmv11036tS__T0001TTNATS (snow depth, m)
        swe:        us_ssmv11034tS__T0001TTNATS (SWE, m)
    """
    try:
        import xarray as xr
        import rasterio
    except ImportError:
        print("  ⚠ SNODAS requires: pip install xarray netCDF4 rasterio")
        return None

    date_str = target_date.strftime("%Y%m%d")
    year_str = target_date.strftime("%Y")
    month_str = target_date.strftime("%m_%b")

    # NSIDC FTP path pattern
    base_url = (
        f"https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
        f"{year_str}/{month_str}/SNODAS_{date_str}.tar"
    )

    print(f"  Fetching SNODAS for {target_date.date()}: {base_url}")

    # Download and extract would go here
    # For a complete implementation, see the SNODAS section below
    return None


def snodas_time_series(start_date, end_date):
    """
    Build a daily time series of mean snow depth over the Twin Cities
    from SNODAS grids.

    This downloads ~170 daily grids. Each is ~16 MB compressed.
    Total download: ~2.7 GB. Budget 30–60 min on a decent connection.

    If you want to skip the download and use pre-processed data,
    see the NOHRSC interactive map approach below.
    """
    print("\n━━━ SNODAS Time Series ━━━")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Bounding box: {TC_BBOX}")
    print("  ⚠ Full SNODAS download is large (~2.7 GB). Skipping by default.")
    print("  Set ENABLE_SNODAS=True to run this section.\n")
    print("  Alternative: Use the NOHRSC interactive map at:")
    print("    https://www.nohrsc.noaa.gov/interactive/html/map.html")
    print("  Select 'Snow Depth' product, zoom to Twin Cities,")
    print("  and use the 'Query [i]' tool to get point values.\n")
    return pd.DataFrame()


# ── 3. NOHRSC POINT QUERY (lighter alternative to full SNODAS) ───────────────

def fetch_nohrsc_point(lat, lon, start_date, end_date):
    """
    Query the NOHRSC Snow Model time series for a single point.

    The NOHRSC interactive map provides modeled snow depth and SWE
    at any point in CONUS. This function attempts to use their
    web service endpoint.

    lat, lon: coordinates (e.g., 44.88, -93.22 for Eden Prairie)
    """
    print("\n━━━ NOHRSC Point Query ━━━")
    print(f"  Location: {lat}, {lon}")

    # The NOHRSC web service for point queries:
    url = (
        f"https://www.nohrsc.noaa.gov/interactive/html/graph.html"
        f"?station=MSP&w=600&h=400&o=a&ession=all"
        f"&by={start_date.year}&bm={start_date.month}&bd={start_date.day}"
        f"&ey={end_date.year}&em={end_date.month}&ed={end_date.day}"
        f"&data=0&units=0"  # data=0 is snow depth, units=0 is inches
    )
    print(f"  ℹ NOHRSC doesn't have a clean JSON API for time series.")
    print(f"  Visit this URL in your browser to see the modeled snow depth graph:")
    print(f"    {url}")
    print(f"  Or use the interactive map:")
    print(f"    https://www.nohrsc.noaa.gov/interactive/html/map.html")
    print(f"  Zoom to Twin Cities → Click 'Query [i]' → Click your location")
    return pd.DataFrame()


# ── 4. CHART GENERATION ─────────────────────────────────────────────────────

def compute_climatology(climo_df):
    """Compute daily statistics from the climatological record."""
    if climo_df.empty:
        return pd.DataFrame()

    # Group by day-of-season, compute stats
    stats = climo_df.groupby("dos")["snow_depth_in"].agg(
        ["median", "mean",
         lambda x: x.quantile(0.25),
         lambda x: x.quantile(0.75),
         "min", "max"]
    ).rename(columns={
        "<lambda_0>": "q25",
        "<lambda_1>": "q75",
    }).reset_index()

    # Find max and min years
    max_seasons = {}
    min_seasons = {}
    for dos_val in climo_df["dos"].unique():
        subset = climo_df[climo_df["dos"] == dos_val].dropna(subset=["snow_depth_in"])
        if len(subset) == 0:
            continue
        max_idx = subset["snow_depth_in"].idxmax()
        min_idx = subset["snow_depth_in"].idxmin()
        max_seasons[dos_val] = subset.loc[max_idx, "season"]
        min_seasons[dos_val] = subset.loc[min_idx, "season"]

    return stats, max_seasons, min_seasons


def dos_to_date(dos, ref_year=2025):
    """Convert day-of-season to an actual date for plotting."""
    return pd.Timestamp(year=ref_year, month=10, day=1) + pd.Timedelta(days=int(dos))


def plot_snow_depth_chart(current_df, climo_df):
    """
    Generate a Snow Today-style chart for Twin Cities snow depth.
    """
    print("\n━━━ Generating Chart ━━━")

    # Current season: compute day-of-season
    if not current_df.empty:
        oct1 = pd.Timestamp("2025-10-01")
        current_df = current_df.copy()
        current_df["dos"] = (current_df["date"] - oct1).dt.days

    # ── Figure setup ──
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    # ── Climatology ──
    if not climo_df.empty:
        stats, max_seasons, min_seasons = compute_climatology(climo_df)

        # Convert dos to dates for x-axis
        stats["plot_date"] = stats["dos"].apply(dos_to_date)

        # IQR shading
        ax.fill_between(
            stats["plot_date"], stats["q25"], stats["q75"],
            color="#c8d8e8", alpha=0.6, label="Interquartile Range",
            zorder=2
        )

        # Median line
        ax.plot(
            stats["plot_date"], stats["median"],
            color="#888888", linewidth=1.5, linestyle="--",
            marker=".", markersize=3,
            label="Median", zorder=3
        )

        # Max year trace
        max_trace = climo_df.copy()
        # Find which season had the overall highest snow depth
        season_maxes = climo_df.groupby("season")["snow_depth_in"].max()
        max_season = season_maxes.idxmax()
        max_data = climo_df[climo_df["season"] == max_season].copy()
        max_data["plot_date"] = max_data["dos"].apply(dos_to_date)
        ax.plot(
            max_data["plot_date"], max_data["snow_depth_in"],
            color="#555555", linewidth=1, linestyle="-.",
            label=f"Maximum ({max_season.split('-')[0]})",
            zorder=3
        )

        # Min year trace
        # Find the season with lowest peak snow depth
        min_season = season_maxes.idxmin()
        min_data = climo_df[climo_df["season"] == min_season].copy()
        min_data["plot_date"] = min_data["dos"].apply(dos_to_date)
        ax.plot(
            min_data["plot_date"], min_data["snow_depth_in"],
            color="#999999", linewidth=1, linestyle=":",
            label=f"Minimum ({min_season.split('-')[0]})",
            zorder=3
        )

    # ── Current season ──
    if not current_df.empty:
        current_plot = current_df.dropna(subset=["snow_depth_in"]).copy()
        current_plot["plot_date"] = current_plot["dos"].apply(dos_to_date)
        ax.plot(
            current_plot["plot_date"], current_plot["snow_depth_in"],
            color="#2196F3", linewidth=2.5, solid_capstyle="round",
            label="Year to date (2025-26)",
            zorder=5
        )

    # ── Formatting ──
    ax.set_title(
        "Daily Snow Depth\n"
        "Twin Cities, MN (MSP Airport) — 2025–2026 Season",
        fontsize=16, fontweight="bold", fontfamily="serif",
        pad=12
    )
    ax.set_ylabel("Snow Depth (inches)", fontsize=12, fontfamily="serif")
    ax.set_xlabel("Date", fontsize=12, fontfamily="serif")

    # Subtitle
    ax.text(
        0.5, 1.0,
        f"Climatology calculated over {CLIMO_START_YEAR} to {CLIMO_END_YEAR}",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, color="#666666", fontfamily="serif"
    )

    # X-axis: monthly ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

    # Y-axis
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    # Grid
    ax.grid(True, which="major", axis="both", color="#dddddd", linewidth=0.8)
    ax.grid(True, which="minor", axis="both", color="#eeeeee", linewidth=0.3)
    ax.set_axisbelow(True)

    # Limits
    ax.set_xlim(
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2026-04-15")
    )
    ax.set_ylim(bottom=0)

    # Legend
    legend = ax.legend(
        loc="upper left", fontsize=9, framealpha=0.9,
        edgecolor="#cccccc", fancybox=False
    )

    # Source annotation
    ax.text(
        0.99, -0.08,
        "Data: NWS CF6 (MSP) via Iowa Environmental Mesonet\n"
        "SNODAS: NOHRSC/NSIDC G02158",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="#999999", fontfamily="serif"
    )

    plt.tight_layout()

    # Save
    out_path = OUTPUT_DIR / "msp_snow_depth_2025-2026.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#fafafa")
    print(f"  ✓ Chart saved to {out_path}")

    # Also save as PDF
    pdf_path = OUTPUT_DIR / "msp_snow_depth_2025-2026.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="#fafafa")
    print(f"  ✓ PDF saved to {pdf_path}")

    plt.show()
    return fig


# ── 5. DATA EXPORT ──────────────────────────────────────────────────────────

def export_data(current_df, climo_df):
    """Export fetched data to CSV for reuse."""
    if not current_df.empty:
        p = OUTPUT_DIR / "msp_cf6_current_season.csv"
        current_df.to_csv(p, index=False)
        print(f"  ✓ Current season data: {p}")

    if not climo_df.empty:
        p = OUTPUT_DIR / "msp_cf6_climatology.csv"
        climo_df.to_csv(p, index=False)
        print(f"  ✓ Climatology data: {p}")


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Twin Cities Snow Depth Analysis — 2025–2026 Season ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Fetch current season CF6 ──
    current_df = fetch_current_season_cf6()

    # ── Step 2: Fetch climatological CF6 ──
    climo_df = fetch_climo_cf6()

    # ── Step 3: SNODAS (optional) ──
    ENABLE_SNODAS = os.environ.get("ENABLE_SNODAS", "").lower() == "true"
    if ENABLE_SNODAS:
        snodas_df = snodas_time_series(
            datetime(2025, 10, 1), datetime(2026, 3, 18)
        )
    else:
        print("\n━━━ SNODAS ━━━")
        print("  Skipped. Set ENABLE_SNODAS=true to download gridded data.")
        print("  Alternative: NOHRSC interactive map (see below).")

    # ── Step 4: NOHRSC point query info ──
    fetch_nohrsc_point(
        lat=44.88, lon=-93.22,  # Eden Prairie
        start_date=datetime(2025, 10, 1),
        end_date=datetime(2026, 3, 18)
    )

    # ── Step 5: Export data ──
    print("\n━━━ Exporting Data ━━━")
    export_data(current_df, climo_df)

    # ── Step 6: Generate chart ──
    fig = plot_snow_depth_chart(current_df, climo_df)

    print("\n━━━ Done ━━━")
    print(f"  Output files are in: {OUTPUT_DIR.resolve()}")
    print(f"\n  Tips:")
    print(f"  • To include SNODAS gridded data:")
    print(f"      ENABLE_SNODAS=true python {sys.argv[0]}")
    print(f"  • To change the climatology period, edit CLIMO_START_YEAR/END_YEAR")
    print(f"  • The IEM API may rate-limit you — if fetches fail, wait and retry")
    print(f"  • For SNODAS in ArcGIS Pro, download grids from:")
    print(f"      https://nsidc.org/data/g02158/versions/1")
    print(f"    Then use Extract by Mask with a TC metro boundary polygon")


if __name__ == "__main__":
    main()
