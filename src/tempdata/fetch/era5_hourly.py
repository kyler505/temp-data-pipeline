"""Fetch ERA5 hourly 2m temperature data via CDS API.

ERA5 is used for deep historical temperature data (1940-2015) before
Open-Meteo forecast archive availability.

Requires:
    - cdsapi package installed
    - ~/.cdsapirc configured with CDS API credentials
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import cdsapi
    HAS_CDS = True
except ImportError:
    HAS_CDS = False

from tempdata.config import raw_era5_dir, stations_csv_path
from tempdata.fetch.noaa_hourly import StationMeta, load_station_mapping, resolve_station
from tempdata.schemas.hourly_obs import (
    RAW_HOURLY_FIELDS,
    ensure_hourly_schema_columns,
    validate_hourly_obs,
)

# ERA5 data availability boundaries
ERA5_START_DATE = date(1940, 1, 1)
ERA5T_LATENCY_DAYS = 5  # ERA5T (preliminary) has ~5 day latency


@dataclass
class ERA5Request:
    """Parameters for a CDS API request."""
    product: Literal["reanalysis-era5-single-levels", "reanalysis-era5-single-levels-preliminary-plus-monthly-means"]
    variable: str
    year: int
    month: int
    days: list[int]
    hours: list[int]
    area: list[float]  # [north, west, south, east]
    output_path: Path


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _month_range(start: date, end: date):
    """Yield (year, month) tuples for the range."""
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    while current <= end_month:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _station_bounding_box(lat: float, lon: float, buffer_deg: float = 0.5) -> list[float]:
    """Create a bounding box around the station.

    ERA5 has ~31km resolution, so a 0.5 degree buffer ensures we capture
    the grid cell containing the station.

    Returns [north, west, south, east] as required by CDS API.
    """
    return [
        min(90, lat + buffer_deg),   # north
        max(-180, lon - buffer_deg), # west
        max(-90, lat - buffer_deg),  # south
        min(180, lon + buffer_deg),  # east
    ]


def fetch_era5_hourly(
    station_id: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    out_dir: str | Path | None = None,
    use_era5t: bool = True,
    force: bool = False,
) -> list[Path]:
    """Fetch ERA5 hourly 2m temperature data for a station.

    Args:
        station_id: Station identifier (e.g., "KLGA")
        start_date: Start date (inclusive)
        end_date: End date (exclusive)
        out_dir: Output directory for parquet files
        use_era5t: Whether to use ERA5T for recent dates (within 5 days)
        force: Force re-download even if files exist

    Returns:
        List of written parquet file paths

    Raises:
        ImportError: If cdsapi is not installed
        ValueError: If date range is invalid
    """
    if not HAS_CDS:
        raise ImportError(
            "cdsapi package is required for ERA5 data. "
            "Install with: pip install cdsapi"
        )

    station = resolve_station(station_id)
    start_d = _to_date(start_date)
    end_d = _to_date(end_date)

    if end_d <= start_d:
        raise ValueError("end_date must be after start_date")

    if start_d < ERA5_START_DATE:
        raise ValueError(f"ERA5 data not available before {ERA5_START_DATE}")

    output_root = Path(out_dir) if out_dir else raw_era5_dir(station.station_id)
    output_root.mkdir(parents=True, exist_ok=True)

    # Initialize CDS API client
    client = cdsapi.Client()

    # Determine bounding box for the station
    area = _station_bounding_box(station.lat, station.lon)

    written: list[Path] = []

    written: list[Path] = []

    # Iterate by year to minimize queue wait time
    # (One request per year is much faster than 12 requests)
    current_year = start_d.year
    final_year = end_d.year

    # If end_d is Jan 1st of a year, that year is excluded (end_date is exclusive)
    if end_d.month == 1 and end_d.day == 1:
        final_year -= 1

    for year in range(current_year, final_year + 1):
        # Check if output already exists
        parquet_path = output_root / f"era5_{year:04d}.parquet"
        if parquet_path.exists() and not force:
            print(f"[era5] {year}: using cached {parquet_path}")
            written.append(parquet_path)
            continue

        print(f"[era5] {year}: fetching full year...")

        # Determine months and days to fetch
        # For historical years, we want the whole year (months 1-12, days 1-31).
        # We let CDS handle the "invalid" days like Feb 30 (it ignores them).

        # However, for the *current* year (or very recent), we must respect availability.
        max_avail_date = get_era5_availability_end()

        # If the entire year is in the future, skip
        if year > max_avail_date.year:
            print(f"[era5] {year}: year is in the future (latest data: {max_avail_date})")
            continue

        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

        # If this is the max available year, we might need to be careful?
        # Actually, if we ask for future dates, CDS often returns error or empty.
        # But 'reanalysis-era5-single-levels' usually fails if any requested date is not available.
        # So for the current available year, we should only ask for available months.

        if year == max_avail_date.year:
            # Limit months to what is available
            months = [f"{m:02d}" for m in range(1, max_avail_date.month + 1)]
            # We assume we can ask for full days in those months (CDS is usually fine with that)
            # worst case, the last month is partial.
            print(f"[era5] {year}: fetching partial year (up to {max_avail_date})")

        nc_path = output_root / f"era5_{year:04d}.nc"

        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": "2m_temperature",
                    "year": str(year),
                    "month": months,
                    "day": days,
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": area,
                    "format": "netcdf",
                },
                str(nc_path),
            )
        except Exception as e:
            print(f"[era5] {year}: fetch failed: {e}")
            continue

        # Parse NetCDF to DataFrame
        df = _parse_era5_netcdf(nc_path, station)

        # Filter strictly to the requested start/end range?
        # Pro: Keeps data clean.
        # Con: If user continually shifts start_date, we might want to keep the whole year cached.
        # Decision: Keep the WHOLE year in the parquet file (cached),
        # but the caller will filter what they need.
        # However, for the very first/last year of the request, we might have fetched more than "requested"
        # but that's good for caching.

        if not df.empty:
            df.sort_values("ts_utc", inplace=True)
            df = df[ensure_hourly_schema_columns(df.columns)]
        else:
            df = pd.DataFrame(columns=RAW_HOURLY_FIELDS)

        # Validate and write
        validate_hourly_obs(df, require_unique_keys=False)

        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(parquet_path)
        written.append(parquet_path)

        # Clean up NetCDF
        if nc_path.exists():
            nc_path.unlink()

        print(f"[era5] {year}: wrote {len(df)} rows -> {parquet_path}")

    return written


def _parse_era5_netcdf(nc_path: Path, station: StationMeta) -> pd.DataFrame:
    """Parse ERA5 NetCDF file and extract timeseries for station location.

    Uses nearest-neighbor interpolation to the station coordinates.
    """
    try:
        import xarray as xr
        return _parse_with_xarray(nc_path, station)
    except ImportError:
        # Fallback to netCDF4 if xarray not available
        try:
            import netCDF4
            return _parse_with_netcdf4(nc_path, station)
        except ImportError:
            raise ImportError(
                "Either xarray or netCDF4 is required to parse ERA5 data. "
                "Install with: pip install xarray netCDF4"
            )


def _parse_with_xarray(nc_path: Path, station: StationMeta) -> pd.DataFrame:
    """Parse ERA5 NetCDF using xarray."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # ERA5 2m temperature variable name
    temp_var = "t2m" if "t2m" in ds else "2m_temperature"

    # Select nearest grid point to station
    temp = ds[temp_var].sel(
        latitude=station.lat,
        longitude=station.lon,
        method="nearest"
    )

    # Convert to DataFrame
    df = temp.to_dataframe().reset_index()

    # Rename columns
    rename_map = {
        "time": "ts_utc",
        "valid_time": "ts_utc",
        temp_var: "temp_k",
    }
    df = df.rename(columns=rename_map)

    # Convert Kelvin to Celsius
    df["temp_c"] = df["temp_k"] - 273.15

    # Build output DataFrame
    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(df["ts_utc"], utc=True),
        "station_id": station.station_id,
        "lat": station.lat,
        "lon": station.lon,
        "temp_c": df["temp_c"],
        "source": "era5",
        "qc_flags": 0,
    }, columns=RAW_HOURLY_FIELDS)

    ds.close()
    return out


def _parse_with_netcdf4(nc_path: Path, station: StationMeta) -> pd.DataFrame:
    """Parse ERA5 NetCDF using netCDF4 (fallback)."""
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path))

    # Get coordinates
    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    times = ds.variables["time"]

    # Find nearest grid point
    lat_idx = int(np.argmin(np.abs(lats - station.lat)))
    lon_idx = int(np.argmin(np.abs(lons - station.lon)))

    # Get time values
    time_units = times.units
    time_vals = nc.num2date(times[:], units=time_units, calendar="standard")

    # Get temperature (Kelvin)
    temp_var = "t2m" if "t2m" in ds.variables else "2m_temperature"
    temp_k = ds.variables[temp_var][:, lat_idx, lon_idx]

    # Convert to Celsius
    temp_c = temp_k - 273.15

    # Build output DataFrame
    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(time_vals).tz_localize("UTC"),
        "station_id": station.station_id,
        "lat": station.lat,
        "lon": station.lon,
        "temp_c": temp_c,
        "source": "era5",
        "qc_flags": 0,
    }, columns=RAW_HOURLY_FIELDS)

    ds.close()
    return out


def get_era5_availability_end() -> date:
    """Get the latest date with ERA5 data available.

    ERA5 has approximately 5-day latency for preliminary (ERA5T) data.
    Final ERA5 data has ~3 month latency.

    Returns:
        Latest date with ERA5T data available
    """
    today = datetime.now(timezone.utc).date()
    return today - timedelta(days=ERA5T_LATENCY_DAYS)
