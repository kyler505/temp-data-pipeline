"""Fetch daily Tmax forecasts from Open-Meteo API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from tempdata.config import clean_openmeteo_forecast_dir, raw_openmeteo_forecast_dir
from tempdata.fetch.noaa_hourly import StationMeta, resolve_station
from tempdata.schemas.daily_tmax_forecast import (
    DAILY_TMAX_FORECAST_FIELDS,
    validate_daily_tmax_forecast,
)

# Open-Meteo forecast API base URLs
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def _compute_lead_hours(
    issue_time_utc: datetime,
    target_date_str: str,
    station_tz: str,
) -> int:
    """Compute lead hours from issue time to target date midnight in station timezone.

    Args:
        issue_time_utc: When the forecast was issued (tz-aware UTC)
        target_date_str: Target date as YYYY-MM-DD string
        station_tz: Station timezone (e.g., "America/New_York")

    Returns:
        Lead hours (floor of the difference)
    """
    tz = ZoneInfo(station_tz)
    # Parse target date and set to midnight in station timezone
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    target_midnight_local = target_date.replace(tzinfo=tz)
    # Convert to UTC for comparison
    target_midnight_utc = target_midnight_local.astimezone(timezone.utc)
    # Compute lead hours
    delta_seconds = (target_midnight_utc - issue_time_utc).total_seconds()
    return int(delta_seconds // 3600)


def _fetch_openmeteo_json(
    lat: float,
    lon: float,
    station_tz: str,
    forecast_days: int,
) -> dict:
    """Fetch forecast JSON from Open-Meteo API.

    Args:
        lat: Latitude
        lon: Longitude
        station_tz: Station timezone for daily aggregation
        forecast_days: Number of days to forecast

    Returns:
        JSON response as dict

    Raises:
        requests.HTTPError: If API request fails
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "timezone": station_tz,
        "forecast_days": forecast_days,
    }
    response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_openmeteo_historical_json(
    lat: float,
    lon: float,
    station_tz: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Fetch historical forecast JSON from Open-Meteo Historical Forecast API.

    Args:
        lat: Latitude
        lon: Longitude
        station_tz: Station timezone for daily aggregation
        start_date: Start date as YYYY-MM-DD
        end_date: End date as YYYY-MM-DD (inclusive)

    Returns:
        JSON response as dict

    Raises:
        requests.HTTPError: If API request fails
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "timezone": station_tz,
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(OPENMETEO_HISTORICAL_FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _parse_forecast_response(
    data: dict,
    station_id: str,
    lat: float,
    lon: float,
    station_tz: str,
    issue_time_utc: datetime,
    ingested_at_utc: datetime,
) -> pd.DataFrame:
    """Parse Open-Meteo JSON response into DataFrame.

    Args:
        data: JSON response from Open-Meteo API
        station_id: Station identifier
        lat: Latitude
        lon: Longitude
        station_tz: Station timezone
        issue_time_utc: When forecast was issued
        ingested_at_utc: When data was ingested

    Returns:
        DataFrame conforming to daily_tmax_forecast schema
    """
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    if not times or not temps:
        return pd.DataFrame(columns=DAILY_TMAX_FORECAST_FIELDS)

    rows = []
    for date_str, temp_c in zip(times, temps):
        if temp_c is None:
            continue

        temp_f = _celsius_to_fahrenheit(temp_c)
        lead_hours = _compute_lead_hours(issue_time_utc, date_str, station_tz)

        # target_date_local as tz-naive datetime at midnight
        target_date_local = pd.Timestamp(date_str)

        rows.append(
            {
                "station_id": station_id,
                "lat": lat,
                "lon": lon,
                "issue_time_utc": issue_time_utc,
                "target_date_local": target_date_local,
                "tmax_pred_c": temp_c,
                "tmax_pred_f": temp_f,
                "lead_hours": lead_hours,
                "model": "openmeteo",
                "source": "openmeteo",
                "ingested_at_utc": ingested_at_utc,
            }
        )

    df = pd.DataFrame(rows, columns=DAILY_TMAX_FORECAST_FIELDS)

    # Ensure proper dtypes
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
    df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], utc=True)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    df["lead_hours"] = df["lead_hours"].astype(int)

    return df


def fetch_openmeteo_daily_tmax_forecast(
    station_id: str,
    lat: float | None = None,
    lon: float | None = None,
    station_tz: str | None = None,
    out_raw_dir: str | Path | None = None,
    out_parquet_dir: str | Path | None = None,
    forecast_days: int = 14,
    write_raw: bool = True,
) -> list[Path]:
    """Fetch daily Tmax forecast from Open-Meteo, write raw JSON and normalized parquet.

    Args:
        station_id: Station identifier (e.g., "KLGA")
        lat: Latitude (optional, resolved from stations.csv if not provided)
        lon: Longitude (optional, resolved from stations.csv if not provided)
        station_tz: Station timezone (optional, resolved from stations.csv if not provided)
        out_raw_dir: Directory for raw JSON output (optional)
        out_parquet_dir: Directory for parquet output (optional)
        forecast_days: Number of days to forecast (default 14)
        write_raw: Whether to write raw JSON (default True)

    Returns:
        List of written file paths

    Raises:
        requests.HTTPError: If API request fails
        ValueError: If schema validation fails
        KeyError: If station not found in stations.csv
    """
    # Resolve station metadata if not provided
    if lat is None or lon is None or station_tz is None:
        station = resolve_station(station_id)
        lat = lat if lat is not None else station.lat
        lon = lon if lon is not None else station.lon
        station_tz = station_tz if station_tz is not None else station.tz

    # Set up output directories
    raw_dir = Path(out_raw_dir) if out_raw_dir else raw_openmeteo_forecast_dir(station_id)
    parquet_dir = Path(out_parquet_dir) if out_parquet_dir else clean_openmeteo_forecast_dir(station_id)

    # Timestamps
    now_utc = datetime.now(timezone.utc)
    issue_time_utc = now_utc
    ingested_at_utc = now_utc

    # Format for filenames: YYYY-MM-DDTHH (hour precision)
    issue_str = issue_time_utc.strftime("%Y-%m-%dT%H")

    # Fetch from API
    data = _fetch_openmeteo_json(lat, lon, station_tz, forecast_days)

    written: list[Path] = []

    # Write raw JSON (optional)
    if write_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{issue_str}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        written.append(raw_path)

    # Parse and normalize
    df = _parse_forecast_response(
        data,
        station_id=station_id,
        lat=lat,
        lon=lon,
        station_tz=station_tz,
        issue_time_utc=issue_time_utc,
        ingested_at_utc=ingested_at_utc,
    )

    # Validate schema before writing
    validate_daily_tmax_forecast(df)

    # Atomic write: write to temp file, then rename
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"part-{issue_str}.parquet"
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(parquet_path)
    written.append(parquet_path)

    return written


def fetch_openmeteo_daily_tmax_forecast_for_station(
    station_id: str,
    forecast_days: int = 14,
    write_raw: bool = True,
) -> list[Path]:
    """Convenience function to fetch forecast for a station by ID.

    Resolves station metadata from stations.csv automatically.

    Args:
        station_id: Station identifier (e.g., "KLGA")
        forecast_days: Number of days to forecast (default 14)
        write_raw: Whether to write raw JSON (default True)

    Returns:
        List of written file paths
    """
    station = resolve_station(station_id)
    return fetch_openmeteo_daily_tmax_forecast(
        station_id=station.station_id,
        lat=station.lat,
        lon=station.lon,
        station_tz=station.tz,
        forecast_days=forecast_days,
        write_raw=write_raw,
    )


def fetch_openmeteo_historical_forecasts(
    station_id: str,
    start_date: str,
    end_date: str,
    lat: float | None = None,
    lon: float | None = None,
    station_tz: str | None = None,
    out_raw_dir: str | Path | None = None,
    out_parquet_dir: str | Path | None = None,
    write_raw: bool = True,
) -> tuple[list[Path], pd.DataFrame]:
    """Fetch historical daily Tmax forecasts from Open-Meteo Historical Forecast API.

    This retrieves archived forecasts for past dates, useful for backtesting
    and model training on historical forecast-truth pairs.

    Args:
        station_id: Station identifier (e.g., "KLGA")
        start_date: Start date as YYYY-MM-DD
        end_date: End date as YYYY-MM-DD (inclusive)
        lat: Latitude (optional, resolved from stations.csv if not provided)
        lon: Longitude (optional, resolved from stations.csv if not provided)
        station_tz: Station timezone (optional, resolved from stations.csv if not provided)
        out_raw_dir: Directory for raw JSON output (optional)
        out_parquet_dir: Directory for parquet output (optional)
        write_raw: Whether to write raw JSON (default True)

    Returns:
        Tuple of (list of written file paths, DataFrame with all forecasts)

    Raises:
        requests.HTTPError: If API request fails
        ValueError: If schema validation fails
        KeyError: If station not found in stations.csv
    """
    # Resolve station metadata if not provided
    if lat is None or lon is None or station_tz is None:
        station = resolve_station(station_id)
        lat = lat if lat is not None else station.lat
        lon = lon if lon is not None else station.lon
        station_tz = station_tz if station_tz is not None else station.tz

    # Set up output directories
    raw_dir = Path(out_raw_dir) if out_raw_dir else raw_openmeteo_forecast_dir(station_id)
    parquet_dir = Path(out_parquet_dir) if out_parquet_dir else clean_openmeteo_forecast_dir(station_id)

    # Timestamps
    now_utc = datetime.now(timezone.utc)
    ingested_at_utc = now_utc

    # For historical forecasts, we simulate issue_time as midnight UTC of each target date
    # This is a simplification; in reality, forecasts are issued at specific model run times
    # The Historical Forecast API returns what the forecast was for each date
    
    # Fetch from Historical Forecast API
    data = _fetch_openmeteo_historical_json(lat, lon, station_tz, start_date, end_date)

    written: list[Path] = []

    # Write raw JSON (optional)
    if write_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"historical_{start_date}_to_{end_date}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        written.append(raw_path)

    # Parse the response
    daily = data.get("daily", {})
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])

    if not times or not temps:
        df = pd.DataFrame(columns=DAILY_TMAX_FORECAST_FIELDS)
    else:
        rows = []
        tz = ZoneInfo(station_tz)
        
        for date_str, temp_c in zip(times, temps):
            if temp_c is None:
                continue

            temp_f = _celsius_to_fahrenheit(temp_c)
            
            # For historical forecasts, we use midnight of the previous day as issue time
            # This simulates a 1-day ahead forecast (lead_hours ~ 24-48)
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            # Issue time: midnight UTC of the day before target
            issue_time_utc = datetime(
                target_date.year, target_date.month, target_date.day,
                tzinfo=timezone.utc
            ) - pd.Timedelta(days=1)
            
            lead_hours = _compute_lead_hours(issue_time_utc, date_str, station_tz)
            target_date_local = pd.Timestamp(date_str)

            rows.append(
                {
                    "station_id": station_id,
                    "lat": lat,
                    "lon": lon,
                    "issue_time_utc": issue_time_utc,
                    "target_date_local": target_date_local,
                    "tmax_pred_c": temp_c,
                    "tmax_pred_f": temp_f,
                    "lead_hours": lead_hours,
                    "model": "openmeteo",
                    "source": "openmeteo",
                    "ingested_at_utc": ingested_at_utc,
                }
            )

        df = pd.DataFrame(rows, columns=DAILY_TMAX_FORECAST_FIELDS)
        
        # Ensure proper dtypes
        df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
        df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], utc=True)
        df["target_date_local"] = pd.to_datetime(df["target_date_local"])
        df["lead_hours"] = df["lead_hours"].astype(int)

    # Validate schema before writing
    validate_daily_tmax_forecast(df)

    # Write parquet
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"historical_{start_date}_to_{end_date}.parquet"
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(parquet_path)
    written.append(parquet_path)

    return written, df
