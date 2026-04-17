"""Enriched feature engineering for temperature prediction.

Fetches additional Open-Meteo variables (humidity, wind, pressure, cloud cover,
radiation, precipitation) and computes daily aggregates as features.

New features added:
  - wind_max_kmh: Max daily wind speed (km/h)
  - wind_dir_dominant: Dominant wind direction (degrees, 0-360)
  - wind_u, wind_v: Wind direction as sin/cos components
  - precip_mm: Total daily precipitation (mm)
  - precip_hours: Hours with precipitation
  - radiation_mj: Shortwave radiation sum (MJ/m²)
  - humidity_mean: Mean relative humidity (%)
  - humidity_range: Max - min relative humidity (%)
  - pressure_mean: Mean surface pressure (hPa)
  - pressure_trend: 24h pressure change (hPa/day)
  - cloud_cover_mean: Mean cloud cover (%)
  - tmin_f: Minimum temperature (°F)
  - diurnal_range_f: Tmax - Tmin (°F)
  - apparent_tmax_f: Apparent/feels-like max temperature (°F)
  - dew_point_mean_f: Mean dew point (°F)
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OPENMETEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "precipitation_sum",
    "precipitation_hours",
    "windspeed_10m_max",
    "winddirection_10m_dominant",
    "shortwave_radiation_sum",
]

HOURLY_VARS = [
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "dew_point_2m",
]


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def fetch_enriched_historical(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz: str = "America/New_York",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch enriched Open-Meteo historical data with daily + hourly variables.

    Uses caching to avoid re-fetching the same date ranges.
    """
    cache_key = f"{lat}_{lon}_{start_date}_{end_date}".replace("-", "").replace(".", "")
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": tz,
        "start_date": start_date,
        "end_date": end_date,
    }

    resp = requests.get(OPENMETEO_HISTORICAL_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Parse daily
    daily = data.get("daily", {})
    daily_df = pd.DataFrame({
        "target_date_local": pd.to_datetime(daily["time"]),
        "tmax_pred_f": [_c_to_f(t) if t is not None else np.nan for t in daily["temperature_2m_max"]],
        "tmin_f": [_c_to_f(t) if t is not None else np.nan for t in daily["temperature_2m_min"]],
        "apparent_tmax_f": [_c_to_f(t) if t is not None else np.nan for t in daily["apparent_temperature_max"]],
        "precip_mm": daily["precipitation_sum"],
        "precip_hours": daily["precipitation_hours"],
        "wind_max_kmh": daily["windspeed_10m_max"],
        "wind_dir_dominant": daily["winddirection_10m_dominant"],
        "radiation_mj": daily["shortwave_radiation_sum"],
    })

    # Parse hourly → daily aggregates
    hourly = data.get("hourly", {})
    hourly_df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "cloud_cover": hourly["cloud_cover"],
        "dew_point_2m": hourly["dew_point_2m"],
    })
    hourly_df["date"] = hourly_df["time"].dt.date

    agg = hourly_df.groupby("date").agg(
        humidity_mean=("humidity", "mean"),
        humidity_max=("humidity", "max"),
        humidity_min=("humidity", "min"),
        pressure_mean=("pressure", "mean"),
        pressure_first=("pressure", "first"),
        pressure_last=("pressure", "last"),
        cloud_cover_mean=("cloud_cover", "mean"),
        dew_point_mean_f=("dew_point_2m", lambda x: _c_to_f(x.mean()) if x.notna().any() else np.nan),
    ).reset_index()
    agg["target_date_local"] = pd.to_datetime(agg["date"])
    agg["humidity_range"] = agg["humidity_max"] - agg["humidity_min"]
    agg["pressure_trend"] = agg["pressure_last"] - agg["pressure_first"]

    # Merge
    df = daily_df.merge(
        agg[["target_date_local", "humidity_mean", "humidity_range",
             "pressure_mean", "pressure_trend", "cloud_cover_mean", "dew_point_mean_f"]],
        on="target_date_local",
        how="left",
    )

    # Derived features
    df["diurnal_range_f"] = df["tmax_pred_f"] - df["tmin_f"]

    # Wind components (handle NaN wind_dir)
    wind_deg = df["wind_dir_dominant"].fillna(0).astype(float)
    wind_rad = np.deg2rad(wind_deg.values)
    df["wind_u"] = np.sin(wind_rad)
    df["wind_v"] = np.cos(wind_rad)

    if cache_dir:
        df.to_parquet(cache_path, index=False)

    return df


def fetch_enriched_in_chunks(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz: str = "America/New_York",
    chunk_days: int = 365,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch enriched data in yearly chunks to avoid API limits."""
    from datetime import datetime, timedelta

    ds = datetime.strptime(start_date, "%Y-%m-%d").date()
    de = datetime.strptime(end_date, "%Y-%m-%d").date()

    chunks = []
    cur = ds
    while cur <= de:
        chunk_end = min(de, cur + timedelta(days=chunk_days - 1))
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {s} to {e}...")
        chunk = fetch_enriched_historical(lat, lon, s, e, tz, cache_dir)
        chunks.append(chunk)
        cur = chunk_end + timedelta(days=1)
        if cur <= de:
            time.sleep(0.5)  # Be nice to the API

    return pd.concat(chunks, ignore_index=True)


def merge_enriched_features(
    train_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge enriched features into the existing training table.

    Preserves all existing columns and adds new feature columns.
    """
    # Normalize date types
    train = train_df.copy()
    enriched = enriched_df.copy()

    train["target_date_local"] = pd.to_datetime(train["target_date_local"]).dt.date
    enriched["target_date_local"] = pd.to_datetime(enriched["target_date_local"]).dt.date

    # Select enrichment columns (skip raw forecast/temp columns to avoid conflicts)
    enrich_cols = [
        "target_date_local",
        "tmin_f", "apparent_tmax_f", "diurnal_range_f",
        "wind_max_kmh", "wind_u", "wind_v",
        "precip_mm", "precip_hours",
        "radiation_mj",
        "humidity_mean", "humidity_range",
        "pressure_mean", "pressure_trend",
        "cloud_cover_mean", "dew_point_mean_f",
    ]
    enrich_cols = [c for c in enrich_cols if c in enriched.columns]
    enriched_subset = enriched[enrich_cols].drop_duplicates(subset=["target_date_local"])

    merged = train.merge(enriched_subset, on="target_date_local", how="left")

    # Fill NaN for new columns with 0 (missing data = neutral)
    new_cols = [c for c in enrich_cols if c != "target_date_local" and c not in train.columns]
    for col in new_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged.reset_index(drop=True)


def get_enriched_feature_names() -> list[str]:
    """Return the list of enriched feature column names for model training."""
    return [
        "tmin_f", "apparent_tmax_f", "diurnal_range_f",
        "wind_max_kmh", "wind_u", "wind_v",
        "precip_mm", "precip_hours",
        "radiation_mj",
        "humidity_mean", "humidity_range",
        "pressure_mean", "pressure_trend",
        "cloud_cover_mean", "dew_point_mean_f",
    ]
