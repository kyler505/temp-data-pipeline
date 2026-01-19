"""Configuration settings for the temp data pipeline."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return project_root() / "data"


def raw_isd_csv_dir(station_id: str) -> Path:
    return data_root() / "raw" / "isd_csv" / station_id


def raw_noaa_hourly_dir(station_id: str) -> Path:
    return data_root() / "raw" / "noaa_hourly" / station_id


def stations_csv_path() -> Path:
    return project_root() / "stations" / "stations.csv"


def raw_openmeteo_forecast_dir(station_id: str) -> Path:
    return data_root() / "raw" / "forecasts" / "openmeteo" / station_id


def clean_openmeteo_forecast_dir(station_id: str) -> Path:
    return data_root() / "clean" / "forecasts" / "openmeteo" / station_id
