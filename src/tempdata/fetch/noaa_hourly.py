"""Fetch NOAA hourly data from public ISD CSVs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from tempdata.config import raw_isd_csv_dir, raw_noaa_hourly_dir, stations_csv_path
from tempdata.schemas.hourly_obs import RAW_HOURLY_FIELDS, ensure_hourly_schema_columns

BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"


@dataclass(frozen=True)
class StationMeta:
    station_id: str
    usaf: str
    wban: str
    name: str
    lat: float
    lon: float
    tz: str

    @property
    def isd_key(self) -> str:
        return f"{self.usaf.zfill(6)}-{self.wban.zfill(5)}"


def _to_datetime(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=timezone.utc)


def _year_range(start: datetime, end: datetime) -> Iterable[int]:
    end_inclusive = end - timedelta(days=1)
    for year in range(start.year, end_inclusive.year + 1):
        yield year


def load_station_mapping(path: Path | None = None) -> dict[str, StationMeta]:
    mapping_path = path or stations_csv_path()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Station mapping file not found: {mapping_path}")
    mapping: dict[str, StationMeta] = {}
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station_id = row["station_id"].strip().upper()
            mapping[station_id] = StationMeta(
                station_id=station_id,
                usaf=row["usaf"].strip(),
                wban=row["wban"].strip(),
                name=row.get("name", "").strip(),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                tz=row.get("tz", "").strip(),
            )
    return mapping


def resolve_station(station_id: str, path: Path | None = None) -> StationMeta:
    mapping = load_station_mapping(path)
    key = station_id.strip().upper()
    if key not in mapping:
        raise KeyError(f"Station {station_id} not found in {path or stations_csv_path()}")
    return mapping[key]


def isd_url(usaf: str, wban: str, year: int) -> str:
    station_key = f"{usaf.zfill(6)}-{wban.zfill(5)}"
    return f"{BASE_URL}/{year}/{station_key}.csv"


def download_csv(
    url: str,
    out_path: Path,
    force: bool = False,
    use_cache: bool = True,
) -> Path:
    if use_cache and out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with out_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return out_path


def _safe_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_temp(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
    else:
        part = value.split(",", maxsplit=1)[0].strip()
        if part == "":
            return None
        try:
            raw = float(part)
        except ValueError:
            return None
    if abs(raw) >= 9990:
        return None
    if abs(raw) > 200:
        raw = raw / 10.0
    return raw


def _parse_csv(csv_path: Path, station: StationMeta) -> pd.DataFrame:
    desired = {"DATE", "TMP", "LATITUDE", "LONGITUDE"}
    df = pd.read_csv(
        csv_path,
        usecols=lambda col: col in desired,
        dtype=str,
        low_memory=False,
    )
    missing = {"DATE", "TMP"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    ts_utc = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    temp_c = df["TMP"].map(_parse_temp)

    if "LATITUDE" in df.columns:
        lat = df["LATITUDE"].map(_safe_float).fillna(station.lat)
    else:
        lat = station.lat
    if "LONGITUDE" in df.columns:
        lon = df["LONGITUDE"].map(_safe_float).fillna(station.lon)
    else:
        lon = station.lon

    out = pd.DataFrame(
        {
            "ts_utc": ts_utc,
            "station_id": station.station_id,
            "lat": lat,
            "lon": lon,
            "temp_c": temp_c,
            "source": "noaa",
            "qc_flags": 0,
        },
        columns=RAW_HOURLY_FIELDS,
    )
    out = out.dropna(subset=["ts_utc"]).copy()
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")
    return out


def _validate_hourly(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"[noaa] {label}: 0 rows")
        return
    if not df["ts_utc"].is_monotonic_increasing:
        print(f"[noaa] {label}: timestamps not sorted, sorting")
        df.sort_values("ts_utc", inplace=True)
    if df["temp_c"].notna().sum() == 0:
        print(f"[noaa] {label}: temp_c all null")
    temp_min = df["temp_c"].min()
    temp_max = df["temp_c"].max()
    if temp_min < -100 or temp_max > 60:
        print(f"[noaa] {label}: temp out of bounds (min={temp_min}, max={temp_max})")

    min_ts = df["ts_utc"].min()
    max_ts = df["ts_utc"].max()
    if pd.notna(min_ts) and pd.notna(max_ts):
        expected = int((max_ts - min_ts).total_seconds() / 3600) + 1
        actual = int(df["ts_utc"].nunique())
        if expected > 0:
            missing_pct = max(0.0, (expected - actual) / expected)
            if missing_pct > 0.05:
                print(
                    f"[noaa] {label}: missing {missing_pct:.1%} hours "
                    f"({actual}/{expected})"
                )
    print(f"[noaa] {label}: rows={len(df)} coverage={min_ts} -> {max_ts}")


def fetch_noaa_hourly(
    station_id: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    use_cache: bool = True,
) -> list[Path]:
    station = resolve_station(station_id)
    start_dt = _to_datetime(start_date)
    end_dt = _to_datetime(end_date)
    if end_dt <= start_dt:
        raise ValueError("end_date must be after start_date")

    cache_root = Path(cache_dir) if cache_dir else raw_isd_csv_dir(station.station_id)
    output_root = Path(out_dir) if out_dir else raw_noaa_hourly_dir(station.station_id)
    output_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for year in _year_range(start_dt, end_dt):
        url = isd_url(station.usaf, station.wban, year)
        csv_path = cache_root / f"{year}.csv"
        if use_cache:
            download_csv(url, csv_path, force=force, use_cache=True)
        else:
            download_csv(url, csv_path, force=True, use_cache=False)

        df = _parse_csv(csv_path, station)
        df = df[(df["ts_utc"] >= start_dt) & (df["ts_utc"] < end_dt)].copy()
        if not df.empty:
            df.sort_values("ts_utc", inplace=True)
            df = df[ensure_hourly_schema_columns(df.columns)]
        else:
            df = pd.DataFrame(columns=RAW_HOURLY_FIELDS)
        _validate_hourly(df, label=str(year))

        parquet_path = output_root / f"{year}.parquet"
        df.to_parquet(parquet_path, index=False)
        written.append(parquet_path)

        if not use_cache and csv_path.exists():
            csv_path.unlink()

    return written
