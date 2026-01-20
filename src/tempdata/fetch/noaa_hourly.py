"""Fetch NOAA hourly data from public ISD or GHCNh sources."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import requests

from tempdata.config import raw_isd_csv_dir, raw_noaa_hourly_dir, stations_csv_path
from tempdata.schemas.hourly_obs import (
    RAW_HOURLY_FIELDS,
    ensure_hourly_schema_columns,
    validate_hourly_obs,
)

# Base URLs for data sources
ISD_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"
GHCNH_BASE_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-hourly/access"

# Transition date: ISD discontinued after this date
ISD_CUTOFF_DATE = date(2025, 8, 29)


@dataclass(frozen=True)
class StationMeta:
    station_id: str
    usaf: str
    wban: str
    ghcn_id: str  # GHCNh station identifier (e.g., USW00014732)
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
                ghcn_id=row.get("ghcn_id", "").strip(),
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
    station_key = f"{usaf}{wban}"
    return f"{ISD_BASE_URL}/{year}/{station_key}.csv"


def ghcnh_url(ghcn_id: str) -> str:
    """Generate URL for GHCNh PSV file.

    GHCNh provides a single consolidated PSV file per station containing all years.
    """
    return f"{GHCNH_BASE_URL}/{ghcn_id}.psv"


def download_file(
    url: str,
    out_path: Path,
    force: bool = False,
    use_cache: bool = True,
) -> Path:
    if use_cache and out_path.exists() and not force:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
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


def _parse_isd_temp(value: str | float | None) -> float | None:
    """Parse ISD temperature field to Celsius.

    ISD stores temperatures in tenths of degrees Celsius (e.g., "150" = 15.0°C).
    Missing values are encoded as 9999 or +9999.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
    else:
        # ISD format: "+0150,1" where first part is temp in tenths C
        part = value.split(",", maxsplit=1)[0].strip()
        if part == "":
            return None
        try:
            raw = float(part)
        except ValueError:
            return None

    # Missing value indicator
    if abs(raw) >= 9990:
        return None

    # ISD temperatures are always in tenths of degrees Celsius
    # Convert to actual Celsius
    raw = raw / 10.0

    return raw


def _parse_ghcnh_temp(value: str | float | None) -> float | None:
    """Parse GHCNh temperature field to Celsius.

    GHCNh stores temperatures in tenths of degrees Celsius.
    Missing values are encoded as empty string or -9999.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
    else:
        stripped = value.strip()
        if stripped == "" or stripped == "-9999":
            return None
        try:
            raw = float(stripped)
        except ValueError:
            return None

    # Missing value indicator
    if raw <= -9990 or raw >= 9990:
        return None

    # GHCNh temperatures are in tenths of degrees Celsius
    return raw / 10.0


def _parse_isd_csv(csv_path: Path, station: StationMeta) -> pd.DataFrame:
    """Parse ISD CSV format."""
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
    temp_c = df["TMP"].map(_parse_isd_temp)

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
            "source": "isd",
            "qc_flags": 0,
        },
        columns=RAW_HOURLY_FIELDS,
    )
    out = out.dropna(subset=["ts_utc"]).copy()
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")
    return out


def _parse_ghcnh_psv(psv_path: Path, station: StationMeta) -> pd.DataFrame:
    """Parse GHCNh Pipe-Separated Values (PSV) format.

    GHCNh PSV format has columns including:
    - Station_ID, Date, Time (or combined datetime), temperature columns

    The exact column names may vary; we look for common patterns.
    """
    # GHCNh uses pipe separator
    df = pd.read_csv(
        psv_path,
        sep="|",
        dtype=str,
        low_memory=False,
    )

    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.strip().str.lower()

    # Try to find datetime column
    datetime_col = None
    for col in ["datetime", "date_time", "observation_time"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        # Try date + hour combination
        if "date" in df.columns and "hour" in df.columns:
            df["_datetime"] = df["date"].astype(str) + " " + df["hour"].astype(str).str.zfill(2) + ":00"
            datetime_col = "_datetime"
        elif "date" in df.columns:
            datetime_col = "date"
        else:
            raise ValueError(f"Cannot find datetime column in GHCNh PSV: {list(df.columns)}")

    ts_utc = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")

    # Try to find temperature column
    temp_col = None
    for col in ["temperature", "temp", "air_temperature", "t"]:
        if col in df.columns:
            temp_col = col
            break

    if temp_col is None:
        raise ValueError(f"Cannot find temperature column in GHCNh PSV: {list(df.columns)}")

    temp_c = df[temp_col].map(_parse_ghcnh_temp)

    # Lat/lon from file or station metadata
    lat = station.lat
    lon = station.lon
    if "latitude" in df.columns:
        lat = df["latitude"].map(_safe_float).fillna(station.lat)
    if "longitude" in df.columns:
        lon = df["longitude"].map(_safe_float).fillna(station.lon)

    out = pd.DataFrame(
        {
            "ts_utc": ts_utc,
            "station_id": station.station_id,
            "lat": lat,
            "lon": lon,
            "temp_c": temp_c,
            "source": "ghcnh",
            "qc_flags": 0,
        },
        columns=RAW_HOURLY_FIELDS,
    )
    out = out.dropna(subset=["ts_utc"]).copy()
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce")
    return out


def _log_coverage(df: pd.DataFrame, label: str, source: str = "noaa") -> None:
    """Log coverage information for debugging (non-fatal warnings)."""
    if df.empty:
        print(f"[{source}] {label}: 0 rows")
        return

    if df["temp_c"].notna().sum() == 0:
        print(f"[{source}] {label}: temp_c all null")

    min_ts = df["ts_utc"].min()
    max_ts = df["ts_utc"].max()
    if pd.notna(min_ts) and pd.notna(max_ts):
        expected = int((max_ts - min_ts).total_seconds() / 3600) + 1
        actual = int(df["ts_utc"].nunique())
        if expected > 0:
            missing_pct = max(0.0, (expected - actual) / expected)
            if missing_pct > 0.05:
                print(
                    f"[{source}] {label}: missing {missing_pct:.1%} hours "
                    f"({actual}/{expected})"
                )
    print(f"[{source}] {label}: rows={len(df)} coverage={min_ts} -> {max_ts}")


def fetch_noaa_hourly(
    station_id: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    source: Literal["isd", "ghcnh", "auto"] = "auto",
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
    use_cache: bool = True,
) -> list[Path]:
    """Fetch NOAA hourly observation data.

    Args:
        station_id: Station identifier (e.g., "KLGA")
        start_date: Start date (inclusive)
        end_date: End date (exclusive)
        source: Data source - "isd" for legacy ISD, "ghcnh" for GHCNh,
                "auto" to select based on date (ISD before 2025-08-29, GHCNh after)
        out_dir: Output directory for parquet files
        cache_dir: Cache directory for downloaded files
        force: Force re-download even if cached
        use_cache: Whether to use cached files

    Returns:
        List of written parquet file paths
    """
    station = resolve_station(station_id)
    start_dt = _to_datetime(start_date)
    end_dt = _to_datetime(end_date)
    if end_dt <= start_dt:
        raise ValueError("end_date must be after start_date")

    output_root = Path(out_dir) if out_dir else raw_noaa_hourly_dir(station.station_id)
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine effective source based on date range
    if source == "auto":
        cutoff_dt = datetime(ISD_CUTOFF_DATE.year, ISD_CUTOFF_DATE.month, ISD_CUTOFF_DATE.day, tzinfo=timezone.utc)
        if end_dt <= cutoff_dt:
            effective_source = "isd"
        elif start_dt >= cutoff_dt:
            effective_source = "ghcnh"
        else:
            # Hybrid: need both sources
            return _fetch_hybrid(station, start_dt, end_dt, cutoff_dt, output_root, cache_dir, force, use_cache)
    else:
        effective_source = source

    if effective_source == "isd":
        return _fetch_isd(station, start_dt, end_dt, output_root, cache_dir, force, use_cache)
    else:
        return _fetch_ghcnh(station, start_dt, end_dt, output_root, cache_dir, force, use_cache)


def _fetch_isd(
    station: StationMeta,
    start_dt: datetime,
    end_dt: datetime,
    output_root: Path,
    cache_dir: str | Path | None,
    force: bool,
    use_cache: bool,
) -> list[Path]:
    """Fetch from ISD source."""
    cache_root = Path(cache_dir) if cache_dir else raw_isd_csv_dir(station.station_id)

    written: list[Path] = []
    for year in _year_range(start_dt, end_dt):
        url = isd_url(station.usaf, station.wban, year)
        csv_path = cache_root / f"{year}.csv"
        if use_cache:
            download_file(url, csv_path, force=force, use_cache=True)
        else:
            download_file(url, csv_path, force=True, use_cache=False)

        try:
            df = _parse_isd_csv(csv_path, station)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"[isd] {year}: corrupted cache file found, deleting and retrying -> {csv_path}")
            if csv_path.exists():
                csv_path.unlink()

            download_file(url, csv_path, force=True, use_cache=False)
            df = _parse_isd_csv(csv_path, station)

        df = df[(df["ts_utc"] >= start_dt) & (df["ts_utc"] < end_dt)].copy()
        if not df.empty:
            df.sort_values("ts_utc", inplace=True)
            df = df[ensure_hourly_schema_columns(df.columns)]
        else:
            df = pd.DataFrame(columns=RAW_HOURLY_FIELDS)

        _log_coverage(df, label=str(year), source="isd")
        validate_hourly_obs(df, require_unique_keys=False)

        parquet_path = output_root / f"isd_{year}.parquet"
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(parquet_path)
        written.append(parquet_path)

        if not use_cache and csv_path.exists():
            csv_path.unlink()

    return written


def _fetch_ghcnh(
    station: StationMeta,
    start_dt: datetime,
    end_dt: datetime,
    output_root: Path,
    cache_dir: str | Path | None,
    force: bool,
    use_cache: bool,
) -> list[Path]:
    """Fetch from GHCNh source."""
    if not station.ghcn_id:
        raise ValueError(f"Station {station.station_id} does not have a GHCN ID configured")

    cache_root = Path(cache_dir) if cache_dir else raw_isd_csv_dir(station.station_id) / "ghcnh"
    cache_root.mkdir(parents=True, exist_ok=True)

    url = ghcnh_url(station.ghcn_id)
    psv_path = cache_root / f"{station.ghcn_id}.psv"

    if use_cache:
        download_file(url, psv_path, force=force, use_cache=True)
    else:
        download_file(url, psv_path, force=True, use_cache=False)

    try:
        df = _parse_ghcnh_psv(psv_path, station)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"[ghcnh] corrupted cache file, retrying -> {psv_path}")
        if psv_path.exists():
            psv_path.unlink()
        download_file(url, psv_path, force=True, use_cache=False)
        df = _parse_ghcnh_psv(psv_path, station)

    df = df[(df["ts_utc"] >= start_dt) & (df["ts_utc"] < end_dt)].copy()
    if not df.empty:
        df.sort_values("ts_utc", inplace=True)
        df = df[ensure_hourly_schema_columns(df.columns)]
    else:
        df = pd.DataFrame(columns=RAW_HOURLY_FIELDS)

    _log_coverage(df, label=f"{start_dt.year}-{end_dt.year}", source="ghcnh")
    validate_hourly_obs(df, require_unique_keys=False)

    # Write per-year parquets for consistency
    written: list[Path] = []
    for year in _year_range(start_dt, end_dt):
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        year_df = df[(df["ts_utc"] >= year_start) & (df["ts_utc"] < year_end)]
        if year_df.empty:
            year_df = pd.DataFrame(columns=RAW_HOURLY_FIELDS)

        parquet_path = output_root / f"ghcnh_{year}.parquet"
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        year_df.to_parquet(tmp_path, index=False)
        tmp_path.rename(parquet_path)
        written.append(parquet_path)

    if not use_cache and psv_path.exists():
        psv_path.unlink()

    return written


def _fetch_hybrid(
    station: StationMeta,
    start_dt: datetime,
    end_dt: datetime,
    cutoff_dt: datetime,
    output_root: Path,
    cache_dir: str | Path | None,
    force: bool,
    use_cache: bool,
) -> list[Path]:
    """Fetch from both ISD (before cutoff) and GHCNh (after cutoff)."""
    written: list[Path] = []

    # Fetch ISD portion
    if start_dt < cutoff_dt:
        isd_end = min(end_dt, cutoff_dt)
        written.extend(_fetch_isd(station, start_dt, isd_end, output_root, cache_dir, force, use_cache))

    # Fetch GHCNh portion
    if end_dt > cutoff_dt:
        ghcnh_start = max(start_dt, cutoff_dt)
        written.extend(_fetch_ghcnh(station, ghcnh_start, end_dt, output_root, cache_dir, force, use_cache))

    return written
