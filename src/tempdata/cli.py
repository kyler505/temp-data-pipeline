"""tempdata package CLI.

This module provides a small set of orchestration commands intended to be the
*primary* entrypoints for running this codebase on local machines and on HPRC.

Commands:
  - tempdata data: fetch/clean/build canonical datasets (including features)
  - tempdata train: run experimentation / training (creates a run folder)
  - tempdata eval: run evaluation/reporting (creates a run folder)

Design goals:
  - Keep domain logic in tempdata.* modules; CLI should be thin wrappers.
  - Make it obvious which commands to run.
  - Allow Open-Meteo forecasts to be optional, with an opt-in strict mode.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from tempdata.aggregate.build_daily_tmax import build_daily_tmax, write_daily_tmax
from tempdata.clean.clean_hourly import clean_hourly_file
from tempdata.eval.config import (
    EvalConfig,
    ModelConfig,
    SplitConfig,
    UncertaintyConfig,
    generate_run_id,
)
from tempdata.eval.data import load_eval_inputs
from tempdata.eval.runner import run_evaluation, run_walk_forward_evaluation
from tempdata.features.build_train_daily_tmax import build_train_daily_tmax, write_train_daily_tmax
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly
from tempdata.fetch.openmeteo_daily_forecast import fetch_openmeteo_historical_forecasts


# -------------------------
# Helpers
# -------------------------

def _parse_date(s: str) -> str:
    # Keep as YYYY-MM-DD string for existing APIs.
    # argparse validates via datetime.strptime.
    datetime.strptime(s, "%Y-%m-%d")
    return s


def _date_range_days(start: str, end: str) -> int:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    return (de - ds).days + 1


def _split_into_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()

    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")

    chunks: list[tuple[str, str]] = []
    cur = ds
    while cur <= de:
        cur_end = min(de, cur + pd.Timedelta(days=chunk_days - 1).to_pytimedelta())
        chunks.append((cur.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
        cur = cur_end + pd.Timedelta(days=1).to_pytimedelta()
    return chunks


def _paths_for_station(data_dir: Path, station: str) -> dict[str, Path]:
    return {
        "raw_noaa_hourly": data_dir / "raw" / "noaa_hourly" / station,
        "cache_isd_csv": data_dir / "cache" / "isd_csv" / station,
        "clean_hourly": data_dir / "clean" / "hourly_obs" / station,
        "daily_tmax": data_dir / "clean" / "daily_tmax" / f"{station}.parquet",
        "forecast_dir": data_dir / "raw" / "daily_tmax_forecast" / station,
        "forecast_historical": data_dir
        / "raw"
        / "daily_tmax_forecast"
        / station
        / "historical.parquet",
        "train_daily_tmax": data_dir / "features" / "train_daily_tmax" / f"{station}.parquet",
    }


def _read_all_parquet(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in paths]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _ensure_forecast_coverage(
    forecast_df: pd.DataFrame,
    station: str,
    start: str,
    end: str,
    lead_hours: list[int] | None = None,
) -> None:
    """Strictly enforce that forecast_df covers [start, end] (and optional lead hours)."""
    if forecast_df is None or forecast_df.empty:
        raise ValueError(
            f"Missing forecast data for station={station}. "
            f"Required full coverage for {start}..{end}."
        )

    df = forecast_df.copy()
    # target_date_local is tz-naive timestamp at midnight
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()

    required_dates = pd.date_range(ds, de, freq="D").date

    if lead_hours is not None:
        df = df[df["lead_hours"].isin(lead_hours)]

    present_dates = set(df["target_date_local"].unique().tolist())
    missing = [d for d in required_dates if d not in present_dates]

    if missing:
        # Show a small summary for usability
        first = missing[0]
        last = missing[-1]
        raise ValueError(
            "Forecast coverage check failed (strict mode). "
            f"station={station} missing {len(missing)} day(s) between {start}..{end}. "
            f"Example missing window: {first}..{last}. "
            "Try adjusting --start/--end to a forecast-supported window or re-fetch forecasts."
        )


# -------------------------
# Subcommands
# -------------------------

def cmd_data(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    station = args.station
    start = args.start
    end = args.end

    paths = _paths_for_station(data_dir, station)

    # 1) Fetch NOAA hourly (raw)
    print(f"[tempdata:data] fetching NOAA hourly station={station} {start}..{end}")
    written_raw = fetch_noaa_hourly(
        station_id=station,
        start_date=start,
        end_date=end,
        out_dir=paths["raw_noaa_hourly"],
        cache_dir=paths["cache_isd_csv"],
    )
    print(f"[tempdata:data] fetched {len(written_raw)} raw parquet file(s)")

    # 2) Clean hourly
    print(f"[tempdata:data] cleaning hourly station={station}")
    paths["clean_hourly"].mkdir(parents=True, exist_ok=True)
    cleaned_paths: list[Path] = []
    for raw_path in written_raw:
        clean_path = paths["clean_hourly"] / raw_path.name
        clean_hourly_file(raw_path, clean_path)
        cleaned_paths.append(clean_path)
    print(f"[tempdata:data] cleaned {len(cleaned_paths)} file(s)")

    # 3) Build daily truth (daily_tmax)
    # We aggregate from *all* cleaned hourly files present for the station, not just newly written ones.
    hourly_files = sorted(paths["clean_hourly"].glob("*.parquet"))
    if not hourly_files:
        raise ValueError(
            f"No cleaned hourly parquet files found at {paths['clean_hourly']}. "
            "Cannot build daily_tmax."
        )

    hourly_df = _read_all_parquet(hourly_files)
    if hourly_df.empty:
        raise ValueError("Cleaned hourly dataframe is empty; cannot build daily_tmax")

    daily_df = build_daily_tmax(
        hourly_df,
        station_tz=args.timezone,
        min_coverage_hours=args.min_coverage,
    )

    # Filter to requested window (strictly for output convenience)
    daily_df["date_local"] = pd.to_datetime(daily_df["date_local"]).dt.date
    ds = datetime.strptime(start, "%Y-%m-%d").date()
    de = datetime.strptime(end, "%Y-%m-%d").date()
    daily_df = daily_df[(daily_df["date_local"] >= ds) & (daily_df["date_local"] <= de)].reset_index(drop=True)

    write_daily_tmax(daily_df, paths["daily_tmax"])

    # 4) Fetch Open-Meteo historical forecasts for the same window (optional)
    #
    # Behavior:
    #  - If forecasts are available for the requested window, we fetch/cache them and build
    #    the training feature table.
    #  - If forecasts are missing/unavailable, we do NOT fail `tempdata data`; instead we
    #    skip forecast-dependent artifacts.
    #
    print(f"[tempdata:data] fetching Open-Meteo historical forecasts station={station} {start}..{end} (optional)")

    forecast_df: pd.DataFrame | None = None
    try:
        paths["forecast_dir"].mkdir(parents=True, exist_ok=True)
        written_fcst, forecast_df = fetch_openmeteo_historical_forecasts(
            station_id=station,
            start_date=start,
            end_date=end,
            out_raw_dir=None,
            out_parquet_dir=paths["forecast_dir"],
            write_raw=args.write_raw_forecast,
        )
        print(f"[tempdata:data] fetched forecast files: {len(written_fcst)}")

        # Consolidate into one canonical parquet for the station (overwrites)
        write_path = paths["forecast_historical"]
        write_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = write_path.with_suffix(".parquet.tmp")
        forecast_df.to_parquet(tmp, index=False)
        tmp.rename(write_path)
        print(f"[tempdata:data] wrote consolidated forecast parquet to {write_path}")

        # Optional coverage check: only enforce strict coverage if requested.
        lead_hours = None
        if args.lead_hours:
            lead_hours = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]
        if args.require_forecast:
            _ensure_forecast_coverage(forecast_df, station, start, end, lead_hours=lead_hours)
            print("[tempdata:data] forecast coverage OK (strict)")
        else:
            # If not strict, just log what we got.
            print("[tempdata:data] forecast fetched; strict coverage not required")

    except Exception as e:
        if args.require_forecast:
            raise
        print(
            "[tempdata:data] WARNING: forecast fetch/coverage failed; continuing without forecasts. "
            f"Reason: {e}"
        )
        forecast_df = None

    # 5) Build training feature table (only if forecasts exist)
    if forecast_df is not None and not forecast_df.empty:
        print("[tempdata:data] building training feature table train_daily_tmax")
        truth_df = pd.read_parquet(paths["daily_tmax"])
        train_df = build_train_daily_tmax(
            forecast_df=forecast_df,
            truth_df=truth_df,
            min_coverage_hours=args.min_coverage,
            drop_warmup_nulls=args.drop_warmup_nulls,
            validate=True,
        )
        # Optional: filter to requested window (target_date_local is the label date)
        train_df["target_date_local"] = pd.to_datetime(train_df["target_date_local"]).dt.date
        train_df = train_df[(train_df["target_date_local"] >= ds) & (train_df["target_date_local"] <= de)].reset_index(drop=True)

        write_train_daily_tmax(train_df, paths["train_daily_tmax"], validate=True)
    else:
        print("[tempdata:data] skipping train_daily_tmax: no forecast data available")

    print("[tempdata:data] done")


def cmd_train(args: argparse.Namespace) -> None:
    # Today this codebase's "evaluation" runner already encapsulates training/experimentation.
    # We expose it under `train` for a clearer workflow.
    _run_eval_like(args, mode="train")


def cmd_eval(args: argparse.Namespace) -> None:
    _run_eval_like(args, mode="eval")


def _run_eval_like(args: argparse.Namespace, mode: str) -> None:
    """Run evaluation pipeline (used for both train and eval orchestration).

    The underlying runner produces model artifacts and metrics. This wrapper primarily
    exists to provide stable CLI entrypoints.
    """

    # Reuse the existing config behavior from scripts/eval_daily_tmax.py
    if args.config is not None:
        cfg = EvalConfig.load(args.config)
    else:
        if not (args.station and args.start and args.end):
            raise SystemExit("Must provide --config or (--station, --start, --end)")

        lead_hours = None
        if args.lead_hours:
            lead_hours = [int(x.strip()) for x in args.lead_hours.split(",") if x.strip()]

        split_type = "static"
        if args.split_method == "walk_forward":
            split_type = "walk_forward"
        elif args.split_method not in {"time", "static"}:
            raise SystemExit(
                f"Unsupported split method '{args.split_method}'. Use time/static or walk_forward."
            )

        cfg = EvalConfig(
            run_name=f"{mode}_{args.station}",
            station_ids=[args.station],
            start_date_local=date.fromisoformat(args.start),
            end_date_local=date.fromisoformat(args.end),
            lead_hours_allowed=lead_hours,
            min_coverage_hours=args.min_coverage,
            split=SplitConfig(
                type=split_type,
                train_frac=args.train_ratio,
                test_frac=args.test_ratio,
                val_frac=args.val_ratio,
            ),
            model=ModelConfig(
                type=args.model_type,
                alpha=args.model_alpha,
            ),
            uncertainty=UncertaintyConfig(
                type=args.sigma_type,
            ),
            random_seed=args.seed,
        )

    # Run ID behavior: keep current behavior (generate new run every time)
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = generate_run_id()

    forecast_df, truth_df, feature_df = load_eval_inputs(
        cfg,
        data_dir=args.data_dir,
        forecast_file=args.forecast_file,
        truth_file=args.truth_file,
        feature_file=args.feature_file,
    )

    print(f"[tempdata:{mode}] run_id={run_id}")

    if cfg.split.type == "walk_forward":
        run_walk_forward_evaluation(
            cfg,
            forecast_df=forecast_df,
            truth_df=truth_df,
            feature_df=feature_df,
            run_id=run_id,
        )
    else:
        run_evaluation(
            cfg,
            forecast_df=forecast_df,
            truth_df=truth_df,
            feature_df=feature_df,
            run_id=run_id,
        )


def cmd_tune(args: argparse.Namespace) -> None:
    """Run hyperparameter tuning for stacked ensemble."""
    from tempdata.tune import main as tune_main
    import sys
    sys.argv = ["tempdata.tune"]
    if args.data_path:
        sys.argv += ["--data-path", str(args.data_path)]
    if args.n_folds:
        sys.argv += ["--n-folds", str(args.n_folds)]
    if args.quick:
        sys.argv += ["--quick"]
    if args.max_trials:
        sys.argv += ["--max-trials", str(args.max_trials)]
    if args.output_dir:
        sys.argv += ["--output-dir", str(args.output_dir)]
    tune_main()


def cmd_ablate(args: argparse.Namespace) -> None:
    """Run feature ablation study."""
    from tempdata.ablate import main as ablate_main
    import sys
    sys.argv = ["tempdata.ablate"]
    if args.data_path:
        sys.argv += ["--data-path", str(args.data_path)]
    if args.model:
        sys.argv += ["--model", args.model]
    if args.strategy:
        sys.argv += ["--strategy", args.strategy]
    if args.train_frac:
        sys.argv += ["--train-frac", str(args.train_frac)]
    if args.quick:
        sys.argv += ["--quick"]
    if args.output_dir:
        sys.argv += ["--output-dir", str(args.output_dir)]
    ablate_main()


# -------------------------
# Parser
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tempdata", description="temp-data-pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # data
    p_data = sub.add_parser("data", help="Fetch/clean/build canonical datasets")
    p_data.add_argument("--station", required=True)
    p_data.add_argument("--start", required=True, type=_parse_date)
    p_data.add_argument("--end", required=True, type=_parse_date)
    p_data.add_argument("--timezone", required=True, help="e.g. America/New_York")
    p_data.add_argument("--min-coverage", type=int, default=18)
    p_data.add_argument("--lead-hours", default=None, help="Comma-separated lead hours to require")
    p_data.add_argument("--drop-warmup-nulls", action="store_true", help="Drop warm-up rows with NaNs in rolling features")
    p_data.add_argument("--data-dir", default="data")
    p_data.add_argument("--write-raw-forecast", action="store_true", help="Also write raw Open-Meteo JSON")
    p_data.add_argument(
        "--require-forecast",
        action="store_true",
        help="Require full Open-Meteo forecast coverage for the requested date range (strict).",
    )

    p_data.set_defaults(func=cmd_data)

    # train
    p_train = sub.add_parser("train", help="Run training/experimentation (creates a run folder)")
    _add_eval_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation/reporting (creates a run folder)")
    _add_eval_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    # tune
    p_tune = sub.add_parser("tune", help="Tune stacked ensemble hyperparameters")
    p_tune.add_argument("--data-path", type=Path, default=Path("data/train/daily_tmax/KLGA/train_daily_tmax_enriched.parquet"))
    p_tune.add_argument("--n-folds", type=int, default=5)
    p_tune.add_argument("--quick", action="store_true", help="Use quick (small) grid")
    p_tune.add_argument("--max-trials", type=int, default=None)
    p_tune.add_argument("--output-dir", type=Path, default=Path("runs/tune"))
    p_tune.set_defaults(func=cmd_tune)

    # ablate
    p_ablate = sub.add_parser("ablate", help="Feature ablation study")
    p_ablate.add_argument("--data-path", type=Path, default=Path("data/train/daily_tmax/KLGA/train_daily_tmax_enriched.parquet"))
    p_ablate.add_argument("--model", type=str, default="ridge", help="Model type to ablate")
    p_ablate.add_argument("--strategy", choices=["loo", "loog", "forward", "all"], default="all")
    p_ablate.add_argument("--train-frac", type=float, default=0.8)
    p_ablate.add_argument("--quick", action="store_true", help="Skip forward selection (slow)")
    p_ablate.add_argument("--output-dir", type=Path, default=Path("runs/ablate"))
    p_ablate.set_defaults(func=cmd_ablate)

    return parser


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help="Path to JSON config file")
    p.add_argument("--run-id", default=None, help="Optional run id; if omitted a new one is generated")

    p.add_argument("--station", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--data-dir", default="data")

    p.add_argument("--forecast-file", type=str, default=None)
    p.add_argument("--truth-file", type=str, default=None)
    p.add_argument("--feature-file", type=str, default=None)
    p.add_argument("--lead-hours", default=None)
    p.add_argument("--min-coverage", type=int, default=18)

    p.add_argument("--split-method", default="time", choices=["time", "static", "walk_forward"])
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-type", choices=["ridge", "passthrough", "persistence", "knn", "xgboost", "lightgbm", "catboost", "stacked"], default="ridge")
    p.add_argument("--model-alpha", type=float, default=1.0)
    p.add_argument("--model-k", type=int, default=5)

    p.add_argument("--sigma-type", choices=["global", "bucketed", "rolling"], default="bucketed")
    p.add_argument("--sigma-buckets", type=int, default=10)
    p.add_argument("--sigma-window-days", type=int, default=30)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
