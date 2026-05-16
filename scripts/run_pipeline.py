#!/usr/bin/env python3
"""Master pipeline orchestrator.

Runs the full daily workflow:
  1. live_forecast — generate predictions
  2. verify_pipeline — validate output integrity
  3. compute_edges — compute betting edges with empirical sigma
  4. paper_trade — settle old trades, place new ones with Kelly sizing
  5. daily_summary — generate human-readable report

Exits non-zero on any critical failure so cron can alert.

Usage:
    python scripts/run_pipeline.py [--station KNYC] [--mock] [--no-edge]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

# Circuit breaker thresholds
MAX_SIGMA = 8.0  # Empirical sigma can be 3-7°F for longer horizons
MAX_DAILY_BETS = 5  # Max bets to recommend per day
DRAWDOWN_LIMIT = -20.0  # Cumulative P&L shutdown threshold


def run_step(cmd: list[str], description: str, cwd: Path) -> bool:
    """Run a subprocess step, print output, return success."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[pipeline] ❌ {description} failed (exit {result.returncode})")
        return False
    print(f"[pipeline] ✅ {description} complete")
    return True


def check_circuit_breakers(log_dir: Path, station_id: str) -> tuple[bool, list[str]]:
    """Check circuit breakers. Returns (allowed, list of warnings)."""
    warnings = []
    log_path = log_dir / "predictions.jsonl"
    if not log_path.exists():
        return True, warnings
    lines = log_path.read_text().strip().split("\n")
    today_preds = []
    for line in lines:
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
        except json.JSONDecodeError:
            continue
        if pred.get("station_id") != station_id:
            continue
        if pred.get("date", "").startswith(date.today().isoformat()):
            today_preds.append(pred)
    for p in today_preds:
        horizon = p.get("horizon_days", 1)
        recent_mae = p.get("recent_mae", 1.5)
        if horizon <= 2:
            sigma = max(recent_mae, 0.8)
        else:
            sigma = recent_mae * (1.0 + 0.15 * (horizon - 2))
        if sigma > MAX_SIGMA:
            warnings.append(f"Sigma {sigma:.2f}°F > {MAX_SIGMA}°F for h={horizon}d — high uncertainty")
    pnl_path = log_dir / "paper_pnl.json"
    if pnl_path.exists():
        try:
            with open(pnl_path) as f:
                pnl_data = json.load(f)
            cum_pnl = pnl_data.get("cumulative_pnl", 0)
            if cum_pnl <= DRAWDOWN_LIMIT:
                warnings.append(f"Cumulative P&L ${cum_pnl:+.2f} <= ${DRAWDOWN_LIMIT} — shutdown triggered")
                return False, warnings
        except Exception:
            pass
    return True, warnings


def main():
    parser = argparse.ArgumentParser(description="Daily pipeline orchestrator")
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    parser.add_argument("--mock", action="store_true", help="Use mock Kalshi data")
    parser.add_argument("--no-edge", action="store_true", help="Skip edge calculation")
    parser.add_argument("--no-paper", action="store_true", help="Skip paper trading")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip forecast (use existing predictions)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    station_id = args.station
    log_dir = args.log_dir
    data_dir = args.data_dir

    print(f"[pipeline] Starting daily pipeline for {station_id}")
    print(f"[pipeline] Date: {date.today().isoformat()}")

    allowed, warnings = check_circuit_breakers(log_dir, station_id)
    for w in warnings:
        print(f"[pipeline] ⚠️ CIRCUIT BREAKER: {w}")
    if not allowed:
        print("[pipeline] 🛑 Pipeline halted by circuit breaker")
        sys.exit(1)

    success = True

    # Step 1: Live forecast
    if not args.skip_forecast:
        success = run_step(
            [sys.executable, "scripts/live_forecast.py", "--station", station_id, "--data-dir", str(data_dir), "--log-dir", str(log_dir)],
            "Live Forecast",
            repo_root,
        )
        if not success:
            sys.exit(1)

    # Step 2: Verify pipeline integrity
    success = run_step(
        [sys.executable, "scripts/verify_pipeline.py", "--station", station_id, "--log-dir", str(log_dir)],
        "Pipeline Verification",
        repo_root,
    )
    if not success:
        sys.exit(1)

    # Step 3: Compute edge opportunities (writes to opportunities.jsonl)
    if not args.no_edge:
        edge_cmd = [
            sys.executable, "scripts/compute_edges.py",
            "--station", station_id,
            "--log-dir", str(log_dir),
            "--min-edge", "0.05",
        ]
        if args.mock:
            edge_cmd.append("--mock")
        success = run_step(edge_cmd, "Edge Computation", repo_root)
        if not success:
            print("[pipeline] ⚠️ Edge computation failed, continuing...")

    # Step 4: Paper trading (reads from opportunities.jsonl)
    if not args.no_paper:
        settle_cmd = [
            sys.executable, "scripts/paper_trade.py",
            "--station", station_id,
            "--log-dir", str(log_dir),
            "--settle",
        ]
        run_step(settle_cmd, "Paper Trade Settlement", repo_root)
        paper_cmd = [
            sys.executable, "scripts/paper_trade.py",
            "--station", station_id,
            "--log-dir", str(log_dir),
        ]
        success = run_step(paper_cmd, "Paper Trading", repo_root)
        if not success:
            print("[pipeline] ⚠️ Paper trading failed, continuing...")

    # Step 5: Daily summary
    success = run_step(
        [sys.executable, "scripts/daily_summary.py", "--station", station_id, "--log-dir", str(log_dir)],
        "Daily Summary",
        repo_root,
    )

    print(f"\n{'='*60}")
    print(f"[pipeline] ✅ Pipeline complete for {station_id}")
    print(f"{'='*60}")
    sys.exit(0)


if __name__ == "__main__":
    main()