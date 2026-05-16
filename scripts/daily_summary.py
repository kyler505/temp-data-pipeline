#!/usr/bin/env python3
"""Generate a daily forecast summary for Discord/CLI display.

Reads predictions.jsonl and model metadata, outputs a compact markdown summary.

Usage:
    python scripts/daily_summary.py [--station KNYC] [--log-dir runs/live]
"""
import argparse
import json
from datetime import date
from pathlib import Path


def load_today_predictions(log_path: Path, station_id: str) -> list[dict]:
    """Load today's latest predictions for a station (deduped by target_date+horizon)."""
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().split("\n")
    today_str = date.today().isoformat()
    by_key = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            pred = json.loads(line)
        except json.JSONDecodeError:
            continue
        if pred.get("station_id") != station_id:
            continue
        pred_date = pred.get("date", "")[:10]
        if pred_date < today_str:
            continue
        key = (pred.get("target_date"), pred.get("horizon_days"))
        existing = by_key.get(key)
        if existing is None or pred.get("date", "") > existing.get("date", ""):
            by_key[key] = pred
    return sorted(by_key.values(), key=lambda p: p.get("horizon_days", 999))


def load_model_meta(log_dir: Path) -> dict | None:
    """Load current model metadata."""
    model_dir = log_dir.parent / "models"
    current_link = model_dir / "current.pkl"
    if not current_link.exists() and not current_link.is_symlink():
        return None
    try:
        meta_path = current_link.resolve().with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def format_summary(preds: list[dict], meta: dict | None, station_id: str) -> str:
    """Format a human-readable summary."""
    lines = []
    lines.append(f"## 🌡️ Daily Forecast Summary — {station_id}")
    lines.append(f"**Date:** {date.today().isoformat()}")

    if meta:
        lines.append(f"**Model:** {meta.get('model_type', 'unknown')} | "
                     f"trained on {meta.get('train_rows', '?')} rows | "
                     f"recent MAE={meta.get('recent_mae', '?')}°F")
    else:
        lines.append("**Model:** unknown (no metadata found)")

    # Paper trading stats
    log_dir = Path(preds[0].get("_log_path", "runs/live/predictions.jsonl")).parent if preds else Path("runs/live")
    trade_log = log_dir / "paper_trades.jsonl"
    if trade_log.exists():
        trades = []
        for line in trade_log.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        settled = [t for t in trades if t["status"] == "settled"]
        open_trades = [t for t in trades if t["status"] == "open"]
        total_pnl = sum(t.get("pnl", 0) or 0 for t in settled)
        wins = sum(1 for t in settled if t.get("win"))
        win_rate = wins / len(settled) if settled else 0
        bankroll = 100.0 + total_pnl - sum(t["bet_size"] for t in open_trades)
        lines.append(f"📊 **Paper Trading:** Bankroll=${bankroll:.2f} | Bets={len(settled)} | P&L=${total_pnl:+.2f} | WR={win_rate:.0%}")

    if not preds:
        lines.append("\n⚠️ No predictions found for today.")
        return "\n".join(lines)

    preds = sorted(preds, key=lambda p: p.get("horizon_days", 999))
    lines.append("\n| Target Date | Horizon | Raw | Predicted | Correction | Model |")
    lines.append("|-------------|---------|-----|-----------|------------|-------|")
    for p in preds:
        td = p["target_date"]
        h = p.get("horizon_days", "?")
        raw = p.get("raw_forecast_f", "?")
        pred_val = p.get("model_prediction_f", p.get("ridge_prediction_f", "?"))
        corr = p.get("correction_f", "?")
        used = p.get("model_used", "?")
        raw_str = f"{raw:.1f}°F" if isinstance(raw, (int, float)) else str(raw)
        pred_str = f"{pred_val:.1f}°F" if isinstance(pred_val, (int, float)) else str(pred_val)
        corr_str = f"{corr:+.1f}°F" if isinstance(corr, (int, float)) else str(corr)
        lines.append(f"| {td} | {h}d | {raw_str} | {pred_str} | {corr_str} | {used} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Daily forecast summary")
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    args = parser.parse_args()
    log_path = args.log_dir / "predictions.jsonl"
    preds = load_today_predictions(log_path, args.station)
    meta = load_model_meta(args.log_dir)
    print(format_summary(preds, meta, args.station))


if __name__ == "__main__":
    main()
