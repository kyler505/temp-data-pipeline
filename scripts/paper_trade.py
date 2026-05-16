#!/usr/bin/env python3
"""Paper trading engine for Kalshi weather markets.

Reads precomputed opportunities from compute_edges.py output (opportunities.jsonl)
and places bets accordingly. This ensures paper trades match the edge engine report.

Usage:
    python scripts/paper_trade.py --station KNYC [--log-dir runs/live]
    python scripts/paper_trade.py --station KNYC --settle  # settle yesterday's bets
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

DEFAULT_BANKROLL = 100.0
BET_SIZE_DOLLARS = 1.0
KELLY_FRACTION = 0.25  # Quarter Kelly for safety
KELLY_CAP = 0.10  # Max 10% of bankroll per bet
MAX_DAILY_BETS = 3
MIN_EDGE = 0.05  # Minimum edge threshold (raised from 0.03 per weekly review)


def load_opportunities(log_dir: Path) -> list[dict]:
    """Load edge opportunities from cache file."""
    cache_path = log_dir / "opportunities.jsonl"
    if not cache_path.exists():
        return []
    ops = []
    for line in cache_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            ops.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Sort by edge descending (already sorted by compute_edges, but ensure)
    ops.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)
    return ops


def load_trades(trade_log: Path) -> list[dict]:
    if not trade_log.exists():
        return []
    trades = []
    for line in trade_log.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            trades.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return trades


def compute_bankroll(trade_log: Path) -> float:
    trades = load_trades(trade_log)
    pnl = sum(t.get("pnl", 0) or 0 for t in trades if t["status"] == "settled")
    open_exposure = sum(t["bet_size"] for t in trades if t["status"] == "open")
    return DEFAULT_BANKROLL + pnl - open_exposure


def kelly_bet_size(bankroll, edge, market_price):
    """Compute Kelly-optimal bet size."""
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return min(1.0, bankroll * 0.02)  # Default: $1 or 2%

    # Kelly formula: f* = edge / (1 - price)
    # For a contract at price p_m with model edge = p - p_m:
    # Payout odds b = (1/p_m - 1), so f* = edge / (1 - p_m)
    full_kelly = edge / (1 - market_price)
    # Cap at 25% Kelly, and at 10% of bankroll
    quarter_kelly = full_kelly * KELLY_FRACTION
    capped = min(quarter_kelly * bankroll, bankroll * KELLY_CAP)
    return max(1.0, round(capped, 2))


def place_paper_bets(opportunities: list[dict], trade_log: Path, bankroll: float) -> list[dict]:
    """Place new paper bets from top opportunities.

    Skips target dates that already have ANY trade history (open or settled).
    This prevents duplicate bets on the same date across multiple pipeline runs.
    """
    trades = []
    daily_budget = min(MAX_DAILY_BETS, int(bankroll / BET_SIZE_DOLLARS))

    # Build set of target dates that already have ANY trade
    excluded_dates = set()
    for line in trade_log.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Exclude dates with any existing trade (open or settled)
        if t.get("target_date"):
            excluded_dates.add(t["target_date"])

    # Group by target date, skipping excluded dates
    by_target = {}
    for op in opportunities:
        td = op["target_date"]
        if td in excluded_dates:
            continue
        by_target.setdefault(td, []).append(op)

    bets_placed = 0
    for target_date in sorted(by_target.keys()):
        if bets_placed >= daily_budget:
            break
        best = by_target[target_date][0]
        trade = {
            "trade_id": f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bets_placed}",
            "timestamp": datetime.now().isoformat(),
            "ticker": best["ticker"],
            "target_date": best["target_date"],
            "side": best["side"],
            "threshold": best["threshold"],
            "model_prob": best["model_prob"],
            "market_price": best["market_price"],
            "edge": best["edge"],
            "bet_size": kelly_bet_size(bankroll, best["edge"], best["market_price"]),
            "status": "open",
            "actual_f": None,
            "pnl": None,
            "settled_at": None,
        }
        trades.append(trade)
        bets_placed += 1

    # Append to log
    with open(trade_log, "a") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    return trades


def settle_trades(trade_log: Path, log_dir: Path, station_id: str):
    """Settle open trades against actuals from predictions.jsonl.

    Settles ALL open trades whose target_date has an actual observation.
    """
    trades = load_trades(trade_log)
    preds_path = log_dir / "predictions.jsonl"
    if not preds_path.exists():
        return 0, 0.0

    # Build lookup of actuals by target_date
    actuals = {}
    for line in preds_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue
        if p.get("station_id") != station_id:
            continue
        if p.get("actual_f") is not None:
            actuals[p["target_date"]] = p["actual_f"]

    settled_count = 0
    total_pnl = 0.0
    updated_trades = []

    for trade in trades:
        if trade["status"] != "open":
            updated_trades.append(trade)
            continue

        target = trade["target_date"]
        if target not in actuals:
            updated_trades.append(trade)
            continue

        actual = actuals[target]
        side = trade["side"]
        threshold = trade["threshold"]
        price = trade["market_price"]
        size = trade["bet_size"]

        win = (actual > threshold) if side == "YES" else (actual <= threshold)
        pnl = size * (1.0 / price - 1.0) if win else -size

        trade["status"] = "settled"
        trade["actual_f"] = actual
        trade["pnl"] = round(pnl, 2)
        trade["settled_at"] = datetime.now().isoformat()
        trade["win"] = win

        settled_count += 1
        total_pnl += pnl
        updated_trades.append(trade)

    # Rewrite log
    with open(trade_log, "w") as f:
        for t in updated_trades:
            f.write(json.dumps(t) + "\n")

    return settled_count, total_pnl


def generate_report(trade_log: Path, new_trades: list[dict]) -> str:
    trades = load_trades(trade_log)
    settled = [t for t in trades if t["status"] == "settled"]
    open_trades = [t for t in trades if t["status"] == "open"]

    total_pnl = sum(t.get("pnl", 0) or 0 for t in settled)
    wins = sum(1 for t in settled if t.get("win"))
    losses = len(settled) - wins
    win_rate = wins / len(settled) if settled else 0

    bankroll = compute_bankroll(trade_log)

    lines = []
    lines.append(f"## 📊 Paper Trading Report — {date.today().isoformat()}")
    lines.append(f"**Bankroll:** ${bankroll:.2f} (started at ${DEFAULT_BANKROLL:.2f})")
    lines.append(f"**Total Settled Bets:** {len(settled)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1%}")
    lines.append(f"**Total P&L:** ${total_pnl:+.2f}")
    if settled:
        lines.append(f"**Avg P&L/Bet:** ${total_pnl/len(settled):+.3f}")

    if new_trades:
        lines.append(f"\n📝 **Today's New Bets ({len(new_trades)}):**")
        lines.append("| Ticker | Side | Threshold | Model Prob | Market Price | Edge | Size |")
        lines.append("|--------|------|-----------|------------|--------------|------|------|")
        for t in new_trades:
            lines.append(f"| {t['ticker']} | {t['side']} | {t['threshold']}°F | {t['model_prob']:.2f} | {t['market_price']:.2f} | {t['edge']:+.3f} | ${t['bet_size']:.0f} |")

    if open_trades:
        lines.append(f"\n📋 **Open Positions ({len(open_trades)}):**")
        lines.append("| Ticker | Side | Threshold | Bet Size | Placed |")
        lines.append("|--------|------|-----------|----------|--------|")
        for t in open_trades:
            placed = t['timestamp'][:10]
            lines.append(f"| {t['ticker']} | {t['side']} | {t['threshold']}°F | ${t['bet_size']:.0f} | {placed} |")

    if settled:
        lines.append(f"\n🏆 **Recent Settled Trades (last 5):**")
        lines.append("| Date | Ticker | Side | Actual | P&L |")
        lines.append("|------|--------|------|--------|-----|")
        for t in sorted(settled, key=lambda x: x.get("settled_at", ""), reverse=True)[:5]:
            actual_str = f"{t['actual_f']:.1f}°F" if t.get('actual_f') is not None else "n/a"
            pnl_str = f"${t['pnl']:+.2f}" if t.get('pnl') is not None else "n/a"
            lines.append(f"| {t['target_date']} | {t['ticker']} | {t['side']} | {actual_str} | {pnl_str} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Paper trading engine")
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    parser.add_argument("--settle", action="store_true", help="Settle open trades with actuals")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--max-bets", type=int, default=MAX_DAILY_BETS)
    parser.add_argument("--bet-size", type=float, default=BET_SIZE_DOLLARS)
    # Unused flags for compatibility
    parser.add_argument("--use-edges", action="store_true", help="Use cached opportunities (always implied)")
    parser.add_argument("--mock", action="store_true", help="Mock markets (handled upstream)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    trade_log = log_dir / "paper_trades.jsonl"

    if args.settle:
        settled, pnl = settle_trades(trade_log, log_dir, args.station)
        print(f"[paper] Settled {settled} trades | P&L: ${pnl:+.2f}")
        return

    # Load opportunities from cache (created by compute_edges.py)
    ops = load_opportunities(log_dir)
    if not ops:
        print("[paper] No opportunities found in cache (did compute_edges.py run?)")
        report = generate_report(trade_log, [])
        print(report)
        return

    # Filter by min_edge (defense in depth)
    ops = [op for op in ops if op.get("edge", 0) >= args.min_edge]
    if not ops:
        print(f"[paper] No opportunities above min_edge={args.min_edge}")
        report = generate_report(trade_log, [])
        print(report)
        return

    bankroll = compute_bankroll(trade_log)
    print(f"[paper] Bankroll: ${bankroll:.2f} | {len(ops)} cached opportunities found")

    new_trades = place_paper_bets(ops, trade_log, bankroll)
    print(f"[paper] Placed {len(new_trades)} paper bet(s)")

    report = generate_report(trade_log, new_trades)
    print(report)


if __name__ == "__main__":
    main()
