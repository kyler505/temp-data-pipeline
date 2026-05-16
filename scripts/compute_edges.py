#!/usr/bin/env python3
"""Edge opportunities cache writer.

Reads predictions, computes edges (using same logic as kalshi_edge.py),
and writes opportunities to disk for paper trading to consume.

This ensures paper trading uses the exact same market data and edge
calculations as the edge engine — no divergence.

Usage:
    python scripts/compute_edges.py --station KNYC [--mock] [--log-dir runs/live]
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

SERIES_TICKER = "KXHIGHNY"
MOCK_SPREAD = 0.05
MIN_EDGE = 0.05


def load_predictions(log_path: Path, station_id: str):
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().split("\n")
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
        key = (pred.get("target_date"), pred.get("horizon_days"))
        existing = by_key.get(key)
        if existing is None or pred.get("date", "") > existing.get("date", ""):
            by_key[key] = pred
    return sorted(by_key.values(), key=lambda p: p.get("horizon_days", 999))


def sigma_for_horizon(horizon_days: int, recent_mae: float) -> float:
    if horizon_days <= 2:
        return max(recent_mae, 0.8)
    return recent_mae * (1.0 + 0.15 * (horizon_days - 2))


def empirical_sigma(log_path, horizon_days, recent_mae, min_samples=10, max_days=60):
    """Compute empirical sigma from recent prediction errors."""
    if not log_path or not log_path.exists():
        return sigma_for_horizon(horizon_days, recent_mae)

    try:
        lines = log_path.read_text().strip().split('\n')
        errors = []
        for line in lines:
            if not line.strip():
                continue
            pred = json.loads(line)
            if pred.get('actual_f') is None:
                continue
            if pred.get('horizon_days') != horizon_days:
                continue
            model_pred = pred.get('model_prediction_f', pred.get('ridge_prediction_f'))
            if model_pred is None:
                continue
            errors.append(abs(model_pred - pred['actual_f']))

        if len(errors) >= min_samples:
            # Use robust sigma estimate: 1.4826 * MAD (median absolute deviation)
            # This is more robust than std to outliers
            arr = np.array(errors[-max_days:])  # Most recent only
            median_abs = np.median(np.abs(arr - np.median(arr)))
            robust_sigma = 1.4826 * median_abs if median_abs > 0 else np.std(arr)
            # Don't let it go below MAE floor
            return max(robust_sigma, 0.8)
    except Exception:
        pass

    return sigma_for_horizon(horizon_days, recent_mae)


def model_probabilities(prediction_f: float, sigma: float, thresholds: list[int]) -> dict[float, float]:
    probs = {}
    for t in thresholds:
        prob = 1.0 - stats.norm.cdf(t, loc=prediction_f, scale=sigma)
        probs[t] = float(np.clip(prob, 0.001, 0.999))
    return probs


def generate_mock_markets(target_date: date, pred: float, sigma: float):
    markets = []
    low = int(np.floor(pred - 3 * sigma))
    high = int(np.ceil(pred + 3 * sigma))
    for threshold in range(low, high + 1):
        model_prob = 1.0 - stats.norm.cdf(threshold, loc=pred, scale=sigma)
        noise = np.random.normal(0, 0.03)
        mid_price = np.clip(model_prob + noise, 0.01, 0.99)
        spread = MOCK_SPREAD
        yes_bid = round(np.clip(mid_price - spread / 2, 0.01, 0.99), 2)
        yes_ask = round(np.clip(mid_price + spread / 2, 0.01, 0.99), 2)
        no_bid = round(np.clip(1 - yes_ask, 0.01, 0.99), 2)
        no_ask = round(np.clip(1 - yes_bid, 0.01, 0.99), 2)
        markets.append({
            "ticker": f"{SERIES_TICKER}-{target_date.strftime('%Y%m%d')}-{threshold}",
            "threshold": threshold,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
        })
    return markets


def compute_edge(prediction_f: float, sigma: float, markets: list[dict], min_edge: float):
    opportunities = []
    for m in markets:
        t = m["threshold"]
        model_prob = 1.0 - stats.norm.cdf(t, loc=prediction_f, scale=sigma)
        model_prob = float(np.clip(model_prob, 0.001, 0.999))

        yes_edge = model_prob - m.get("yes_ask", 1.0)
        no_edge = (1.0 - model_prob) - m.get("no_ask", 1.0)

        if yes_edge >= min_edge:
            opportunities.append({
                "ticker": m["ticker"],
                "target_date": m.get("target_date"),
                "side": "YES",
                "threshold": t,
                "model_prob": round(model_prob, 3),
                "market_price": m.get("yes_ask"),
                "edge": round(yes_edge, 3),
                "horizon_days": m.get("horizon_days"),
            })
        if no_edge >= min_edge:
            opportunities.append({
                "ticker": m["ticker"],
                "target_date": m.get("target_date"),
                "side": "NO",
                "threshold": t,
                "model_prob": round(1.0 - model_prob, 3),
                "market_price": m.get("no_ask"),
                "edge": round(no_edge, 3),
                "horizon_days": m.get("horizon_days"),
            })
    opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
    return opportunities


def main():
    parser = argparse.ArgumentParser(description="Compute edge opportunities cache")
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--log-dir", default="runs/live", type=Path)
    parser.add_argument("--mock", action="store_true", help="Use mock market data")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    args = parser.parse_args()

    log_path = args.log_dir / "predictions.jsonl"
    preds = load_predictions(log_path, args.station)
    if not preds:
        print("[edges] No predictions found")
        sys.exit(1)

    out_path = args.log_dir / "opportunities.jsonl"
    all_ops = []

    for pred in preds:
        target = date.fromisoformat(pred["target_date"])
        horizon = pred.get("horizon_days", 1)
        model_pred = pred.get("model_prediction_f", pred.get("ridge_prediction_f"))
        recent_mae = pred.get("recent_mae", 1.5)
        sigma = empirical_sigma(log_path, horizon, recent_mae)

        if args.mock:
            markets = generate_mock_markets(target, model_pred, sigma)
            # Add metadata to each market
            for m in markets:
                m["target_date"] = target.isoformat()
                m["horizon_days"] = horizon
        else:
            # Live mode: fetch from Kalshi (simplified for now)
            # In production, you'd fetch real orderbooks here
            print(f"[edges] WARNING: live mode not fully implemented, falling back to mock for {target}")
            markets = generate_mock_markets(target, model_pred, sigma)
            for m in markets:
                m["target_date"] = target.isoformat()
                m["horizon_days"] = horizon

        ops = compute_edge(model_pred, sigma, markets, args.min_edge)
        all_ops.extend(ops)

    # Write opportunities
    with open(out_path, "w") as f:
        for op in all_ops:
            f.write(json.dumps(op) + "\n")

    print(f"[edges] Wrote {len(all_ops)} opportunities to {out_path}")


if __name__ == "__main__":
    main()
