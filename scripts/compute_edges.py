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
MIN_EDGE = 0.05  # Default fallback (used when horizon isn't in dict)
# Horizon-dependent edge thresholds
# Research shows h=1d markets are well-calibrated; even 3% edges are real
# Longer horizons need wider threshold due to higher uncertainty
MIN_EDGE_BY_HORIZON = {1: 0.03, 2: 0.05, 3: 0.07, 5: 0.08, 7: 0.10}


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


def ensemble_probability(ensemble_members_str: str | None, threshold: float, prediction_f: float, sigma: float) -> float:
    """Use ensemble members for direct probability, fall back to normal CDF.

    If ensemble_members_str is available (JSON string of member tmax values in °F),
    computes P(tmax > threshold) directly as count_exceeding / len(members).
    Otherwise falls back to 1 - CDF(threshold) using the normal distribution.
    """
    if ensemble_members_str:
        try:
            members = json.loads(ensemble_members_str)
            if isinstance(members, list) and len(members) >= 10:
                count_exceeding = sum(1 for m in members if m > threshold)
                prob = count_exceeding / len(members)
                return float(np.clip(prob, 0.001, 0.999))
        except Exception:
            pass
    # Fall back to normal CDF
    prob = 1.0 - stats.norm.cdf(threshold, loc=prediction_f, scale=sigma)
    return float(np.clip(prob, 0.001, 0.999))


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


def compute_edge(prediction_f: float, sigma: float, markets: list[dict], min_edge: float, horizon_days: int = 1, ensemble_members: str | None = None):
    opportunities = []
    effective_min_edge = MIN_EDGE_BY_HORIZON.get(horizon_days, min_edge)
    for m in markets:
        threshold = m.get("threshold", 0)
        model_prob = ensemble_probability(ensemble_members, threshold, prediction_f, sigma)

        yes_ask = m.get("yes_ask") or 1.0
        no_ask = m.get("no_ask") or 1.0

        yes_edge = model_prob - yes_ask
        no_edge = (1.0 - model_prob) - no_ask

        if yes_edge >= effective_min_edge:
            opportunities.append({
                "ticker": m["ticker"],
                "target_date": m.get("target_date"),
                "side": "YES",
                "threshold": threshold,
                "model_prob": round(model_prob, 3),
                "market_price": m.get("yes_ask"),
                "edge": round(yes_edge, 3),
                "horizon_days": m.get("horizon_days"),
            })
        if no_edge >= effective_min_edge:
            opportunities.append({
                "ticker": m["ticker"],
                "target_date": m.get("target_date"),
                "side": "NO",
                "threshold": threshold,
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

    def fetch_kalshi_markets(target_date, horizon, min_edge=0.05):
        """Fetch live Kalshi orderbook data."""
        try:
            import os, re, time
            from kalshi_python import KalshiClient, MarketsApi, Configuration

            key_id = os.getenv('KALSHI_API_KEY')
            key_path = os.path.expanduser(os.getenv('KALSHI_RSA_KEY', '~/.kalshi/rsa_key.pem'))

            if not key_id or not os.path.exists(key_path):
                print('[kalshi] KALSHI_API_KEY or RSA key not configured, falling back to mock')
                return None

            config = Configuration(
                host='https://api.elections.kalshi.com/trade-api/v2',
            )
            config.api_key = key_id
            config.key_file = key_path

            client = KalshiClient(configuration=config)
            api = MarketsApi(client)

            # Kalshi uses KXHIGHNY as the series ticker (no date suffix)
            # Markets are named KXHIGHNY-26MAY17-T## or KXHIGHNY-26MAY17-B##.#
            time.sleep(0.5)
            resp = api.get_markets(series_ticker='KXHIGHNY', status='open')
            time.sleep(0.3)
            markets = []
            target_str = target_date.strftime('%y%b%d').upper() if hasattr(target_date, 'strftime') else str(target_date)

            for m in resp.markets or []:
                ticker = m.ticker
                # Skip markets for other dates
                if target_str not in ticker:
                    continue

                try:
                    ob = api.get_market_orderbook(ticker)
                    time.sleep(0.3)
                except Exception:
                    continue

                # Parse threshold from ticker suffix
                # Format: KXHIGHNY-26MAY17-T90  (standard, threshold=90)
                #         KXHIGHNY-26MAY17-B89.5 (bucket, center=89.5)
                suffix = ticker.split('-')[-1]
                if suffix.startswith('T'):
                    threshold = int(suffix[1:])
                elif suffix.startswith('B'):
                    # Bucket contract: use the center value as threshold
                    threshold = float(suffix[1:])
                else:
                    continue

                yes_bid = max((lvl.price for lvl in (ob.orderbook.yes or [])), default=None) if ob.orderbook else None
                yes_ask = min((lvl.price for lvl in (ob.orderbook.yes or [])), default=None) if ob.orderbook else None
                no_bid = max((lvl.price for lvl in (ob.orderbook.no or [])), default=None) if ob.orderbook else None
                no_ask = min((lvl.price for lvl in (ob.orderbook.no or [])), default=None) if ob.orderbook else None

                markets.append({
                    'ticker': ticker,
                    'threshold': threshold,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'no_bid': no_bid,
                    'no_ask': no_ask,
                    'is_bucket': suffix.startswith('B'),
                })

            return markets if markets else None
        except Exception as e:
            print(f'[kalshi] Error fetching Kalshi data: {e}')
            return None

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
        elif target <= date.today():
            # Past dates don't have active Kalshi markets (all settled)
            # Use mock for edge analysis on scored predictions
            markets = generate_mock_markets(target, model_pred, sigma)
        else:
            live_markets = fetch_kalshi_markets(target, horizon, args.min_edge)
            if live_markets:
                markets = live_markets
            else:
                print(f'[edges] Live Kalshi unavailable for {target}, falling back to mock')
                markets = generate_mock_markets(target, model_pred, sigma)

        for m in markets:
            m['target_date'] = target.isoformat()
            m['horizon_days'] = horizon

        ops = compute_edge(model_pred, sigma, markets, args.min_edge, horizon_days=horizon, ensemble_members=pred.get("ensemble_members"))
        all_ops.extend(ops)

    # Write opportunities
    with open(out_path, "w") as f:
        for op in all_ops:
            f.write(json.dumps(op) + "\n")

    print(f"[edges] Wrote {len(all_ops)} opportunities to {out_path}")


if __name__ == "__main__":
    main()
