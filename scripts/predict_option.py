#!/usr/bin/env python
"""
Predict option value under different BTC scenarios.

Usage:
    python scripts/predict_option.py --spot 80800 --strike 79000 --dte 8 --type put --premium 720 --scenarios 77000 75000 83000
    python scripts/predict_option.py --spot 80800 --strike 79000 --dte 8 --type put --premium 720  # Uses default scenarios
"""
import argparse
import os
import sys

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aitrader.settings')
import django
django.setup()

from pathlib import Path
from execution.services.option_pricer import LeveragePredictor, BlackScholes


def main():
    parser = argparse.ArgumentParser(description='Predict option price under BTC scenarios')
    parser.add_argument('--spot', type=float, required=True, help='Current BTC spot price')
    parser.add_argument('--strike', type=float, required=True, help='Option strike price')
    parser.add_argument('--dte', type=int, required=True, help='Days to expiry')
    parser.add_argument('--type', choices=['call', 'put'], required=True, help='Option type')
    parser.add_argument('--premium', type=float, help='Current option premium (per BTC)')
    parser.add_argument('--iv', type=float, default=0.50, help='Implied volatility estimate')
    parser.add_argument('--scenarios', type=float, nargs='+', help='BTC prices to simulate')
    parser.add_argument('--model', type=str, default='models/option_response_predictor_v2.json', help='Model path')
    
    args = parser.parse_args()
    
    # Load predictor
    model_path = Path(args.model)
    if model_path.exists():
        predictor = LeveragePredictor(model_path=model_path)
        print(f"Loaded model: {model_path} ({len(predictor.bucket_stats)} buckets)")
    else:
        predictor = LeveragePredictor()
        print("Using synthetic fallback (no trained model)")
    
    # Default scenarios if not provided
    if args.scenarios:
        scenarios = sorted(args.scenarios)
    else:
        # Default: ±3%, ±5%, ±8%
        scenarios = sorted([
            args.spot * 0.92,
            args.spot * 0.95,
            args.spot * 0.97,
            args.spot * 1.03,
            args.spot * 1.05,
            args.spot * 1.08,
        ])
    
    # Current position info
    moneyness = (args.strike - args.spot) / args.spot
    print(f"\n{'='*60}")
    print(f"CURRENT POSITION")
    print(f"{'='*60}")
    print(f"Spot: ${args.spot:,.0f}")
    print(f"Strike: ${args.strike:,.0f}")
    print(f"Type: {args.type.upper()}")
    print(f"DTE: {args.dte}")
    print(f"Moneyness: {moneyness:.1%}")
    print(f"IV: {args.iv:.0%}")
    if args.premium:
        print(f"Current premium: ${args.premium:,.0f}")
    
    # Process scenarios
    print(f"\n{'='*60}")
    print(f"SCENARIO ANALYSIS")
    print(f"{'='*60}")
    print(f"{'BTC Price':>12} | {'Change':>8} | {'p10':>8} | {'p50':>8} | {'p90':>8} | {'Intrinsic':>10} | {'BS@50%':>10}")
    print("-" * 85)
    
    for target_spot in scenarios:
        btc_change = (target_spot - args.spot) / args.spot
        
        # Model prediction
        pred = predictor.predict(
            dte=args.dte,
            moneyness=moneyness,
            option_type=args.type,
            btc_change_pct=btc_change,
            iv=args.iv,
            days_held=1,
        )
        
        # Intrinsic value
        if args.type == 'put':
            intrinsic = max(args.strike - target_spot, 0)
        else:
            intrinsic = max(target_spot - args.strike, 0)
        
        # Black-Scholes reference
        T = max(args.dte - 1, 1) / 365
        if args.type == 'put':
            bs_price = BlackScholes.put_price(target_spot, args.strike, T, 0.05, 0.50)
        else:
            bs_price = BlackScholes.call_price(target_spot, args.strike, T, 0.05, 0.50)
        
        print(f"${target_spot:>11,.0f} | {btc_change:>+7.1%} | {pred.p10:>+7.0%} | {pred.p50:>+7.0%} | {pred.p90:>+7.0%} | ${intrinsic:>9,.0f} | ${bs_price:>9,.0f}")
    
    # Premium estimates if provided
    if args.premium:
        print(f"\n{'='*60}")
        print(f"ESTIMATED PREMIUM VALUES")
        print(f"{'='*60}")
        print(f"{'BTC Price':>12} | {'Conservative':>14} | {'Base':>14} | {'Optimistic':>14}")
        print("-" * 60)
        
        for target_spot in scenarios:
            btc_change = (target_spot - args.spot) / args.spot
            pred = predictor.predict(
                dte=args.dte,
                moneyness=moneyness,
                option_type=args.type,
                btc_change_pct=btc_change,
                iv=args.iv,
                days_held=1,
            )
            
            cons = args.premium * (1 + pred.p10)
            base = args.premium * (1 + pred.p50)
            opt = args.premium * (1 + pred.p90)
            
            print(f"${target_spot:>11,.0f} | ${cons:>13,.0f} | ${base:>13,.0f} | ${opt:>13,.0f}")


if __name__ == '__main__':
    main()
