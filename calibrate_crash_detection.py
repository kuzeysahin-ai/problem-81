"""
CRASH DETECTION PARAMETER SWEEP
================================

STRATEGIC EVOLUTION:
  Phase 1: Panic Exit (defensive) - Lower threshold OK
  Phase 2: Panic Short (offensive) - NEEDS HIGH PRECISION

CRITICAL DISTINCTION:
  ❌ Bad Exit: Sold at bottom, missed upside (annoying but safe)
  ❌ Bad Short: Shorted, then price rallied (LOSES MONEY!)

GOAL: Find optimal parameters for SHORT SELLING strategy

PARAMETER SWEEP GRID:
  ATR Multiplier:    [3.0, 4.0, 5.0]  ← Crash severity
  Volume Multiplier: [2.0, 3.0]       ← Panic confirmation
  Short Duration:    [1h, 3h, 6h]     ← Holding period

TOTAL COMBINATIONS: 3 × 2 × 3 = 18 strategies

EVALUATION METRICS:
  1. Precision: % of profitable shorts (CRITICAL)
  2. Frequency: How often signal triggers
  3. Avg Profit: Average gain per short
  4. Max Loss: Worst case scenario
  5. Sharpe Ratio: Risk-adjusted returns

HYPOTHESIS:
  - Tighter thresholds (4x-5x ATR) → Higher precision, fewer trades
  - Longer holds (6h) → Capture full crash, but more reversal risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime
import sys
from itertools import product

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "data" / "crash_calibration"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# THE BIG 5
CORE_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']

# PARAMETER SWEEP GRID
ATR_MULTIPLIERS = [3.0, 4.0, 5.0]
VOLUME_MULTIPLIERS = [2.0, 3.0]
SHORT_DURATIONS = [1, 3, 6]  # hours

# Fixed parameters
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20


def calculate_vwap(df):
    """Calculate VWAP"""
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(
            x['high'] - x['low'],
            abs(x['high'] - x['prev_close']) if pd.notna(x['prev_close']) else 0,
            abs(x['low'] - x['prev_close']) if pd.notna(x['prev_close']) else 0
        ),
        axis=1
    )
    df['ATR'] = df['tr'].rolling(period).mean()
    return df


def add_indicators(df):
    """Add all technical indicators"""
    df = calculate_vwap(df)
    df = calculate_atr(df, ATR_PERIOD)
    df['volume_ma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['price_drop'] = df['open'] - df['close']  # Positive if red candle
    return df


def detect_crashes(df, atr_mult, vol_mult):
    """
    Detect crash signals with given parameters

    Signal = (Price_Drop > atr_mult*ATR) AND (Volume > vol_mult*Volume_MA) AND (Price < VWAP)
    """
    df = df.copy()

    condition_price = df['price_drop'] > (atr_mult * df['ATR'])
    condition_volume = df['volume'] > (vol_mult * df['volume_ma'])
    condition_vwap = df['close'] < df['vwap']

    df['crash_signal'] = condition_price & condition_volume & condition_vwap

    return df


def backtest_short_strategy(df, atr_mult, vol_mult, short_duration):
    """
    Backtest short selling strategy

    Returns:
        - num_trades: Number of short positions
        - win_rate: % of profitable shorts
        - avg_profit: Average profit per short (%)
        - max_loss: Worst single trade (%)
        - total_return: Cumulative return (%)
        - sharpe: Risk-adjusted return
    """
    # Detect signals
    df = detect_crashes(df, atr_mult, vol_mult)

    trades = []
    capital = 10000.0

    for i in range(len(df)):
        if not df.iloc[i]['crash_signal']:
            continue

        # Entry point
        entry_price = df.iloc[i]['close']
        entry_time = df.index[i]

        # Exit point (after short_duration hours)
        exit_idx = i + short_duration
        if exit_idx >= len(df):
            continue  # Not enough data

        exit_price = df.iloc[exit_idx]['close']
        exit_time = df.index[exit_idx]

        # Short profit (price DROP = profit)
        profit_pct = (entry_price - exit_price) / entry_price * 100

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'profitable': profit_pct > 0
        })

    # Calculate metrics
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_loss': 0,
            'total_return': 0,
            'sharpe': 0
        }

    trades_df = pd.DataFrame(trades)

    num_trades = len(trades_df)
    win_rate = (trades_df['profitable'].sum() / num_trades * 100) if num_trades > 0 else 0
    avg_profit = trades_df['profit_pct'].mean()
    max_loss = trades_df['profit_pct'].min()
    total_return = trades_df['profit_pct'].sum()

    # Sharpe ratio
    returns = trades_df['profit_pct']
    sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0

    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'max_loss': max_loss,
        'total_return': total_return,
        'sharpe': sharpe,
        'trades_df': trades_df
    }


def fetch_data(ticker, period='2y', interval='1h'):
    """Fetch historical data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        return None


def run_parameter_sweep(ticker):
    """Run full parameter sweep for one stock"""
    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: {ticker}")
    print(f"{'='*70}")

    # Fetch data
    print(f"  📥 Fetching data...")
    df = fetch_data(ticker)
    if df is None:
        print(f"    ❌ Failed to fetch {ticker}")
        return None

    print(f"    ✅ {len(df)} hours of data")

    # Add indicators
    df = add_indicators(df)

    # Test all parameter combinations
    results = []
    total_combos = len(ATR_MULTIPLIERS) * len(VOLUME_MULTIPLIERS) * len(SHORT_DURATIONS)
    current = 0

    print(f"\n  🔬 Testing {total_combos} parameter combinations...")

    for atr_mult, vol_mult, short_dur in product(ATR_MULTIPLIERS, VOLUME_MULTIPLIERS, SHORT_DURATIONS):
        current += 1

        # Backtest this configuration
        metrics = backtest_short_strategy(df, atr_mult, vol_mult, short_dur)

        results.append({
            'ticker': ticker,
            'atr_mult': atr_mult,
            'vol_mult': vol_mult,
            'short_dur': short_dur,
            **metrics
        })

        # Progress indicator
        if current % 6 == 0:
            print(f"    Progress: {current}/{total_combos} combinations tested")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'trades_df'} for r in results])

    print(f"\n  ✅ Sweep complete: {len(results_df)} configurations tested")

    return results_df


def analyze_sweep_results(all_results):
    """Analyze parameter sweep results across all stocks"""
    print("\n\n" + "="*70)
    print("PARAMETER SWEEP ANALYSIS")
    print("="*70)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    # Filter out zero-trade configs
    valid_configs = combined[combined['num_trades'] > 0].copy()

    if valid_configs.empty:
        print("\n⚠️  NO VALID CONFIGURATIONS FOUND")
        print("   All parameter combinations resulted in zero trades.")
        print("   → Thresholds may be too strict")
        return

    # Aggregate by configuration (average across stocks)
    config_summary = valid_configs.groupby(['atr_mult', 'vol_mult', 'short_dur']).agg({
        'num_trades': 'sum',
        'win_rate': 'mean',
        'avg_profit': 'mean',
        'max_loss': 'min',
        'total_return': 'sum',
        'sharpe': 'mean'
    }).reset_index()

    # Sort by composite score (win_rate * sharpe)
    config_summary['score'] = config_summary['win_rate'] * config_summary['sharpe']
    config_summary = config_summary.sort_values('score', ascending=False)

    print("\n📊 TOP 10 CONFIGURATIONS (by Win Rate × Sharpe):\n")
    top10 = config_summary.head(10)
    print(top10[['atr_mult', 'vol_mult', 'short_dur', 'num_trades', 'win_rate', 'avg_profit', 'sharpe']].to_string(index=False))

    # Best by different criteria
    print("\n\n" + "="*70)
    print("OPTIMAL CONFIGURATIONS BY CRITERIA")
    print("="*70)

    best_winrate = config_summary.loc[config_summary['win_rate'].idxmax()]
    best_sharpe = config_summary.loc[config_summary['sharpe'].idxmax()]
    best_profit = config_summary.loc[config_summary['avg_profit'].idxmax()]

    print(f"\n🎯 HIGHEST WIN RATE ({best_winrate['win_rate']:.1f}%):")
    print(f"   ATR: {best_winrate['atr_mult']}x | VOL: {best_winrate['vol_mult']}x | HOLD: {best_winrate['short_dur']}h")
    print(f"   Trades: {best_winrate['num_trades']:.0f} | Avg Profit: {best_winrate['avg_profit']:.2f}% | Sharpe: {best_winrate['sharpe']:.2f}")

    print(f"\n📈 BEST SHARPE RATIO ({best_sharpe['sharpe']:.2f}):")
    print(f"   ATR: {best_sharpe['atr_mult']}x | VOL: {best_sharpe['vol_mult']}x | HOLD: {best_sharpe['short_dur']}h")
    print(f"   Trades: {best_sharpe['num_trades']:.0f} | Win Rate: {best_sharpe['win_rate']:.1f}% | Avg Profit: {best_sharpe['avg_profit']:.2f}%")

    print(f"\n💰 HIGHEST AVG PROFIT ({best_profit['avg_profit']:.2f}%):")
    print(f"   ATR: {best_profit['atr_mult']}x | VOL: {best_profit['vol_mult']}x | HOLD: {best_profit['short_dur']}h")
    print(f"   Trades: {best_profit['num_trades']:.0f} | Win Rate: {best_profit['win_rate']:.1f}% | Sharpe: {best_profit['sharpe']:.2f}")

    # Recommendation
    print("\n\n" + "="*70)
    print("CFO/CTO RECOMMENDATION")
    print("="*70)

    # Find "safe" configs (high win rate + reasonable frequency)
    safe_configs = config_summary[
        (config_summary['win_rate'] >= 60) &
        (config_summary['num_trades'] >= 10) &
        (config_summary['sharpe'] > 0)
    ].sort_values('sharpe', ascending=False)

    if not safe_configs.empty:
        recommended = safe_configs.iloc[0]

        print(f"\n✅ RECOMMENDED FOR PRODUCTION:")
        print(f"   ATR Multiplier:    {recommended['atr_mult']}x")
        print(f"   Volume Multiplier: {recommended['vol_mult']}x")
        print(f"   Short Duration:    {recommended['short_dur']} hours")
        print(f"\n📊 Expected Performance:")
        print(f"   Win Rate:      {recommended['win_rate']:.1f}%")
        print(f"   Avg Profit:    {recommended['avg_profit']:.2f}% per trade")
        print(f"   Sharpe Ratio:  {recommended['sharpe']:.2f}")
        print(f"   Trade Frequency: {recommended['num_trades']:.0f} trades/2yr")
        print(f"   Max Loss:      {recommended['max_loss']:.2f}%")

        print(f"\n🎯 Risk Assessment:")
        if recommended['win_rate'] >= 70:
            print(f"   ✅ EXCELLENT: Very high precision")
        elif recommended['win_rate'] >= 60:
            print(f"   ✅ GOOD: Acceptable precision for shorts")
        else:
            print(f"   ⚠️  MODERATE: Use with caution")

        if recommended['num_trades'] < 5:
            print(f"   ⚠️  LOW FREQUENCY: Very rare signals")
        elif recommended['num_trades'] < 20:
            print(f"   ✅ MODERATE FREQUENCY: Selective signals")
        else:
            print(f"   ✅ ACTIVE: Regular opportunities")

    else:
        print("\n⚠️  NO CONFIGURATIONS MEET SAFETY CRITERIA")
        print("   Required: Win Rate ≥ 60%, Trades ≥ 10, Sharpe > 0")
        print("\n💡 Consider:")
        print("   1. Lower thresholds to increase frequency")
        print("   2. Use defensive exits only (no shorts)")
        print("   3. Wait for more extreme market conditions")

    # Save results
    summary_path = RESULTS_DIR / "parameter_sweep_summary.csv"
    config_summary.to_csv(summary_path, index=False)
    print(f"\n\n💾 Full results saved: {summary_path}")

    # Save detailed per-stock results
    detailed_path = RESULTS_DIR / "parameter_sweep_detailed.csv"
    combined.to_csv(detailed_path, index=False)
    print(f"💾 Detailed results saved: {detailed_path}")


def main():
    """Run parameter sweep across all stocks"""
    print("="*70)
    print("CRASH DETECTION PARAMETER SWEEP")
    print("="*70)
    print(f"\n🎯 OBJECTIVE: Find optimal SHORT SELLING parameters")
    print(f"📦 Stocks: {', '.join(CORE_STOCKS)}")
    print(f"\n⚙️  PARAMETER GRID:")
    print(f"   ATR Multipliers:    {ATR_MULTIPLIERS}")
    print(f"   Volume Multipliers: {VOLUME_MULTIPLIERS}")
    print(f"   Short Durations:    {SHORT_DURATIONS} hours")
    print(f"\n📊 Total Combinations: {len(ATR_MULTIPLIERS) * len(VOLUME_MULTIPLIERS) * len(SHORT_DURATIONS)}")
    print(f"🕐 Period: 2 years, hourly data\n")

    all_results = []

    for ticker in CORE_STOCKS:
        results_df = run_parameter_sweep(ticker)
        if results_df is not None:
            all_results.append(results_df)

    if not all_results:
        print("\n❌ No results generated")
        return

    # Analyze combined results
    analyze_sweep_results(all_results)


if __name__ == "__main__":
    main()
