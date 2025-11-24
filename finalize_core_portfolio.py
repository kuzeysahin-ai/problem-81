"""
PHASE 15: THE FOCUSED CORE PORTFOLIO
=====================================

THE EVOLUTION:
  Phase 7-8: Tested 10+ stocks, found 4 winners (NVDA, XOM, GOOG, CAT)
  Phase 9: Smart Execution -> 58.95% Win Rate, +3.54% Return
  Phase 15: CFO Decision -> Focus on 5 Sector Leaders

THE BIG 5 SELECTION:
  1. GOOG  - Tech (Proven Winner)
  2. XOM   - Energy (Proven Hedge)
  3. NVDA  - Semi (Proven Sniper)
  4. JPM   - Finance (New Sector Test)
  5. KO    - Consumer Staples (Stability Test)

STRATEGY:
  - Phase 9 Intraday Model + Smart Execution
  - Cumulative Hold (ride trends)
  - Volatility Filter (ATR > 0.2%)
  - Commission: 0.10% (realistic)
  - Timeframe: 2 years, 1h intervals

FINAL VALIDATION before LIVE deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import yfinance as yf
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models" / "intraday"
DATA_DIR = PROJECT_ROOT / "data" / "intraday"
RESULTS_DIR = PROJECT_ROOT / "data" / "core_portfolio"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# THE BIG 5 - FOCUSED CORE PORTFOLIO
CORE_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']

# Backtest parameters
STARTING_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # 0.10% per round trip
ALLOCATION_PER_STOCK = STARTING_CAPITAL / len(CORE_STOCKS)

# SMART EXECUTION PARAMETERS (from Phase 9)
MIN_ATR_THRESHOLD = 0.002  # 0.2% - Don't trade if ATR < 0.2% of price
ATR_PERIOD = 14

# Feature columns
EXCLUDE_COLS = [
    'target_1h_return', 'target_class',
    'open', 'high', 'low', 'close', 'volume',
    'dividends', 'stock_splits',
    'hour', 'minute', 'day_of_week',
    'return', 'log_return', 'prev_close', 'tr'
]


def calculate_vwap(df):
    """Calculate VWAP"""
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    return df


def add_intraday_features(df):
    """Add all intraday features (EXACT COPY FROM PHASE 9)"""
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = calculate_vwap(df)
    df['momentum_3h'] = df['return'].rolling(3).sum()
    df['momentum_5h'] = df['return'].rolling(5).sum()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)

    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['ma_gap_5_20'] = (df['MA5'] - df['MA20']) / df['MA20']

    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma10'] + 1e-6)

    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']

    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['is_opening'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)
    df['is_morning'] = ((df['hour'] >= 10) & (df['hour'] < 12)).astype(int)
    df['is_lunch'] = ((df['hour'] >= 12) & (df['hour'] < 14)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 14) & (df['hour'] < 15)).astype(int)
    df['is_closing'] = (df['hour'] >= 15).astype(int)

    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['volume_lag1'] = df['volume'].shift(1)

    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda x: max(x['high'] - x['low'],
                     abs(x['high'] - x['prev_close']),
                     abs(x['low'] - x['prev_close'])) if pd.notna(x['prev_close']) else x['high'] - x['low'],
        axis=1
    )
    df['ATR'] = df['tr'].rolling(ATR_PERIOD).mean()
    df['ATR_pct'] = df['ATR'] / df['close']

    return df


def fetch_intraday_data(ticker, period='2y', interval='1h'):
    """Fetch 2 years of hourly data"""
    print(f"  📥 Fetching {ticker} ({period}, {interval})...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            print(f"    ⚠️  No data for {ticker}")
            return None

        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.index = pd.to_datetime(df.index)
        df = add_intraday_features(df)
        df = df.dropna()
        print(f"    ✅ {len(df)} rows | {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def train_model(df, ticker):
    """Train XGBoost model (or load if exists)"""
    from xgboost import XGBClassifier

    model_path = MODELS_DIR / f"xgb_intraday_{ticker}.pkl"

    # Check if model exists
    if model_path.exists():
        print(f"  ♻️  Loading existing model: {model_path.name}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    print(f"  🔧 Training new model for {ticker}...")

    # Prepare features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df['target_class'].copy()

    # Split train/test (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Save
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"    ✅ Model saved: {model_path.name}")
    return model


def backtest_smart_execution(df, model, ticker):
    """Backtest with Phase 9 Smart Execution"""
    print(f"\n  🎯 Backtesting {ticker} with Smart Execution...")

    # Prepare features
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Align features with model (handle missing features)
    if hasattr(model, 'get_booster'):
        model_features = model.get_booster().feature_names
        if model_features:
            # Add missing features as zeros
            for feat in model_features:
                if feat not in X.columns:
                    X[feat] = 0
            # Keep only model features in correct order
            X = X[model_features]

    # Predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Add to dataframe
    df = df.copy()
    df['prediction'] = predictions
    df['confidence'] = probabilities

    # SMART EXECUTION LOGIC
    capital = ALLOCATION_PER_STOCK
    position = 0  # 0 = flat, 1 = long
    entry_price = 0
    trades = []
    equity_curve = [capital]

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = df.iloc[i]['prediction']
        atr_pct = df.iloc[i]['ATR_pct']
        timestamp = df.index[i]

        # Volatility filter
        volatility_ok = atr_pct >= MIN_ATR_THRESHOLD

        # ENTRY LOGIC
        if position == 0 and signal == 1 and volatility_ok:
            # Enter long
            shares = capital / current_price
            entry_price = current_price
            position = 1
            capital -= current_price * shares * TRANSACTION_COST  # Entry cost

        # EXIT LOGIC
        elif position == 1 and signal == 0:
            # Exit long
            shares = capital / entry_price  # shares we bought
            exit_price = current_price
            pnl = (exit_price - entry_price) * shares
            capital += pnl
            capital -= exit_price * shares * TRANSACTION_COST  # Exit cost

            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_time': timestamp
            })

            position = 0

        # Update equity curve
        if position == 1:
            # Mark-to-market
            shares = capital / entry_price
            current_equity = capital + (current_price - entry_price) * shares
            equity_curve.append(current_equity)
        else:
            equity_curve.append(capital)

    # Final metrics
    total_return = (capital - ALLOCATION_PER_STOCK) / ALLOCATION_PER_STOCK * 100
    num_trades = len(trades)

    if num_trades > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / num_trades * 100
    else:
        win_rate = 0

    # Sharpe ratio
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5) if returns.std() > 0 else 0

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print(f"    Trades: {num_trades} | Win Rate: {win_rate:.1f}% | Return: {total_return:+.2f}%")

    return {
        'ticker': ticker,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_capital': capital,
        'equity_curve': equity_curve
    }


def main():
    """PHASE 15: Finalize Core Portfolio"""
    print("=" * 70)
    print("PHASE 15: THE FOCUSED CORE PORTFOLIO - FINAL VALIDATION")
    print("=" * 70)
    print(f"\n📋 THE BIG 5: {', '.join(CORE_STOCKS)}")
    print(f"💰 Starting Capital: ${STARTING_CAPITAL:,.2f}")
    print(f"📊 Allocation per Stock: ${ALLOCATION_PER_STOCK:,.2f}")
    print(f"💸 Commission: {TRANSACTION_COST * 100:.2f}%")
    print(f"📅 Period: 2 years, 1h intervals")
    print(f"🎯 Strategy: Phase 9 Smart Execution (Cumulative Hold)\n")

    results = []

    for ticker in CORE_STOCKS:
        print(f"\n{'='*70}")
        print(f"Processing {ticker}")
        print(f"{'='*70}")

        # Fetch data
        df = fetch_intraday_data(ticker)
        if df is None:
            continue

        # Train/load model
        model = train_model(df, ticker)

        # Backtest
        result = backtest_smart_execution(df, model, ticker)
        results.append(result)

    # Generate Final Report
    print("\n\n" + "=" * 70)
    print("FINAL PERFORMANCE REPORT - THE BIG 5")
    print("=" * 70)

    report_df = pd.DataFrame(results)
    report_df = report_df.sort_values('total_return', ascending=False)

    print("\n📊 INDIVIDUAL STOCK PERFORMANCE:\n")
    print(report_df[['ticker', 'win_rate', 'total_return', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))

    # Portfolio aggregate
    total_capital = sum([r['final_capital'] for r in results])
    portfolio_return = (total_capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    avg_win_rate = report_df['win_rate'].mean()
    avg_sharpe = report_df['sharpe_ratio'].mean()

    print(f"\n\n{'='*70}")
    print("PORTFOLIO AGGREGATE METRICS")
    print(f"{'='*70}")
    print(f"  Total Capital:     ${total_capital:,.2f}")
    print(f"  Portfolio Return:  {portfolio_return:+.2f}%")
    print(f"  Avg Win Rate:      {avg_win_rate:.2f}%")
    print(f"  Avg Sharpe Ratio:  {avg_sharpe:.2f}")
    print(f"{'='*70}\n")

    # Save results
    report_path = RESULTS_DIR / "core_portfolio_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"✅ Report saved: {report_path}")

    # Decision criteria
    print("\n\n" + "=" * 70)
    print("CFO DECISION MATRIX")
    print("=" * 70)

    if portfolio_return > 0:
        print("✅ PORTFOLIO IS PROFITABLE")
    else:
        print("⚠️  PORTFOLIO HAS NEGATIVE RETURN")

    if avg_win_rate > 50:
        print("✅ WIN RATE > 50%")
    else:
        print("⚠️  WIN RATE < 50%")

    if avg_sharpe > 1.0:
        print("✅ SHARPE RATIO > 1.0 (Good risk-adjusted returns)")
    else:
        print("⚠️  SHARPE RATIO < 1.0")

    print("\n" + "=" * 70)
    if portfolio_return > 0 and avg_win_rate > 50:
        print("🎯 RECOMMENDATION: APPROVED FOR LIVE TRADING")
        print("\nNext Step: Run the reset script below to update current_positions.csv")
    else:
        print("⚠️  RECOMMENDATION: NEEDS OPTIMIZATION")
        print("\nConsider adjusting parameters or stock selection before going live.")
    print("=" * 70)


if __name__ == "__main__":
    main()
