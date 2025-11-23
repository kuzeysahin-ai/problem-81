"""
PHASE 8: EQUITY CURVE SIMULATION - BACKTEST WINNERS
====================================================

Strategic Goal:
  Convert theoretical Precision into REAL MONEY.
  Answer: "If we traded these models, how much would we have made?"

Winner Models (from Phase 7):
  1. NVDA: 61.36% Precision (Ultra-high, low coverage)
  2. XOM:  55.02% Precision (Best balance)
  3. GOOG: 57.87% Precision (High precision, decent coverage)
  4. CAT:  60.53% Precision (High precision, low coverage)

Simulation Rules:
  - Starting Capital: $10,000
  - Equal allocation: $2,500 per stock
  - Strategy: 1-Hour Hold (Buy on signal, Sell next hour)
  - Transaction Cost: 0.10% per round trip (Buy+Sell)
  - Period: Test set (15% of data, same as Phase 7)

Output:
  - Total Return (%)
  - Max Drawdown (%)
  - Sharpe Ratio
  - Win Rate (%)
  - Final Portfolio Value ($)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models" / "intraday"
DATA_DIR = PROJECT_ROOT / "data" / "intraday"
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Winner stocks from Phase 7
WINNER_STOCKS = ['NVDA', 'XOM', 'GOOG', 'CAT']

# Backtest parameters
STARTING_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # 0.10% per round trip (0.05% buy + 0.05% sell)
ALLOCATION_PER_STOCK = STARTING_CAPITAL / len(WINNER_STOCKS)

# Feature columns (must match Phase 7)
EXCLUDE_COLS = [
    'target_1h_return', 'target_class',
    'open', 'high', 'low', 'close', 'volume',
    'dividends', 'stock_splits',
    'hour', 'minute', 'day_of_week',
    'return', 'log_return'  # Intermediate calculations
]


def calculate_vwap(df):
    """Calculate Volume Weighted Average Price (institutional indicator)"""
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    return df


def add_intraday_features(df):
    """Add intraday-specific features (EXACT COPY FROM PHASE 7)"""

    # Price and volume basics
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # VWAP (institutional benchmark)
    df = calculate_vwap(df)

    # Intraday momentum
    df['momentum_3h'] = df['return'].rolling(3).sum()
    df['momentum_5h'] = df['return'].rolling(5).sum()

    # RSI (hourly)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands (hourly)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)

    # Moving averages
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['ma_gap_5_20'] = (df['MA5'] - df['MA20']) / df['MA20']

    # Volume features
    df['volume_ma10'] = df['volume'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma10'] + 1e-6)

    # Price range
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']

    # SESSION TIME FEATURES (CRITICAL FOR INTRADAY)
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['is_opening'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)  # 9:30-10:00
    df['is_morning'] = ((df['hour'] >= 10) & (df['hour'] < 12)).astype(int)
    df['is_lunch'] = ((df['hour'] >= 12) & (df['hour'] < 14)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 14) & (df['hour'] < 15)).astype(int)
    df['is_closing'] = (df['hour'] >= 15).astype(int)  # 3:00-4:00 PM

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Lag features
    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['volume_lag1'] = df['volume'].shift(1)

    # Target (next hour return)
    df['target_1h_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target_class'] = (df['target_1h_return'] > 0).astype(int)

    return df


def load_intraday_data(ticker):
    """Load cached intraday data and add features"""
    cache_file = DATA_DIR / f"{ticker}_1h.csv"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cached data not found: {cache_file}")

    # Load raw data
    df = pd.read_csv(cache_file, index_col=0)

    # CRITICAL: Convert index to datetime with timezone awareness
    df.index = pd.to_datetime(df.index, utc=True)

    # Rename columns to lowercase (match Phase 7)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Add all intraday features
    df = add_intraday_features(df)

    # Drop rows with NaN (from rolling calculations)
    df = df.dropna()

    return df


def prepare_features(df, model):
    """Prepare feature matrix (same as Phase 7)"""
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    # Fill NaN with 0 (same as training)
    X = X.fillna(0)

    # Replace inf with 0
    X = X.replace([np.inf, -np.inf], 0)

    # CRITICAL: Match exact feature order from training
    # XGBoost requires features in same order as training
    if hasattr(model, 'get_booster'):
        feature_names = model.get_booster().feature_names
        if feature_names:
            # Reorder columns to match training
            X = X[feature_names]

    return X


def load_model(ticker):
    """Load trained intraday model"""
    model_path = MODELS_DIR / f"xgb_intraday_{ticker}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def split_data(df, train_ratio=0.70, val_ratio=0.15):
    """Split data chronologically (same as Phase 7)"""
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_end = train_size
    val_end = train_size + val_size

    test_df = df.iloc[val_end:].copy()

    return test_df


def backtest_stock(ticker):
    """
    Backtest a single stock with 1-hour hold strategy.

    Strategy:
      - Model predicts UP (class 1) -> BUY at current close
      - Hold for 1 hour -> SELL at next hour's close
      - Calculate profit/loss after transaction costs

    Returns:
      dict with performance metrics
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*80}")

    # Load data and model
    df = load_intraday_data(ticker)
    model = load_model(ticker)

    # Get test set (last 15%)
    test_df = split_data(df)

    print(f"  Test Period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"  Test Bars: {len(test_df)}")

    # Prepare features (pass model for feature ordering)
    X_test = prepare_features(test_df, model)

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Prepare actual returns (next hour's return)
    test_df['next_return'] = test_df['target_1h_return'].shift(-1)

    # Initialize portfolio tracking
    trades = []
    capital = ALLOCATION_PER_STOCK
    shares = 0
    position_open = False

    for i in range(len(test_df) - 1):  # -1 because we need next hour
        current_idx = test_df.index[i]
        next_idx = test_df.index[i + 1]

        prediction = y_pred[i]
        confidence = y_proba[i]
        current_close = test_df.loc[current_idx, 'close']
        next_close = test_df.loc[next_idx, 'close']
        actual_return = test_df.loc[current_idx, 'next_return']

        # Strategy: If model predicts UP (1), open position
        if prediction == 1 and not position_open:
            # BUY at current close
            shares = capital / current_close
            buy_cost = capital * TRANSACTION_COST
            capital -= buy_cost
            position_open = True

            # SELL at next close (1-hour hold)
            sell_proceeds = shares * next_close
            sell_cost = sell_proceeds * TRANSACTION_COST
            sell_proceeds -= sell_cost

            # Calculate profit/loss
            profit = sell_proceeds - capital
            profit_pct = profit / capital

            # Update capital
            capital = sell_proceeds
            position_open = False

            # Record trade
            trades.append({
                'entry_time': current_idx,
                'exit_time': next_idx,
                'entry_price': current_close,
                'exit_price': next_close,
                'shares': shares,
                'profit': profit,
                'profit_pct': profit_pct,
                'confidence': confidence,
                'actual_return': actual_return,
                'win': profit > 0
            })

    # Calculate metrics
    if len(trades) == 0:
        print(f"\n  [WARNING] No trades executed for {ticker}")
        return {
            'ticker': ticker,
            'num_trades': 0,
            'final_value': ALLOCATION_PER_STOCK,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

    trades_df = pd.DataFrame(trades)

    # Calculate equity curve
    trades_df['cumulative_capital'] = ALLOCATION_PER_STOCK + trades_df['profit'].cumsum()

    # Total Return
    final_value = capital
    total_return = (final_value - ALLOCATION_PER_STOCK) / ALLOCATION_PER_STOCK

    # Max Drawdown
    cumulative_max = trades_df['cumulative_capital'].cummax()
    drawdown = (trades_df['cumulative_capital'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # Sharpe Ratio (assuming 252*6.5 = 1638 trading hours per year, risk-free rate = 0)
    returns = trades_df['profit_pct']
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(test_df)) if returns.std() > 0 else 0

    # Win Rate
    win_rate = trades_df['win'].sum() / len(trades_df)

    # Average Profit/Loss
    winning_trades = trades_df[trades_df['win'] == True]
    losing_trades = trades_df[trades_df['win'] == False]

    avg_profit = winning_trades['profit_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['profit_pct'].mean() if len(losing_trades) > 0 else 0

    # Profit Factor
    total_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    # Print results
    print(f"\n  PERFORMANCE METRICS:")
    print(f"    Total Trades:     {len(trades_df)}")
    print(f"    Win Rate:         {win_rate:.2%}")
    print(f"    Total Return:     {total_return:.2%}")
    print(f"    Max Drawdown:     {max_drawdown:.2%}")
    print(f"    Sharpe Ratio:     {sharpe_ratio:.3f}")
    print(f"    Avg Win:          {avg_profit:.2%}")
    print(f"    Avg Loss:         {avg_loss:.2%}")
    print(f"    Profit Factor:    {profit_factor:.2f}")
    print(f"    Final Value:      ${final_value:,.2f}")
    print(f"    P&L:              ${final_value - ALLOCATION_PER_STOCK:,.2f}")

    # Save trade log
    trades_df.to_csv(RESULTS_DIR / f"{ticker}_trades.csv", index=False)

    return {
        'ticker': ticker,
        'num_trades': len(trades_df),
        'final_value': final_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def run_portfolio_backtest():
    """Run backtest for all winner stocks and calculate portfolio metrics"""
    print("="*80)
    print("PHASE 8: EQUITY CURVE SIMULATION - BACKTEST WINNERS")
    print("="*80)
    print(f"\nWinner Stocks: {', '.join(WINNER_STOCKS)}")
    print(f"Starting Capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Allocation per Stock: ${ALLOCATION_PER_STOCK:,.2f}")
    print(f"Transaction Cost: {TRANSACTION_COST:.2%} per round trip")
    print(f"Strategy: 1-Hour Hold (Buy on signal, Sell next hour)")

    # Backtest each stock
    results = []
    for ticker in WINNER_STOCKS:
        try:
            result = backtest_stock(ticker)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to backtest {ticker}: {e}")
            continue

    # Portfolio Summary
    print(f"\n{'='*80}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*80}")

    results_df = pd.DataFrame(results)

    # Portfolio metrics
    portfolio_final_value = results_df['final_value'].sum()
    portfolio_return = (portfolio_final_value - STARTING_CAPITAL) / STARTING_CAPITAL
    portfolio_pnl = portfolio_final_value - STARTING_CAPITAL

    # Weighted average metrics
    total_trades = results_df['num_trades'].sum()
    avg_win_rate = results_df['win_rate'].mean()
    avg_sharpe = results_df['sharpe_ratio'].mean()
    worst_drawdown = results_df['max_drawdown'].min()

    print(f"\n  PORTFOLIO PERFORMANCE:")
    print(f"    Starting Capital:   ${STARTING_CAPITAL:,.2f}")
    print(f"    Final Value:        ${portfolio_final_value:,.2f}")
    print(f"    Total Return:       {portfolio_return:.2%}")
    print(f"    P&L:                ${portfolio_pnl:,.2f}")
    print(f"")
    print(f"  RISK METRICS:")
    print(f"    Max Drawdown (Worst):  {worst_drawdown:.2%}")
    print(f"    Avg Sharpe Ratio:      {avg_sharpe:.3f}")
    print(f"")
    print(f"  TRADING METRICS:")
    print(f"    Total Trades:       {total_trades}")
    print(f"    Avg Win Rate:       {avg_win_rate:.2%}")

    # Stock-by-stock breakdown
    print(f"\n  STOCK BREAKDOWN:")
    print(f"  {'-'*78}")
    print(f"  {'Stock':<8} {'Trades':<8} {'Win Rate':<10} {'Return':<12} {'Final Value':<15} {'P&L':<12}")
    print(f"  {'-'*78}")

    for _, row in results_df.iterrows():
        pnl = row['final_value'] - ALLOCATION_PER_STOCK
        print(f"  {row['ticker']:<8} {row['num_trades']:<8} {row['win_rate']:<10.2%} "
              f"{row['total_return']:<12.2%} ${row['final_value']:<14,.2f} ${pnl:<11,.2f}")

    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<8} {total_trades:<8} {avg_win_rate:<10.2%} "
          f"{portfolio_return:<12.2%} ${portfolio_final_value:<14,.2f} ${portfolio_pnl:<11,.2f}")
    print(f"  {'-'*78}")

    # Save summary
    results_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)

    # Portfolio summary
    portfolio_summary = {
        'starting_capital': STARTING_CAPITAL,
        'final_value': portfolio_final_value,
        'total_return': portfolio_return,
        'pnl': portfolio_pnl,
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate,
        'avg_sharpe_ratio': avg_sharpe,
        'worst_drawdown': worst_drawdown
    }

    portfolio_df = pd.DataFrame([portfolio_summary])
    portfolio_df.to_csv(RESULTS_DIR / "portfolio_summary.csv", index=False)

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved:")
    print(f"  - Individual trades: {RESULTS_DIR}/[TICKER]_trades.csv")
    print(f"  - Stock summary:     {RESULTS_DIR}/backtest_summary.csv")
    print(f"  - Portfolio summary: {RESULTS_DIR}/portfolio_summary.csv")

    # Final verdict
    print(f"\n{'='*80}")
    print("CFO VERDICT")
    print(f"{'='*80}")

    if portfolio_return > 0.05:  # >5% return
        print(f"  [SUCCESS] Portfolio returned {portfolio_return:.2%}")
        print(f"  System is PROFITABLE with realistic transaction costs.")
    elif portfolio_return > 0:
        print(f"  [MARGINAL] Portfolio returned {portfolio_return:.2%}")
        print(f"  System is profitable but barely beats transaction costs.")
    else:
        print(f"  [FAILED] Portfolio lost {abs(portfolio_return):.2%}")
        print(f"  Transaction costs eroded theoretical edge.")

    print(f"\n  Bottom Line: If you invested ${STARTING_CAPITAL:,.2f} using this system,")
    print(f"               you would now have ${portfolio_final_value:,.2f}")
    print(f"               ({portfolio_pnl:+,.2f} P&L)")
    print()


if __name__ == "__main__":
    run_portfolio_backtest()
