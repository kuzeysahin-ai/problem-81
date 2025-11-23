"""
LIVE TRADING SIGNAL GENERATOR
==============================

Your Personal Trading Assistant - Real-Time Market Signals

This script answers the question: "WHAT SHOULD I DO RIGHT NOW?"

Usage:
    python live_trading_signal.py

Features:
    - Fetches LIVE market data (last 60 days, 1-hour bars)
    - Loads trained Phase 7 models (GOOG, XOM, NVDA, CAT)
    - Applies Phase 9 SMART EXECUTION rules
    - Generates actionable BUY/SELL/HOLD signals
    - Beautiful terminal dashboard with color-coded actions

Output:
    Real-time signal table showing:
    - Current price
    - Model prediction (BULLISH/BEARISH)
    - Volatility check (ATR filter)
    - Final ACTION (BUY/SELL/HOLD/WAIT)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import yfinance as yf

# Configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models" / "intraday"
POSITIONS_FILE = PROJECT_ROOT / "data" / "current_positions.csv"

# Trading universe (Phase 9 winners)
TRADING_STOCKS = ['GOOG', 'XOM', 'NVDA', 'CAT']

# SMART EXECUTION PARAMETERS (from Phase 9)
MIN_ATR_THRESHOLD = 0.002  # 0.2% - Don't trade if ATR < 0.2% of price
ATR_PERIOD = 14  # 14-hour ATR

# Feature columns to exclude (must match training)
EXCLUDE_COLS = [
    'target_1h_return', 'target_class',
    'open', 'high', 'low', 'close', 'volume',
    'dividends', 'stock_splits',
    'hour', 'minute', 'day_of_week',
    'return', 'log_return', 'prev_close', 'tr'
]


def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    return df


def add_intraday_features(df):
    """Add all intraday features (EXACT COPY FROM PHASE 7)"""

    # Price and volume basics
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # VWAP
    df = calculate_vwap(df)

    # Intraday momentum
    df['momentum_3h'] = df['return'].rolling(3).sum()
    df['momentum_5h'] = df['return'].rolling(5).sum()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
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

    # SESSION TIME FEATURES
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['is_opening'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)
    df['is_morning'] = ((df['hour'] >= 10) & (df['hour'] < 12)).astype(int)
    df['is_lunch'] = ((df['hour'] >= 12) & (df['hour'] < 14)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 14) & (df['hour'] < 15)).astype(int)
    df['is_closing'] = (df['hour'] >= 15).astype(int)

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # Lag features
    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['volume_lag1'] = df['volume'].shift(1)

    # ATR (Average True Range)
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


def fetch_live_data(ticker, period='60d', interval='1h'):
    """Fetch live market data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"    [ERROR] No data received for {ticker}")
            return None

        # Lowercase column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Filter to market hours (rough filter)
        df.index = pd.to_datetime(df.index)

        # Add features
        df = add_intraday_features(df)

        # Drop NaN
        df = df.dropna()

        return df

    except Exception as e:
        print(f"    [ERROR] Failed to fetch {ticker}: {e}")
        return None


def load_model(ticker):
    """Load trained model"""
    model_path = MODELS_DIR / f"xgb_intraday_{ticker}.pkl"

    if not model_path.exists():
        print(f"    [WARNING] Model not found: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def prepare_features(df, model):
    """Prepare features for prediction"""
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    # Fill NaN and inf
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Match feature order from training
    if hasattr(model, 'get_booster'):
        feature_names = model.get_booster().feature_names
        if feature_names:
            # Ensure all features exist
            for feat in feature_names:
                if feat not in X.columns:
                    X[feat] = 0
            X = X[feature_names]

    return X


def load_current_positions():
    """Load current positions from file (if exists)"""
    if not POSITIONS_FILE.exists():
        # No positions file - assume all FLAT
        return {ticker: False for ticker in TRADING_STOCKS}

    try:
        df = pd.read_csv(POSITIONS_FILE)
        positions = {}
        for _, row in df.iterrows():
            positions[row['ticker']] = row['has_position']
        return positions
    except Exception as e:
        print(f"[WARNING] Could not load positions: {e}")
        return {ticker: False for ticker in TRADING_STOCKS}


def save_positions(positions):
    """Save current positions to file"""
    df = pd.DataFrame([
        {'ticker': ticker, 'has_position': has_pos}
        for ticker, has_pos in positions.items()
    ])
    df.to_csv(POSITIONS_FILE, index=False)


def generate_signal(ticker):
    """Generate trading signal for a single stock"""

    # Fetch live data
    df = fetch_live_data(ticker)
    if df is None or len(df) < 50:
        return {
            'ticker': ticker,
            'price': 0.0,
            'signal': 'N/A',
            'confidence': 0.0,
            'volatility_ok': False,
            'atr_pct': 0.0,
            'action': 'ERROR',
            'reason': 'No data'
        }

    # Load model
    model = load_model(ticker)
    if model is None:
        return {
            'ticker': ticker,
            'price': df.iloc[-1]['close'],
            'signal': 'N/A',
            'confidence': 0.0,
            'volatility_ok': False,
            'atr_pct': 0.0,
            'action': 'ERROR',
            'reason': 'No model'
        }

    # Get latest bar
    latest = df.iloc[-1]
    current_price = latest['close']
    atr_pct = latest['ATR_pct']

    # Prepare features for prediction
    X = prepare_features(df.tail(1), model)

    # Get prediction
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0, 1]

    # SMART EXECUTION LOGIC (Phase 9)
    signal_text = 'BULLISH' if prediction == 1 else 'BEARISH'
    volatility_ok = atr_pct >= MIN_ATR_THRESHOLD

    # Determine ACTION
    if prediction == 1:  # Model says UP
        if volatility_ok:
            action = 'BUY'
            reason = f'Model bullish + ATR {atr_pct:.2%} > {MIN_ATR_THRESHOLD:.2%}'
        else:
            action = 'WAIT'
            reason = f'Low volatility (ATR {atr_pct:.2%} < {MIN_ATR_THRESHOLD:.2%})'
    else:  # Model says DOWN
        action = 'SELL/FLAT'
        reason = 'Model bearish - exit/avoid position'

    return {
        'ticker': ticker,
        'price': current_price,
        'signal': signal_text,
        'confidence': confidence,
        'volatility_ok': volatility_ok,
        'atr_pct': atr_pct,
        'action': action,
        'reason': reason,
        'last_update': df.index[-1]
    }


def print_dashboard(signals, positions):
    """Print beautiful trading dashboard"""

    # Header
    print("\n" + "="*100)
    print("LIVE TRADING SIGNALS - PHASE 9 SMART EXECUTION")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Based on: Last 60 days hourly data")
    print(f"Models: Phase 7 Intraday (GOOG, XOM, NVDA, CAT)")
    print(f"Strategy: Smart Execution with ATR Filter ({MIN_ATR_THRESHOLD:.2%})")
    print("="*100)

    # Positions summary
    num_positions = sum(positions.values())
    print(f"\nCURRENT POSITIONS: {num_positions}/4")
    for ticker, has_pos in positions.items():
        status = "[LONG]" if has_pos else "[FLAT]"
        print(f"  {ticker}: {status}")

    # Signal table header
    print("\n" + "-"*100)
    print(f"{'TICKER':<8} {'PRICE':<12} {'SIGNAL':<15} {'CONFIDENCE':<12} {'ATR %':<10} {'VOL OK?':<10} {'ACTION':<15}")
    print("-"*100)

    # Sort by action priority: BUY > SELL/FLAT > WAIT > ERROR
    action_priority = {'BUY': 0, 'SELL/FLAT': 1, 'WAIT': 2, 'ERROR': 3}
    signals_sorted = sorted(signals, key=lambda x: action_priority.get(x['action'], 4))

    # Print each signal with color coding (using symbols)
    for sig in signals_sorted:
        price_str = f"${sig['price']:.2f}"
        conf_str = f"{sig['confidence']:.1%}"
        atr_str = f"{sig['atr_pct']:.2%}"
        vol_ok_str = "YES" if sig['volatility_ok'] else "NO"

        # Action with visual indicators
        action_display = sig['action']
        if sig['action'] == 'BUY':
            action_display = ">>> BUY <<<"
        elif sig['action'] == 'SELL/FLAT':
            action_display = "<<< SELL/FLAT <<<"
        elif sig['action'] == 'WAIT':
            action_display = "--- WAIT ---"

        print(f"{sig['ticker']:<8} {price_str:<12} {sig['signal']:<15} {conf_str:<12} "
              f"{atr_str:<10} {vol_ok_str:<10} {action_display:<15}")

    print("-"*100)

    # Action summary
    print("\nACTION SUMMARY:")
    actions = {}
    for sig in signals:
        action = sig['action']
        if action not in actions:
            actions[action] = []
        actions[action].append(sig['ticker'])

    for action, tickers in actions.items():
        print(f"  {action}: {', '.join(tickers)}")

    # Detailed recommendations
    print("\nDETAILED RECOMMENDATIONS:")
    print("-"*100)
    for sig in signals_sorted:
        if sig['action'] in ['BUY', 'SELL/FLAT']:
            print(f"\n{sig['ticker']} - {sig['action']}:")
            print(f"  Price:       ${sig['price']:.2f}")
            print(f"  Signal:      {sig['signal']} (Confidence: {sig['confidence']:.1%})")
            print(f"  Volatility:  ATR {sig['atr_pct']:.2%} ({'OK' if sig['volatility_ok'] else 'LOW'})")
            print(f"  Reason:      {sig['reason']}")
            print(f"  Last Update: {sig['last_update']}")

            # Position-specific advice
            has_position = positions.get(sig['ticker'], False)
            if sig['action'] == 'BUY' and not has_position:
                print(f"  >> RECOMMENDATION: ENTER LONG POSITION")
            elif sig['action'] == 'BUY' and has_position:
                print(f"  >> RECOMMENDATION: HOLD EXISTING POSITION")
            elif sig['action'] == 'SELL/FLAT' and has_position:
                print(f"  >> RECOMMENDATION: EXIT POSITION (Model turned bearish)")
            elif sig['action'] == 'SELL/FLAT' and not has_position:
                print(f"  >> RECOMMENDATION: STAY FLAT (Avoid new position)")

    print("\n" + "="*100)
    print("RISK WARNING:")
    print("  - Past performance does not guarantee future results")
    print("  - Phase 9 backtest: +3.54% return, 58.95% win rate, Sharpe 4.64")
    print("  - Always use stop losses and proper position sizing")
    print("  - This is for educational purposes - trade at your own risk")
    print("="*100 + "\n")


def main():
    """Main execution"""
    print("\nFetching live market data...")
    print("-" * 50)

    # Load current positions
    positions = load_current_positions()

    # Generate signals for all stocks
    signals = []
    for ticker in TRADING_STOCKS:
        print(f"  Processing {ticker}...", end=' ')
        signal = generate_signal(ticker)
        signals.append(signal)
        print(f"[DONE] {signal['action']}")

    # Print dashboard
    print_dashboard(signals, positions)

    # Optional: Update positions based on actions
    # (This is manual - user should update positions.csv after executing trades)
    print("NOTE: After executing trades, update your positions in:")
    print(f"      {POSITIONS_FILE}")
    print("\nFormat: ticker,has_position")
    print("Example:")
    print("  GOOG,True")
    print("  XOM,False")
    print("  NVDA,False")
    print("  CAT,True")


if __name__ == "__main__":
    main()
