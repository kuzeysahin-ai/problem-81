"""
PHASE 19: SMART DATA FETCHING OPTIMIZATION
============================================

Web-based Trading Control Panel with Advanced Risk Controls

Features:
    - Live market signals with real-time updates
    - One-click position management (no manual CSV editing!)
    - Portfolio P&L tracking
    - Interactive price charts
    - Dark mode professional UI
    - PHASE 16: Dynamic Stop-Loss (ATR-based)
    - PHASE 16: Panic Detection (VWAP + Volume)
    - PHASE 16: Risk Zone Indicators (Safe/Warning/Critical)
    - PHASE 13: Google Sheets Persistence (Cloud-first with local fallback)
    - PHASE 18: Kelly Criterion Position Sizing (Fractional Kelly 0.5x)
    - PHASE 19: Smart Data Fetching (3-min cache, retry logic, stale data fallback)

Usage:
    streamlit run dashboard.py

Then open: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# PHASE 13: Import Storage Manager
from storage_manager import StorageManager

# Configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models" / "intraday"

# PHASE 13: Initialize Storage Manager (replaces direct CSV access)
storage = StorageManager()

# Trading universe
TRADING_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']

# Constants
MIN_ATR_THRESHOLD = 0.002
ATR_PERIOD = 14
EXCLUDE_COLS = [
    'target_1h_return', 'target_class',
    'open', 'high', 'low', 'close', 'volume',
    'dividends', 'stock_splits',
    'hour', 'minute', 'day_of_week',
    'return', 'log_return', 'prev_close', 'tr'
]

# PHASE 16: RISK MANAGEMENT PARAMETERS
STOP_LOSS_ATR_MULTIPLIER = 2.0  # Stop = Entry - (2 * ATR)
PANIC_VWAP_THRESHOLD = 0.98     # Price < VWAP * 0.98 = panic
PANIC_VOLUME_MULTIPLIER = 2.0   # Volume > 2x average = panic
STOP_WARNING_THRESHOLD = 0.01   # Warn when within 1% of stop

# Page config
st.set_page_config(
    page_title="Trading Dashboard - Phase 9",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff00;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #333;
        margin: 0.5rem 0;
    }
    .buy-signal {
        background-color: #0d4d0d;
        border-left: 5px solid #00ff00;
    }
    .sell-signal {
        background-color: #4d0d0d;
        border-left: 5px solid #ff0000;
    }
    .wait-signal {
        background-color: #4d4d0d;
        border-left: 5px solid #ffaa00;
    }
    .profit-positive {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .profit-negative {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-safe {
        background-color: #0d4d0d;
        border-left: 5px solid #00ff00;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-warning {
        background-color: #4d4d0d;
        border-left: 5px solid #ffaa00;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .risk-critical {
        background-color: #4d0d0d;
        border-left: 5px solid #ff0000;
        padding: 0.5rem;
        border-radius: 5px;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PHASE 18: KELLY POSITION SIZER
# ============================================================================

class KellySizer:
    """
    Kelly Criterion Position Sizing Engine

    Dynamically calculates optimal position size based on:
    - Model confidence (win probability)
    - Risk/Reward ratio
    - Account size

    Uses Fractional Kelly (0.5x) for safety
    """

    @staticmethod
    def calculate_position_size(win_rate, account_size, risk_reward_ratio=1.5,
                                max_position_pct=0.30, kelly_fraction=0.5):
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            account_size: Total account capital
            risk_reward_ratio: Average Win / Average Loss (default 1.5)
            max_position_pct: Maximum % of account per position (default 0.30)
            kelly_fraction: Fraction of Kelly to use (default 0.5 = Half Kelly)

        Returns:
            dict with:
                - kelly_pct: Raw Kelly percentage
                - safe_pct: Fractional Kelly percentage
                - position_size: Dollar amount to invest
                - share_count: Number of shares (if current_price provided)
                - recommendation: Text recommendation
        """
        # Kelly Formula: f = (p * r - q) / r
        # Where: p = win_rate, q = (1 - win_rate), r = risk_reward_ratio

        q = 1 - win_rate
        kelly_pct = (win_rate - (q / risk_reward_ratio))

        # Apply Fractional Kelly (reduce aggressiveness)
        safe_pct = kelly_pct * kelly_fraction

        # Apply hard cap
        safe_pct = min(safe_pct, max_position_pct)

        # No negative positions
        safe_pct = max(safe_pct, 0.0)

        # Calculate dollar amount
        position_size = account_size * safe_pct

        # Recommendation text
        if safe_pct == 0:
            recommendation = "🚫 NO POSITION - Edge too small"
        elif safe_pct < 0.05:
            recommendation = "⚠️ MINIMAL - Low confidence"
        elif safe_pct < 0.15:
            recommendation = "✅ MODERATE - Standard size"
        elif safe_pct < 0.25:
            recommendation = "🔥 LARGE - High confidence"
        else:
            recommendation = "🚀 MAX SIZE - Very high confidence (capped at 30%)"

        return {
            'kelly_pct': kelly_pct,
            'safe_pct': safe_pct,
            'position_size': position_size,
            'recommendation': recommendation
        }

    @staticmethod
    def calculate_share_count(position_size, current_price):
        """Calculate number of shares for given position size"""
        if current_price <= 0:
            return 0
        return int(position_size / current_price)


# ============================================================================
# PHASE 16: RISK MANAGEMENT ENGINE
# ============================================================================

class RiskManager:
    """
    Advanced Risk Management Engine

    Features:
    - Dynamic ATR-based stop-loss (trailing)
    - Panic detection (VWAP + Volume)
    - Risk zone classification
    """

    @staticmethod
    def calculate_dynamic_stop(entry_price, current_atr, highest_price_since_entry=None):
        """
        Calculate dynamic trailing stop-loss

        Args:
            entry_price: Entry price of position
            current_atr: Current ATR value
            highest_price_since_entry: Highest price since entry (for trailing)

        Returns:
            stop_price: Dynamic stop-loss level
        """
        # Base stop: Entry - (2 * ATR)
        base_stop = entry_price - (STOP_LOSS_ATR_MULTIPLIER * current_atr)

        # Trailing logic: If price went up, move stop up (never down)
        if highest_price_since_entry and highest_price_since_entry > entry_price:
            trailing_stop = highest_price_since_entry - (STOP_LOSS_ATR_MULTIPLIER * current_atr)
            # Use whichever is higher (more protective)
            stop_price = max(base_stop, trailing_stop)
        else:
            stop_price = base_stop

        return stop_price

    @staticmethod
    def detect_panic(current_price, vwap, current_volume, avg_volume):
        """
        Detect panic selling conditions

        Args:
            current_price: Current market price
            vwap: Volume-weighted average price
            current_volume: Current bar volume
            avg_volume: Average volume (e.g., 20-period MA)

        Returns:
            is_panic: Boolean
            panic_reason: String description
        """
        # Condition 1: Price crash below VWAP
        price_panic = current_price < (vwap * PANIC_VWAP_THRESHOLD)

        # Condition 2: Volume spike
        volume_panic = current_volume > (avg_volume * PANIC_VOLUME_MULTIPLIER)

        # Both conditions must be true
        is_panic = price_panic and volume_panic

        if is_panic:
            panic_reason = "CRITICAL: Price crash + Volume spike detected!"
        elif price_panic:
            panic_reason = "Price below VWAP threshold"
        elif volume_panic:
            panic_reason = "Volume spike detected"
        else:
            panic_reason = None

        return is_panic, panic_reason

    @staticmethod
    def classify_risk_zone(current_price, stop_price, entry_price):
        """
        Classify current risk zone

        Args:
            current_price: Current market price
            stop_price: Stop-loss level
            entry_price: Entry price

        Returns:
            zone: 'SAFE', 'WARNING', or 'STOP_HIT'
            emoji: Visual indicator
            color_class: CSS class
        """
        # Calculate distance to stop (as percentage of entry)
        distance_to_stop = (current_price - stop_price) / entry_price

        if current_price <= stop_price:
            return 'STOP_HIT', '🔴', 'risk-critical'
        elif distance_to_stop <= STOP_WARNING_THRESHOLD:
            return 'WARNING', '🟡', 'risk-warning'
        else:
            return 'SAFE', '🟢', 'risk-safe'


@st.cache_data(ttl=180)  # Cache for 3 minutes (PHASE 19: Smart caching)
def calculate_vwap(df):
    """Calculate VWAP"""
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
    return df


@st.cache_data(ttl=180)  # Cache for 3 minutes (PHASE 19: Smart caching)
def add_intraday_features(df):
    """Add all intraday features"""
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


@st.cache_data(ttl=180)  # Cache for 3 minutes (optimized from 5 minutes)
def fetch_live_data(ticker, period='60d', interval='1h'):
    """
    Fetch live data with retry logic

    PHASE 19: Smart Data Fetching
    - 3-minute cache (was 5 minutes)
    - 3 retry attempts with 1-second backoff
    - Graceful degradation: returns stale data if available
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

            # Success: Process and cache the data
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df.index = pd.to_datetime(df.index)
            df = add_intraday_features(df)
            df = df.dropna()

            # Store successful fetch timestamp and backup data
            if 'data_fetch_times' not in st.session_state:
                st.session_state.data_fetch_times = {}
            if 'stale_data' not in st.session_state:
                st.session_state.stale_data = {}

            st.session_state.data_fetch_times[ticker] = datetime.now()
            st.session_state.stale_data[ticker] = df.copy()  # Save as backup

            return df

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                # Not the last attempt - retry after delay
                time.sleep(RETRY_DELAY)
                continue
            else:
                # Last attempt failed - check for stale data
                if 'stale_data' not in st.session_state:
                    st.session_state.stale_data = {}

                # Try to return stale data if available
                if ticker in st.session_state.stale_data:
                    st.warning(f"⚠️ Using stale data for {ticker} (fetch failed: {str(e)[:50]})")
                    return st.session_state.stale_data[ticker]
                else:
                    st.error(f"❌ Error fetching {ticker}: {e}")
                    return None

    return None


@st.cache_resource
def load_model(ticker):
    """Load model"""
    model_path = MODELS_DIR / f"xgb_intraday_{ticker}.pkl"
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def prepare_features(df, model):
    """Prepare features"""
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    if hasattr(model, 'get_booster'):
        feature_names = model.get_booster().feature_names
        if feature_names:
            for feat in feature_names:
                if feat not in X.columns:
                    X[feat] = 0
            X = X[feature_names]
    return X


# PHASE 13: Storage functions now use StorageManager
# (Functions removed - using storage.load_positions(), storage.save_positions(), etc.)


def generate_signal(ticker):
    """Generate signal for ticker with risk metrics"""
    df = fetch_live_data(ticker)
    if df is None or len(df) < 50:
        return None

    model = load_model(ticker)
    if model is None:
        return None

    latest = df.iloc[-1]
    current_price = latest['close']
    atr_pct = latest['ATR_pct']
    atr_value = latest['ATR']  # Absolute ATR value
    vwap = latest['vwap']
    current_volume = latest['volume']

    # Calculate volume average (20-period)
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]

    X = prepare_features(df.tail(1), model)
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0, 1]

    signal_text = 'BULLISH' if prediction == 1 else 'BEARISH'
    volatility_ok = atr_pct >= MIN_ATR_THRESHOLD

    if prediction == 1 and volatility_ok:
        action = 'BUY'
    elif prediction == 1 and not volatility_ok:
        action = 'WAIT'
    else:
        action = 'SELL'

    return {
        'ticker': ticker,
        'price': current_price,
        'signal': signal_text,
        'confidence': confidence,
        'volatility_ok': volatility_ok,
        'atr_pct': atr_pct,
        'atr_value': atr_value,
        'vwap': vwap,
        'current_volume': current_volume,
        'volume_ma': volume_ma,
        'action': action,
        'last_update': df.index[-1],
        'df': df  # For charting
    }


def enter_position(ticker, price):
    """Enter position (PHASE 13: uses StorageManager)"""
    storage.enter_position(ticker, price)
    st.success(f"✅ Entered {ticker} position at ${price:.2f}")
    st.rerun()


def update_highest_price(ticker, current_price):
    """Update highest price for trailing stop (PHASE 13: uses StorageManager)"""
    storage.update_highest_price(ticker, current_price)


def exit_position(ticker, exit_price):
    """Exit position (PHASE 13: uses StorageManager)"""
    pnl = storage.exit_position(ticker, exit_price)
    st.success(f"✅ Exited {ticker} position | P&L: ${pnl:.2f}")
    st.rerun()


def create_price_chart(df, ticker):
    """Create interactive price chart with indicators"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price & Indicators', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                  line=dict(color='rgba(250, 250, 250, 0.3)', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                  line=dict(color='rgba(250, 250, 250, 0.3)', width=1),
                  fill='tonexty', fillcolor='rgba(250, 250, 250, 0.1)'),
        row=1, col=1
    )

    # VWAP
    fig.add_trace(
        go.Scatter(x=df.index, y=df['vwap'], name='VWAP',
                  line=dict(color='orange', width=2)),
        row=1, col=1
    )

    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA5'], name='MA5',
                  line=dict(color='cyan', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                  line=dict(color='magenta', width=1)),
        row=1, col=1
    )

    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume',
               marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig


def render_live_trading_tab(positions, account_capital):
    """Render Live Trading tab content (PHASE 18: with Kelly sizing)"""

    # Portfolio Summary
    st.header("💼 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    num_positions = positions['has_position'].sum()
    total_pnl = 0.0

    # Calculate total P&L
    for _, pos in positions.iterrows():
        if pos['has_position']:
            signal = generate_signal(pos['ticker'])
            if signal:
                entry_price = float(pos['entry_price'])
                current_price = signal['price']
                pnl = current_price - entry_price
                total_pnl += pnl

    with col1:
        st.metric("Active Positions", f"{num_positions}/4")
    with col2:
        st.metric("Total P&L", f"${total_pnl:.2f}",
                 delta=f"{total_pnl/2500*100:.2f}%" if num_positions > 0 else None)
    with col3:
        st.metric("Capital Deployed", f"${num_positions * 2500:.0f}")
    with col4:
        st.metric("Cash Available", f"${(4-num_positions) * 2500:.0f}")

    st.divider()

    # Live Signals
    st.header("🎯 Live Trading Signals")

    # Generate signals for all stocks
    signals = []
    for ticker in TRADING_STOCKS:
        with st.spinner(f"Loading {ticker}..."):
            signal = generate_signal(ticker)
            if signal:
                signals.append(signal)

    # Display each signal
    for signal in signals:
        ticker = signal['ticker']
        pos = positions[positions['ticker'] == ticker].iloc[0]
        has_position = pos['has_position']

        # PHASE 16: Pre-calculate risk metrics for active positions
        risk_zone = None
        risk_emoji = None
        risk_class = None
        is_panic = False
        panic_reason = None
        stop_price = None

        if has_position:
            entry_price = float(pos['entry_price'])
            highest_price = float(pos['highest_price'])
            current_price = signal['price']

            # Update highest price if new high
            update_highest_price(ticker, current_price)
            if current_price > highest_price:
                highest_price = current_price

            # Calculate risk metrics
            stop_price = RiskManager.calculate_dynamic_stop(
                entry_price, signal['atr_value'], highest_price
            )
            risk_zone, risk_emoji, risk_class = RiskManager.classify_risk_zone(
                current_price, stop_price, entry_price
            )
            is_panic, panic_reason = RiskManager.detect_panic(
                current_price, signal['vwap'],
                signal['current_volume'], signal['volume_ma']
            )

        # Determine card color
        if signal['action'] == 'BUY':
            card_class = 'buy-signal'
            emoji = '🟢'
        elif signal['action'] == 'SELL':
            card_class = 'sell-signal'
            emoji = '🔴'
        else:
            card_class = 'wait-signal'
            emoji = '🟡'

        # Create card
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 2])

            with col1:
                st.subheader(f"{emoji} {ticker}")

                # PHASE 19: Show data freshness indicator
                if 'data_fetch_times' in st.session_state and ticker in st.session_state.data_fetch_times:
                    fetch_time = st.session_state.data_fetch_times[ticker]
                    age_seconds = (datetime.now() - fetch_time).total_seconds()
                    if age_seconds > 180:  # Older than cache TTL
                        st.caption(f"⚠️ Data {age_seconds/60:.1f}m old")

                st.metric("Current Price", f"${signal['price']:.2f}")

                if has_position:
                    entry_price = float(pos['entry_price'])
                    current_price = signal['price']

                    # Calculate P&L
                    pnl = current_price - entry_price
                    pnl_pct = (pnl / entry_price) * 100

                    pnl_class = "profit-positive" if pnl > 0 else "profit-negative"
                    st.markdown(f'<p class="{pnl_class}">P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)</p>',
                               unsafe_allow_html=True)
                    st.caption(f"Entry: ${entry_price:.2f}")

                    # PHASE 16: Display Risk Metrics
                    st.markdown(f"🛑 **Stop:** ${stop_price:.2f}")

                    # Display risk zone with color
                    st.markdown(
                        f'<div class="{risk_class}">{risk_emoji} <b>{risk_zone}</b></div>',
                        unsafe_allow_html=True
                    )

                    # Display panic warning if detected
                    if is_panic:
                        st.markdown(
                            f'<div class="risk-critical">⚠️ <b>PANIC: {panic_reason}</b></div>',
                            unsafe_allow_html=True
                        )

            with col2:
                st.write("**Signal Analysis**")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric("Model Signal", signal['signal'])
                    st.metric("Confidence", f"{signal['confidence']:.1%}")

                with col_b:
                    st.metric("Volatility (ATR)", f"{signal['atr_pct']:.2%}")
                    vol_status = "✅ OK" if signal['volatility_ok'] else "⚠️ LOW"
                    st.metric("Vol Check", vol_status)

                # Action recommendation
                action_text = signal['action']
                if has_position:
                    if signal['action'] == 'BUY':
                        action_text = "🔵 HOLD POSITION"
                    elif signal['action'] == 'SELL':
                        action_text = "🔴 EXIT RECOMMENDED"
                else:
                    if signal['action'] == 'BUY':
                        action_text = "🟢 ENTER LONG"
                    elif signal['action'] == 'SELL':
                        action_text = "⭕ STAY FLAT"

                st.write(f"**Action:** {action_text}")

            with col3:
                st.write("**Position Control**")

                if not has_position:
                    # PHASE 18: Kelly Position Calculator
                    kelly_result = KellySizer.calculate_position_size(
                        win_rate=signal['confidence'],
                        account_size=account_capital
                    )

                    # Display Kelly calculation
                    st.info(f"""
                    **📊 Kelly Position Sizer**

                    Model Confidence: **{signal['confidence']:.1%}**
                    Kelly Allocation: **{kelly_result['safe_pct']:.1%}**
                    Recommended: **${kelly_result['position_size']:,.0f}**
                    Shares: **{KellySizer.calculate_share_count(kelly_result['position_size'], signal['price'])}**

                    {kelly_result['recommendation']}
                    """)

                    if st.button(f"📈 ENTER POSITION", key=f"enter_{ticker}",
                                use_container_width=True):
                        enter_position(ticker, signal['price'])
                else:
                    # PHASE 16: Enhanced EXIT button based on risk
                    exit_label = "📉 EXIT POSITION"
                    exit_type = "primary"

                    # Make button critical if stop hit or panic detected
                    if risk_zone == 'STOP_HIT' or is_panic:
                        exit_label = "🚨 EMERGENCY EXIT 🚨"
                        exit_type = "primary"
                    elif risk_zone == 'WARNING':
                        exit_label = "⚠️ EXIT POSITION ⚠️"
                        exit_type = "primary"

                    if st.button(exit_label, key=f"exit_{ticker}",
                                use_container_width=True, type=exit_type):
                        exit_position(ticker, signal['price'])

                    # Display critical exit warning
                    if risk_zone == 'STOP_HIT':
                        st.markdown(
                            '<div class="risk-critical" style="text-align: center; padding: 0.5rem; margin-top: 0.5rem;">🔴 <b>STOP HIT!</b></div>',
                            unsafe_allow_html=True
                        )
                    elif is_panic:
                        st.markdown(
                            '<div class="risk-critical" style="text-align: center; padding: 0.5rem; margin-top: 0.5rem;">🔴 <b>PANIC EXIT!</b></div>',
                            unsafe_allow_html=True
                        )

                # Show chart button
                if st.button(f"📊 View Chart", key=f"chart_{ticker}",
                            use_container_width=True):
                    st.session_state[f'show_chart_{ticker}'] = True

            # Show chart if requested
            if st.session_state.get(f'show_chart_{ticker}', False):
                with st.expander(f"📊 {ticker} Chart", expanded=True):
                    fig = create_price_chart(signal['df'].tail(100), ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button(f"Close Chart", key=f"close_chart_{ticker}"):
                        st.session_state[f'show_chart_{ticker}'] = False
                        st.rerun()

            st.divider()


def render_analytics_tab():
    """Render Analytics tab with performance metrics (PHASE 13: uses StorageManager)"""
    st.header("📊 Performance Analytics")

    # Load trade history
    history = storage.load_trade_history()

    if history.empty:
        st.info("📭 No trades yet. Start trading to see analytics!")
        return

    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(history)
    total_pnl = history['pnl'].sum()
    winning_trades = len(history[history['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${total_pnl:.2f}",
                 delta=f"{total_pnl/2500*100:.2f}%" if total_trades > 0 else None)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        avg_pnl = history['pnl'].mean()
        st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")

    st.divider()

    # Equity Curve
    st.subheader("💰 Equity Curve")

    # Calculate cumulative P&L
    history_sorted = history.sort_values('exit_time')
    history_sorted['cumulative_pnl'] = history_sorted['pnl'].cumsum()

    # Create line chart
    st.line_chart(history_sorted.set_index('exit_time')['cumulative_pnl'])

    st.divider()

    # Trade History Table
    st.subheader("📋 Recent Trades")

    # Format display dataframe
    display_df = history[['ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                          'entry_time', 'exit_time', 'hold_duration']].copy()

    # Format numeric columns
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
    display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")

    # Rename columns for display
    display_df.columns = ['Ticker', 'Entry Price', 'Exit Price', 'P&L ($)',
                          'P&L (%)', 'Entry Time', 'Exit Time', 'Duration']

    # Show most recent first
    st.dataframe(display_df.iloc[::-1], use_container_width=True, hide_index=True)

    # Per-ticker breakdown
    st.divider()
    st.subheader("🎯 Per-Ticker Performance")

    ticker_stats = history.groupby('ticker').agg({
        'pnl': ['sum', 'mean', 'count']
    }).round(2)

    ticker_stats.columns = ['Total P&L', 'Avg P&L', 'Trades']
    st.dataframe(ticker_stats, use_container_width=True)


def main():
    """Main dashboard with tabs"""

    # Header
    st.markdown('<h1 class="main-header">📈 PHASE 9 TRADING DASHBOARD</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        st.write("**Live Trading System**")
        st.write("Phase 9 Smart Execution")
        st.divider()

        # PHASE 19: Enhanced Refresh with data freshness indicator
        if st.button("🔄 Refresh Signals", use_container_width=True, help="Force refresh all market data (clears 3-min cache)"):
            st.cache_data.clear()
            # Clear stale data tracking
            if 'stale_data' in st.session_state:
                del st.session_state.stale_data
            if 'data_fetch_times' in st.session_state:
                del st.session_state.data_fetch_times
            st.rerun()

        # PHASE 19: Detailed Data Freshness Indicator
        st.divider()
        if 'data_fetch_times' in st.session_state and st.session_state.data_fetch_times:
            # Get latest fetch time
            latest_fetch = max(st.session_state.data_fetch_times.values())

            # Convert to NY time (US/Eastern)
            import pytz
            ny_tz = pytz.timezone('US/Eastern')

            # Ensure latest_fetch is timezone-aware
            if latest_fetch.tzinfo is None:
                # Make it timezone-aware (assume local time)
                local_tz = pytz.timezone('UTC')
                latest_fetch = local_tz.localize(latest_fetch)

            ny_time = latest_fetch.astimezone(ny_tz)

            # Calculate data age (use timezone-aware datetime)
            current_time = datetime.now(pytz.UTC)
            if latest_fetch.tzinfo is None:
                latest_fetch = pytz.UTC.localize(latest_fetch)

            time_diff_seconds = (current_time - latest_fetch).total_seconds()
            time_diff_minutes = time_diff_seconds / 60

            # Format last update time
            last_update_str = ny_time.strftime('%H:%M:%S')

            # Format data age
            if time_diff_seconds < 60:
                data_age_str = f"{time_diff_seconds:.0f}s ago"
            else:
                data_age_str = f"{time_diff_minutes:.1f}m ago"

            # Display with color coding
            st.write("**📡 Data Freshness**")

            # Last update time (always shown)
            st.caption(f"🕒 Last Update: **{last_update_str}** (NY)")

            # Data age with color coding
            if time_diff_minutes > 15:
                # STALE: Red warning
                st.error(f"⚠️ Data Age: **{data_age_str}** - STALE DATA!")
                st.caption("⚠️ Data is over 15 minutes old. Click Refresh!")
            elif time_diff_minutes > 5:
                # AGING: Yellow warning
                st.warning(f"📊 Data Age: **{data_age_str}** - Getting old")
                st.caption("Consider refreshing for latest data")
            else:
                # FRESH: Green checkmark
                st.success(f"✅ Data Age: **{data_age_str}** - Fresh")
        else:
            # No data fetched yet
            st.write("**📡 Data Freshness**")
            st.info("⏳ No data loaded yet. Click signals to fetch.")

        # PHASE 18: Account Capital Setting
        st.divider()
        st.write("**💰 Account Capital**")
        if 'account_capital' not in st.session_state:
            st.session_state.account_capital = 10000.0

        account_capital = st.number_input(
            "Total Capital ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=st.session_state.account_capital,
            step=1000.0,
            format="%.2f",
            help="Total account size for Kelly position sizing"
        )
        st.session_state.account_capital = account_capital

        # Display available capital per position
        max_per_position = account_capital * 0.30
        st.caption(f"Max per position: ${max_per_position:,.0f} (30%)")

        st.divider()
        st.write("**System Stats**")
        st.metric("Win Rate", "58.95%")
        st.metric("Sharpe Ratio", "4.64")
        st.metric("Backtest Return", "+3.54%")

        # PHASE 13: Storage status
        st.divider()
        st.write("**Storage Mode**")
        storage_mode = storage.get_mode()
        if storage_mode == "CLOUD":
            st.success(f"☁️ Google Sheets")
        else:
            st.info(f"💾 Local CSV")

            # Show error if Google Sheets connection failed
            error_msg = storage.get_connection_error()
            if error_msg:
                st.error(f"⚠️ **Google Sheets Error:**\n\n{error_msg}")
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    **Common Issues:**

                    1. **Missing secrets**: Check `.streamlit/secrets.toml` exists
                    2. **Wrong format**: Verify JSON structure matches GCP service account
                    3. **Missing libraries**: Run `pip install gspread oauth2client`
                    4. **API not enabled**: Enable Google Sheets API in GCP Console
                    5. **Permission denied**: Share spreadsheet with service account email

                    See `GOOGLE_SHEETS_SETUP.md` for detailed setup guide.
                    """)

    # PHASE 13: Load positions from StorageManager
    positions = storage.load_positions()

    # Create tabs
    tab1, tab2 = st.tabs(["📉 Live Trading", "📊 Analytics"])

    with tab1:
        # PHASE 18: Pass account capital for Kelly sizing
        render_live_trading_tab(positions, account_capital)

    with tab2:
        render_analytics_tab()

    # Footer
    st.divider()
    st.caption("⚠️ **Risk Warning:** Past performance does not guarantee future results. "
              "This is for educational purposes. Trade at your own risk.")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
