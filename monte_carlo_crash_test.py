"""
PHASE 17: MONTE CARLO & STRESS TESTING
=======================================

HYPOTHESIS: "AI Bubble" may burst. Are we prepared?

CURRENT STATE:
  - Big 5 portfolio tested on 2023-2025 bull market
  - Win Rate: 70.57%, Return: +556.87%
  - BUT: No crisis testing yet

THE QUESTION:
  "What if AI stocks crash 30% in 60 days with 2.5x volatility?"

MONTE CARLO METHODOLOGY:
  - 10,000 simulation paths
  - 60 trading days forward
  - Two regimes:
    A) Normal Market (historical params)
    B) AI Bubble Burst (crash scenario)

STRESS TEST PARAMETERS:
  Bubble Burst Scenario:
    - Annual Return: -30% (severe bear)
    - Volatility: 2.5x historical
    - Correlation: →1.0 (everything drops together)

RISK MANAGEMENT INTEGRATION:
  - Stop-loss at 2% or ATR-based levels
  - Position sizing: Equal weight
  - Question: Does our stop-loss save us from ruin?

EVALUATION METRICS:
  1. Probability of Ruin: P(Capital < 50% of initial)
  2. VaR (99%): Max loss with 99% confidence
  3. Survival Rate: % of paths that stay profitable
  4. Max Drawdown: Worst case scenario
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "data" / "monte_carlo"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# THE BIG 5
CORE_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']

# SIMULATION PARAMETERS
NUM_SIMULATIONS = 10000
FORECAST_DAYS = 60
INITIAL_CAPITAL = 10000.0
POSITION_SIZE = INITIAL_CAPITAL / len(CORE_STOCKS)

# RISK MANAGEMENT
STOP_LOSS_PCT = 0.02  # 2% stop-loss
RUIN_THRESHOLD = 0.50  # Ruin = lose 50% of capital

# STRESS TEST SCENARIOS
SCENARIOS = {
    'NORMAL': {
        'name': 'Normal Market',
        'drift_multiplier': 1.0,
        'vol_multiplier': 1.0,
        'correlation_boost': 0.0
    },
    'BUBBLE_BURST': {
        'name': 'AI Bubble Burst',
        'drift_multiplier': -0.30,  # -30% annual return
        'vol_multiplier': 2.5,      # 2.5x volatility
        'correlation_boost': 0.5    # Push correlations toward 1.0
    }
}


def fetch_historical_data(ticker, period='2y', interval='1d'):
    """Fetch historical daily data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.index = pd.to_datetime(df.index)
        return df['close']

    except Exception as e:
        print(f"    ⚠️  Error fetching {ticker}: {e}")
        return None


def calculate_portfolio_params(price_data):
    """
    Calculate portfolio parameters from historical data

    Returns:
        mu: Mean returns (daily)
        sigma: Volatility (daily std)
        corr_matrix: Correlation matrix
    """
    # Calculate daily returns
    returns = price_data.pct_change().dropna()

    # Annualize parameters (252 trading days)
    mu_annual = returns.mean() * 252
    sigma_annual = returns.std() * np.sqrt(252)

    # Daily parameters (convert to numpy arrays)
    mu_daily = (mu_annual / 252).values
    sigma_daily = (sigma_annual / np.sqrt(252)).values

    # Correlation matrix
    corr_matrix = returns.corr()

    return mu_daily, sigma_daily, corr_matrix


def generate_correlated_paths(mu, sigma, corr_matrix, scenario_params, num_sims, num_days):
    """
    Generate correlated price paths using Cholesky decomposition

    Args:
        mu: Mean returns (daily) for each stock
        sigma: Volatility (daily) for each stock
        corr_matrix: Correlation matrix
        scenario_params: Scenario configuration
        num_sims: Number of simulation paths
        num_days: Number of days to simulate

    Returns:
        Array of shape (num_sims, num_stocks, num_days+1)
    """
    num_stocks = len(mu)

    # Apply scenario adjustments
    if scenario_params['drift_multiplier'] < 0:
        # Crash scenario: negative drift
        adjusted_mu = np.ones(num_stocks) * scenario_params['drift_multiplier'] / 252
    else:
        adjusted_mu = mu * scenario_params['drift_multiplier']

    adjusted_sigma = sigma * scenario_params['vol_multiplier']

    # Adjust correlation (push toward 1.0 in crash)
    adjusted_corr = corr_matrix.values + scenario_params['correlation_boost']
    adjusted_corr = np.clip(adjusted_corr, -1, 1)
    np.fill_diagonal(adjusted_corr, 1.0)

    # Ensure positive definite (fix for numerical issues)
    eigenvalues, eigenvectors = np.linalg.eig(adjusted_corr)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    adjusted_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Cholesky decomposition for correlated random variables
    try:
        L = np.linalg.cholesky(adjusted_corr)
    except np.linalg.LinAlgError:
        # Fallback: use original correlation
        L = np.linalg.cholesky(corr_matrix.values)

    # Initialize price paths (start at 100)
    paths = np.zeros((num_sims, num_stocks, num_days + 1))
    paths[:, :, 0] = 100.0

    # Generate correlated random walks
    for t in range(1, num_days + 1):
        # Independent random shocks
        Z = np.random.normal(0, 1, (num_sims, num_stocks))

        # Correlate them
        Z_corr = Z @ L.T

        # Geometric Brownian Motion
        drift = adjusted_mu - 0.5 * adjusted_sigma**2
        diffusion = adjusted_sigma * Z_corr

        returns = drift + diffusion
        paths[:, :, t] = paths[:, :, t-1] * np.exp(returns)

    return paths


def apply_stop_loss(paths, stop_pct):
    """
    Apply stop-loss to simulation paths

    When price drops stop_pct below entry (day 0), freeze position

    Args:
        paths: Array (num_sims, num_stocks, num_days+1)
        stop_pct: Stop-loss percentage (e.g., 0.02 for 2%)

    Returns:
        stopped_paths: Modified paths with stops applied
        stop_triggered: Boolean array (num_sims, num_stocks)
    """
    num_sims, num_stocks, num_days = paths.shape
    stopped_paths = paths.copy()
    stop_triggered = np.zeros((num_sims, num_stocks), dtype=bool)

    entry_price = paths[:, :, 0]  # Day 0 price
    stop_level = entry_price * (1 - stop_pct)

    for t in range(1, num_days):
        # Check if price hit stop
        hit_stop = (paths[:, :, t] <= stop_level) & ~stop_triggered

        # Freeze at stop level
        stopped_paths[hit_stop, t:] = stop_level[hit_stop, np.newaxis]
        stop_triggered |= hit_stop

    return stopped_paths, stop_triggered


def calculate_portfolio_returns(paths, position_sizes):
    """
    Calculate portfolio returns from price paths

    Args:
        paths: Array (num_sims, num_stocks, num_days+1)
        position_sizes: Array of capital allocated to each stock

    Returns:
        portfolio_values: Array (num_sims, num_days+1)
    """
    num_sims, num_stocks, num_days = paths.shape

    # Calculate returns for each stock
    returns = paths / paths[:, :, 0:1] - 1  # Normalize to day 0

    # Reshape position_sizes for broadcasting: (1, num_stocks, 1)
    position_sizes_broadcast = position_sizes.reshape(1, -1, 1)

    # Calculate position values
    position_values = position_sizes_broadcast * (1 + returns)

    # Sum across stocks to get portfolio value
    portfolio_values = position_values.sum(axis=1)

    return portfolio_values


def calculate_risk_metrics(portfolio_values, initial_capital):
    """
    Calculate risk metrics from portfolio simulations

    Args:
        portfolio_values: Array (num_sims, num_days+1)
        initial_capital: Starting capital

    Returns:
        Dict of risk metrics
    """
    final_values = portfolio_values[:, -1]

    # 1. Probability of Ruin
    ruin_threshold = initial_capital * RUIN_THRESHOLD
    prob_ruin = (final_values < ruin_threshold).mean() * 100

    # 2. VaR (Value at Risk) at 99% confidence
    var_99 = np.percentile(final_values - initial_capital, 1)

    # 3. Survival Rate (end positive)
    survival_rate = (final_values > initial_capital).mean() * 100

    # 4. Max Drawdown
    running_max = np.maximum.accumulate(portfolio_values, axis=1)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min(axis=1).mean() * 100

    # 5. Expected final value
    expected_value = final_values.mean()

    # 6. Worst case (1st percentile)
    worst_case = np.percentile(final_values, 1)

    return {
        'prob_ruin': prob_ruin,
        'var_99': var_99,
        'survival_rate': survival_rate,
        'max_drawdown': max_drawdown,
        'expected_value': expected_value,
        'worst_case': worst_case,
        'median_value': np.median(final_values)
    }


def plot_simulation_paths(portfolio_values, scenario_name, metrics):
    """Create visualization of simulation paths"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Sample paths
    ax1 = axes[0]
    sample_indices = np.random.choice(len(portfolio_values), 100, replace=False)

    for idx in sample_indices:
        ax1.plot(portfolio_values[idx], alpha=0.1, color='blue')

    # Percentiles
    p5 = np.percentile(portfolio_values, 5, axis=0)
    p50 = np.percentile(portfolio_values, 50, axis=0)
    p95 = np.percentile(portfolio_values, 95, axis=0)

    ax1.plot(p5, 'r--', label='5th Percentile', linewidth=2)
    ax1.plot(p50, 'g-', label='Median', linewidth=2)
    ax1.plot(p95, 'b--', label='95th Percentile', linewidth=2)
    ax1.axhline(INITIAL_CAPITAL, color='black', linestyle=':', label='Initial Capital')
    ax1.axhline(INITIAL_CAPITAL * RUIN_THRESHOLD, color='red', linestyle=':', label='Ruin Threshold')

    ax1.set_title(f'{scenario_name} - Portfolio Simulations (100 sample paths)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final value distribution
    ax2 = axes[1]
    final_values = portfolio_values[:, -1]

    ax2.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(INITIAL_CAPITAL, color='black', linestyle=':', linewidth=2, label='Initial Capital')
    ax2.axvline(metrics['median_value'], color='green', linestyle='--', linewidth=2, label=f"Median: ${metrics['median_value']:.0f}")
    ax2.axvline(INITIAL_CAPITAL * RUIN_THRESHOLD, color='red', linestyle=':', linewidth=2, label=f"Ruin: ${INITIAL_CAPITAL * RUIN_THRESHOLD:.0f}")

    ax2.set_title('Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Final Portfolio Value ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = RESULTS_DIR / f"{scenario_name.replace(' ', '_').lower()}_simulation.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def run_monte_carlo(scenario_name, scenario_params, price_data):
    """Run Monte Carlo simulation for one scenario"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_params['name']}")
    print(f"{'='*70}")

    # Calculate historical parameters
    mu, sigma, corr_matrix = calculate_portfolio_params(price_data)

    print(f"\n📊 Historical Parameters (Daily):")
    print(f"   Mean Returns: {mu}")
    print(f"   Volatilities: {sigma}")
    print(f"\n📈 Scenario Adjustments:")
    print(f"   Drift Multiplier: {scenario_params['drift_multiplier']}")
    print(f"   Vol Multiplier: {scenario_params['vol_multiplier']}x")
    print(f"   Correlation Boost: {scenario_params['correlation_boost']}")

    # Generate paths
    print(f"\n🎲 Generating {NUM_SIMULATIONS:,} simulation paths...")
    paths = generate_correlated_paths(
        mu, sigma, corr_matrix,
        scenario_params,
        NUM_SIMULATIONS,
        FORECAST_DAYS
    )

    # Apply stop-loss
    print(f"🛡️  Applying {STOP_LOSS_PCT*100}% stop-loss...")
    stopped_paths, stops_triggered = apply_stop_loss(paths, STOP_LOSS_PCT)

    stop_rate = stops_triggered.mean() * 100
    print(f"   Stop-loss triggered: {stop_rate:.1f}% of positions")

    # Calculate portfolio returns
    position_sizes = np.array([POSITION_SIZE] * len(CORE_STOCKS))
    portfolio_values = calculate_portfolio_returns(stopped_paths, position_sizes)

    # Calculate metrics
    print(f"\n📉 Calculating risk metrics...")
    metrics = calculate_risk_metrics(portfolio_values, INITIAL_CAPITAL)

    # Plot
    plot_file = plot_simulation_paths(portfolio_values, scenario_params['name'], metrics)
    print(f"📊 Visualization saved: {plot_file.name}")

    return metrics, portfolio_values


def main():
    """Run Monte Carlo stress test"""
    print("="*70)
    print("PHASE 17: MONTE CARLO & STRESS TESTING")
    print("="*70)
    print(f"\n💼 Portfolio: {', '.join(CORE_STOCKS)}")
    print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"📊 Simulations: {NUM_SIMULATIONS:,}")
    print(f"📅 Forecast Period: {FORECAST_DAYS} days")
    print(f"🛡️  Stop-Loss: {STOP_LOSS_PCT*100}%")
    print(f"☠️  Ruin Threshold: {RUIN_THRESHOLD*100}% capital loss\n")

    # Fetch historical data
    print("📥 Fetching historical data...")
    price_data = pd.DataFrame()

    for ticker in CORE_STOCKS:
        print(f"   {ticker}...", end=" ")
        prices = fetch_historical_data(ticker)
        if prices is not None:
            price_data[ticker] = prices
            print("✅")
        else:
            print("❌")

    if price_data.empty:
        print("\n❌ Failed to fetch data")
        return

    # Align dates
    price_data = price_data.dropna()
    print(f"\n✅ Data loaded: {len(price_data)} days")

    # Run scenarios
    results = {}

    for scenario_key, scenario_params in SCENARIOS.items():
        metrics, portfolio_values = run_monte_carlo(scenario_key, scenario_params, price_data)
        results[scenario_key] = {
            'metrics': metrics,
            'portfolio_values': portfolio_values
        }

    # Generate comparison report
    print("\n\n" + "="*70)
    print("STRESS TEST RESULTS - SCENARIO COMPARISON")
    print("="*70)

    comparison_df = pd.DataFrame({
        scenario: res['metrics']
        for scenario, res in results.items()
    }).T

    print(f"\n{comparison_df.to_string()}\n")

    # Critical assessment
    print("\n" + "="*70)
    print("CRITICAL ASSESSMENT")
    print("="*70)

    bubble_metrics = results['BUBBLE_BURST']['metrics']
    normal_metrics = results['NORMAL']['metrics']

    print(f"\n🎯 AI BUBBLE BURST SCENARIO:")
    print(f"   Probability of Ruin:  {bubble_metrics['prob_ruin']:.1f}%")
    print(f"   VaR (99%):            ${bubble_metrics['var_99']:,.2f}")
    print(f"   Survival Rate:        {bubble_metrics['survival_rate']:.1f}%")
    print(f"   Max Drawdown:         {bubble_metrics['max_drawdown']:.1f}%")
    print(f"   Expected Final Value: ${bubble_metrics['expected_value']:,.2f}")
    print(f"   Worst Case (1%):      ${bubble_metrics['worst_case']:,.2f}")

    print(f"\n📊 NORMAL MARKET SCENARIO:")
    print(f"   Probability of Ruin:  {normal_metrics['prob_ruin']:.1f}%")
    print(f"   Expected Final Value: ${normal_metrics['expected_value']:,.2f}")
    print(f"   Survival Rate:        {normal_metrics['survival_rate']:.1f}%")

    # Risk assessment
    print(f"\n\n{'='*70}")
    print("RISK MANAGEMENT VERDICT")
    print(f"{'='*70}")

    if bubble_metrics['prob_ruin'] < 10:
        print("✅ EXCELLENT: Low ruin probability even in crash")
        verdict = "APPROVED"
    elif bubble_metrics['prob_ruin'] < 25:
        print("⚠️  MODERATE: Some ruin risk in extreme crash")
        verdict = "CAUTION"
    else:
        print("❌ HIGH RISK: Significant ruin probability")
        verdict = "REJECTED"

    print(f"\n🛡️  Stop-Loss Effectiveness:")
    if bubble_metrics['var_99'] > -INITIAL_CAPITAL * 0.3:
        print("✅ Stop-loss limits losses to acceptable levels")
    else:
        print("⚠️  Stops may not be sufficient in crash scenario")

    print(f"\n💰 Expected Outcome (Bubble Burst):")
    if bubble_metrics['expected_value'] > INITIAL_CAPITAL:
        print(f"✅ Portfolio EXPECTED to survive (+${bubble_metrics['expected_value']-INITIAL_CAPITAL:,.0f})")
    else:
        print(f"❌ Portfolio EXPECTED to lose (-${INITIAL_CAPITAL-bubble_metrics['expected_value']:,.0f})")

    print(f"\n\n{'='*70}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*70}")

    if verdict == "APPROVED":
        print("\n✅ System is STRESS-TESTED and ready for live deployment")
        print("   Risk controls are adequate for crash scenarios")
    elif verdict == "CAUTION":
        print("\n⚠️  System may survive but with significant risk")
        print("   Consider: Lower position sizes, tighter stops")
    else:
        print("\n❌ System is NOT ready for extreme scenarios")
        print("   Recommendation: Increase cash reserves, reduce leverage")

    # Save results
    summary_path = RESULTS_DIR / "monte_carlo_summary.csv"
    comparison_df.to_csv(summary_path)
    print(f"\n💾 Results saved: {summary_path}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
