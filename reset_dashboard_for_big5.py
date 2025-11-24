"""
RESET DASHBOARD FOR THE BIG 5
==============================

This script prepares the dashboard for live trading with THE BIG 5:
  - GOOG, XOM, NVDA, JPM, KO

Actions:
  1. Reset current_positions.csv with The Big 5
  2. Clear trade_history.csv (fresh start)
  3. Update TRADING_STOCKS in dashboard.py

USE THIS ONLY AFTER:
  - finalize_core_portfolio.py shows positive results
  - CFO approval received
  - You're ready for LIVE deployment

⚠️  WARNING: This will erase existing trade history!
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
POSITIONS_FILE = DATA_DIR / "current_positions.csv"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history.csv"
DASHBOARD_FILE = PROJECT_ROOT / "dashboard.py"

# THE BIG 5
CORE_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']


def backup_files():
    """Create backups of existing files"""
    print("\n📦 Creating backups...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = DATA_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    files_to_backup = [POSITIONS_FILE, TRADE_HISTORY_FILE]

    for file in files_to_backup:
        if file.exists():
            backup_path = backup_dir / file.name
            shutil.copy2(file, backup_path)
            print(f"  ✅ Backed up: {file.name} -> {backup_path}")

    print(f"\n📂 Backups saved to: {backup_dir}")


def reset_positions():
    """Reset current_positions.csv with The Big 5"""
    print("\n🔄 Resetting current_positions.csv...")

    positions_df = pd.DataFrame([
        {'ticker': ticker, 'has_position': False, 'entry_price': 0.0, 'entry_time': ''}
        for ticker in CORE_STOCKS
    ])

    positions_df.to_csv(POSITIONS_FILE, index=False)
    print(f"  ✅ Created new positions file with {len(CORE_STOCKS)} stocks:")
    print(f"     {', '.join(CORE_STOCKS)}")


def reset_trade_history():
    """Clear trade_history.csv"""
    print("\n🗑️  Clearing trade history...")

    history_df = pd.DataFrame(columns=[
        'ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
        'entry_time', 'exit_time', 'hold_duration'
    ])

    history_df.to_csv(TRADE_HISTORY_FILE, index=False)
    print("  ✅ Trade history cleared (fresh start)")


def update_dashboard_stocks():
    """Update TRADING_STOCKS in dashboard.py"""
    print("\n📝 Updating dashboard.py...")

    # Read dashboard
    with open(DASHBOARD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace TRADING_STOCKS
    old_line = "TRADING_STOCKS = ['GOOG', 'XOM', 'NVDA', 'CAT']"
    new_line = f"TRADING_STOCKS = {CORE_STOCKS}"

    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"  ✅ Updated TRADING_STOCKS:")
        print(f"     OLD: {old_line}")
        print(f"     NEW: {new_line}")
    else:
        # Try to find any TRADING_STOCKS line
        import re
        pattern = r"TRADING_STOCKS = \[.*?\]"
        match = re.search(pattern, content)

        if match:
            old_value = match.group(0)
            content = re.sub(pattern, new_line, content)
            print(f"  ✅ Updated TRADING_STOCKS:")
            print(f"     OLD: {old_value}")
            print(f"     NEW: {new_line}")
        else:
            print("  ⚠️  Could not find TRADING_STOCKS in dashboard.py")
            print("     You may need to update it manually.")
            return

    # Write back
    with open(DASHBOARD_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print("  ✅ dashboard.py updated")


def verify_setup():
    """Verify the new setup"""
    print("\n\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check positions file
    positions = pd.read_csv(POSITIONS_FILE)
    print(f"\n✅ current_positions.csv:")
    print(positions.to_string(index=False))

    # Check history file
    history = pd.read_csv(TRADE_HISTORY_FILE)
    print(f"\n✅ trade_history.csv: {len(history)} trades (should be 0)")

    # Check dashboard
    with open(DASHBOARD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if 'TRADING_STOCKS' in line and '=' in line:
                print(f"\n✅ dashboard.py TRADING_STOCKS:")
                print(f"   {line.strip()}")
                break

    print("\n" + "=" * 70)
    print("SETUP COMPLETE - READY FOR LIVE TRADING")
    print("=" * 70)


def main():
    """Reset dashboard for The Big 5"""
    print("=" * 70)
    print("RESET DASHBOARD FOR THE BIG 5")
    print("=" * 70)
    print(f"\nThe Big 5: {', '.join(CORE_STOCKS)}")

    # Confirm
    print("\n⚠️  WARNING: This will:")
    print("   1. Replace current positions with The Big 5")
    print("   2. Clear all trade history")
    print("   3. Update dashboard.py")
    print("\nExisting files will be backed up.")

    response = input("\nProceed? (yes/no): ").strip().lower()

    if response != 'yes':
        print("\n❌ Operation cancelled.")
        return

    # Execute
    backup_files()
    reset_positions()
    reset_trade_history()
    update_dashboard_stocks()
    verify_setup()

    print("\n\n🎯 Next Steps:")
    print("   1. Run: streamlit run dashboard.py")
    print("   2. Verify The Big 5 appear in Live Trading tab")
    print("   3. Start trading!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
