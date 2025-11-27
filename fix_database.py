"""
DATABASE REPAIR SCRIPT
======================

One-time script to fix corrupted position data.

PROBLEM:
    - Entry prices accidentally saved as timestamps (e.g., NVDA = 1.8 billion)
    - Causes Risk Manager stop-loss calculation errors

SOLUTION:
    - Hard reset all positions to clean state
    - Works with both Google Sheets and Local CSV (via StorageManager)

USAGE:
    python fix_database.py
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Import StorageManager
from storage_manager import StorageManager

# The Big 5 Portfolio
TRADING_STOCKS = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']


def fix_corrupted_positions():
    """
    Reset all positions to clean state

    This fixes:
        - Corrupted entry_price values (timestamps instead of prices)
        - Invalid highest_price values
        - Orphaned position states
    """
    print("=" * 70)
    print("DATABASE REPAIR SCRIPT - PHASE 19")
    print("=" * 70)

    # Initialize Storage Manager
    print("\n🔌 Connecting to storage...")
    storage = StorageManager()
    print(f"   Storage Mode: {storage.get_mode()}")

    # Load current positions
    print("\n📥 Loading current positions...")
    positions = storage.load_positions()

    # Show BEFORE state
    print("\n" + "=" * 70)
    print("BEFORE REPAIR:")
    print("=" * 70)
    for _, row in positions.iterrows():
        ticker = row['ticker']
        has_pos = row['has_position']
        entry_price = row['entry_price']
        entry_time = row.get('entry_time', 'N/A')
        highest = row.get('highest_price', 'N/A')

        status = "🔴 ACTIVE" if has_pos else "⚪ EMPTY"
        print(f"{status} {ticker:5} | Entry: ${entry_price:,.2f} | Time: {entry_time} | Highest: {highest}")

    # Fix corrupted data
    print("\n" + "=" * 70)
    print("REPAIRING DATABASE:")
    print("=" * 70)

    repaired_count = 0

    for idx, row in positions.iterrows():
        ticker = row['ticker']
        old_entry = row['entry_price']
        old_has_pos = row['has_position']

        # Check if data is corrupted (entry price > $10,000 is suspicious)
        is_corrupted = old_entry > 10000 or (old_has_pos and old_entry == 0)

        if is_corrupted or old_has_pos:
            print(f"\n🔧 Fixing {ticker}:")
            print(f"   BEFORE: has_position={old_has_pos}, entry_price=${old_entry:,.2f}")

            # Hard reset
            positions.at[idx, 'has_position'] = False
            positions.at[idx, 'entry_price'] = 0.0
            positions.at[idx, 'entry_time'] = ''

            # Reset highest_price if column exists
            if 'highest_price' in positions.columns:
                positions.at[idx, 'highest_price'] = 0.0

            # Reset stop_loss_price if column exists
            if 'stop_loss_price' in positions.columns:
                positions.at[idx, 'stop_loss_price'] = 0.0

            print(f"   AFTER:  has_position=False, entry_price=$0.00")
            print(f"   ✅ {ticker} REPAIRED")

            repaired_count += 1
        else:
            print(f"✓ {ticker}: Already clean (no repair needed)")

    # Save repaired data
    print("\n" + "=" * 70)
    print("SAVING REPAIRED DATA:")
    print("=" * 70)

    storage.save_positions(positions)
    print(f"✅ Saved to {storage.get_mode()} storage")

    # Show AFTER state
    print("\n" + "=" * 70)
    print("AFTER REPAIR:")
    print("=" * 70)

    # Reload to verify
    positions_after = storage.load_positions()

    for _, row in positions_after.iterrows():
        ticker = row['ticker']
        has_pos = row['has_position']
        entry_price = row['entry_price']
        entry_time = row.get('entry_time', 'N/A')
        highest = row.get('highest_price', 'N/A')

        status = "🔴 ACTIVE" if has_pos else "✅ CLEAN"
        print(f"{status} {ticker:5} | Entry: ${entry_price:,.2f} | Time: {entry_time} | Highest: {highest}")

    # Summary
    print("\n" + "=" * 70)
    print("REPAIR SUMMARY:")
    print("=" * 70)
    print(f"✅ Repaired {repaired_count} position(s)")
    print(f"✅ Database is now CLEAN and READY")
    print(f"✅ Risk Manager will function correctly")
    print("\n💡 You can now safely run: streamlit run dashboard.py")


def verify_database():
    """Verify database integrity after repair"""
    print("\n" + "=" * 70)
    print("DATABASE INTEGRITY CHECK:")
    print("=" * 70)

    storage = StorageManager()
    positions = storage.load_positions()

    issues_found = 0

    for _, row in positions.iterrows():
        ticker = row['ticker']
        entry_price = row['entry_price']
        has_pos = row['has_position']

        # Check for suspicious values
        if entry_price > 10000:
            print(f"❌ {ticker}: Suspicious entry_price (${entry_price:,.2f})")
            issues_found += 1
        elif has_pos and entry_price == 0:
            print(f"⚠️  {ticker}: Active position with $0 entry (inconsistent)")
            issues_found += 1
        else:
            print(f"✅ {ticker}: OK")

    if issues_found == 0:
        print("\n✅ DATABASE INTEGRITY: PERFECT")
    else:
        print(f"\n⚠️  Found {issues_found} issue(s)")

    return issues_found == 0


if __name__ == "__main__":
    import sys

    # Check for --force flag
    force_mode = '--force' in sys.argv

    if not force_mode:
        print("\n🚨 WARNING: This script will RESET all active positions!")
        print("   All position data will be cleared.")
        print("   Trade history will NOT be affected.\n")

        response = input("Continue? (yes/no): ").strip().lower()
    else:
        print("\n🚨 FORCE MODE: Skipping confirmation (--force flag detected)")
        response = 'yes'

    if response == 'yes':
        print("\n🔧 Starting repair process...\n")

        try:
            fix_corrupted_positions()

            # Verify repair
            print("\n")
            if verify_database():
                print("\n" + "=" * 70)
                print("✅ REPAIR COMPLETE - DATABASE VERIFIED")
                print("=" * 70)
            else:
                print("\n" + "=" * 70)
                print("⚠️  REPAIR COMPLETE - Manual verification recommended")
                print("=" * 70)

        except Exception as e:
            print(f"\n❌ ERROR during repair: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print("\n❌ Repair cancelled by user")
        sys.exit(0)
