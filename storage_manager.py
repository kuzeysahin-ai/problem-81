"""
PHASE 13: GOOGLE SHEETS PERSISTENCE LAYER
==========================================

Hybrid Storage System: Cloud-first with local fallback

ARCHITECTURE:
  1. Check for Google Sheets credentials in st.secrets
  2. If available: Use Google Sheets (Cloud Mode)
  3. If not available or error: Use local CSV (Fallback Mode)

STORAGE MODES:
  - CLOUD MODE: Data persists on Google Sheets (survives Streamlit restarts)
  - FALLBACK MODE: Data stored locally (for development/testing)

SECURITY:
  - NO credentials hardcoded
  - Uses Streamlit Secrets management
  - Credentials NEVER committed to Git
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import streamlit as st
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


class StorageManager:
    """
    Hybrid storage manager with Google Sheets persistence

    Automatically detects environment and chooses storage backend:
    - Streamlit Cloud → Google Sheets
    - Local Development → CSV files
    """

    def __init__(self):
        """Initialize storage manager with automatic mode detection"""
        self.mode = None
        self.gspread_client = None
        self.spreadsheet = None
        self.connection_error = None  # Store error message for debugging

        # Local CSV paths (fallback)
        self.project_root = Path(__file__).parent
        self.positions_file = self.project_root / "data" / "current_positions.csv"
        self.trade_history_file = self.project_root / "data" / "trade_history.csv"

        # Trading universe
        self.trading_stocks = ['GOOG', 'XOM', 'NVDA', 'JPM', 'KO']

        # Initialize storage
        self._initialize_storage()

    def _initialize_storage(self):
        """Detect environment and initialize appropriate storage backend"""
        try:
            # Try to initialize Google Sheets
            if self._init_google_sheets():
                self.mode = "CLOUD"
                self.connection_error = None
                print("✅ Storage Mode: GOOGLE SHEETS (Cloud)")
            else:
                self.mode = "FALLBACK"
                # connection_error already set by _init_google_sheets
                print("⚠️  Storage Mode: LOCAL CSV (Fallback)")
        except Exception as e:
            self.mode = "FALLBACK"
            self.connection_error = f"Initialization error: {type(e).__name__}: {str(e)}"
            print(f"⚠️  Storage Mode: LOCAL CSV (Fallback) - {str(e)}")

    def _init_google_sheets(self):
        """
        Initialize Google Sheets connection using Streamlit secrets

        Required secrets in .streamlit/secrets.toml:
        [gcp_service_account]
        type = "service_account"
        project_id = "your-project-id"
        private_key_id = "..."
        private_key = "..."
        client_email = "..."
        client_id = "..."
        auth_uri = "https://accounts.google.com/o/oauth2/auth"
        token_uri = "https://oauth2.googleapis.com/token"
        auth_provider_x509_cert_url = "..."
        client_x509_cert_url = "..."

        [sheets]
        spreadsheet_id = "your-spreadsheet-id"
        """
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            # Check if secrets exist
            if "gcp_service_account" not in st.secrets:
                self.connection_error = "Missing 'gcp_service_account' in Streamlit secrets"
                return False

            if "sheets" not in st.secrets:
                self.connection_error = "Missing 'sheets' section in Streamlit secrets"
                return False

            # Setup credentials
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]

            credentials = ServiceAccountCredentials.from_json_keyfile_dict(
                st.secrets["gcp_service_account"],
                scope
            )

            # Authorize and open spreadsheet
            self.gspread_client = gspread.authorize(credentials)
            spreadsheet_id = st.secrets["sheets"]["spreadsheet_id"]
            self.spreadsheet = self.gspread_client.open_by_key(spreadsheet_id)

            # Verify worksheets exist (create if needed)
            self._ensure_worksheets_exist()

            return True

        except ImportError as e:
            # gspread not installed
            self.connection_error = f"Missing library: {str(e)}. Run 'pip install gspread oauth2client'"
            return False
        except KeyError as e:
            # Missing key in secrets
            self.connection_error = f"Missing secret key: {str(e)}"
            return False
        except Exception as e:
            # Other errors (API errors, auth errors, etc.)
            error_type = type(e).__name__
            self.connection_error = f"{error_type}: {str(e)}"
            print(f"Google Sheets init failed: {error_type}: {e}")
            return False

    def _ensure_worksheets_exist(self):
        """Ensure required worksheets exist in spreadsheet"""
        try:
            worksheets = [ws.title for ws in self.spreadsheet.worksheets()]

            # Create 'positions' worksheet if needed
            if 'positions' not in worksheets:
                worksheet = self.spreadsheet.add_worksheet(title='positions', rows=100, cols=10)
                # Add headers
                worksheet.update('A1:E1', [[
                    'ticker', 'has_position', 'entry_price', 'entry_time', 'highest_price'
                ]])
                # Add initial data
                for i, ticker in enumerate(self.trading_stocks, start=2):
                    worksheet.update(f'A{i}:E{i}', [[ticker, 'False', '0.0', '', '0.0']])

            # Create 'trade_history' worksheet if needed
            if 'trade_history' not in worksheets:
                worksheet = self.spreadsheet.add_worksheet(title='trade_history', rows=1000, cols=10)
                # Add headers
                worksheet.update('A1:H1', [[
                    'ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                    'entry_time', 'exit_time', 'hold_duration'
                ]])

        except Exception as e:
            print(f"Error ensuring worksheets: {e}")
            raise

    # ========================================================================
    # PUBLIC API: Position Management
    # ========================================================================

    def load_positions(self):
        """
        Load current positions

        Returns:
            pd.DataFrame with columns: ticker, has_position, entry_price, entry_time, highest_price
        """
        if self.mode == "CLOUD":
            return self._load_positions_cloud()
        else:
            return self._load_positions_local()

    def save_positions(self, positions_df):
        """
        Save positions

        Args:
            positions_df: DataFrame with position data
        """
        if self.mode == "CLOUD":
            self._save_positions_cloud(positions_df)
        else:
            self._save_positions_local(positions_df)

    def enter_position(self, ticker, price):
        """
        Enter a new position

        Args:
            ticker: Stock ticker
            price: Entry price
        """
        positions = self.load_positions()
        idx = positions[positions['ticker'] == ticker].index[0]

        positions.at[idx, 'has_position'] = True
        positions.at[idx, 'entry_price'] = price
        positions.at[idx, 'entry_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        positions.at[idx, 'highest_price'] = price

        self.save_positions(positions)

    def exit_position(self, ticker, exit_price):
        """
        Exit a position and log trade

        Args:
            ticker: Stock ticker
            exit_price: Exit price

        Returns:
            pnl: Profit/loss amount
        """
        positions = self.load_positions()
        idx = positions[positions['ticker'] == ticker].index[0]

        # Get position details before clearing
        entry_price = float(positions.at[idx, 'entry_price'])
        entry_time = positions.at[idx, 'entry_time']

        # Calculate P&L
        pnl = exit_price - entry_price

        # Log trade to history
        self.log_trade(ticker, entry_price, entry_time, exit_price)

        # Clear position
        positions.at[idx, 'has_position'] = False
        positions.at[idx, 'entry_price'] = 0.0
        positions.at[idx, 'entry_time'] = ''
        positions.at[idx, 'highest_price'] = 0.0

        self.save_positions(positions)

        return pnl

    def update_highest_price(self, ticker, current_price):
        """
        Update highest price for trailing stop

        Args:
            ticker: Stock ticker
            current_price: Current market price
        """
        positions = self.load_positions()
        idx = positions[positions['ticker'] == ticker].index[0]

        if positions.at[idx, 'has_position']:
            highest = float(positions.at[idx, 'highest_price'])
            if current_price > highest:
                positions.at[idx, 'highest_price'] = current_price
                self.save_positions(positions)

    # ========================================================================
    # PUBLIC API: Trade History
    # ========================================================================

    def load_trade_history(self):
        """
        Load trade history

        Returns:
            pd.DataFrame with trade history
        """
        if self.mode == "CLOUD":
            return self._load_history_cloud()
        else:
            return self._load_history_local()

    def log_trade(self, ticker, entry_price, entry_time, exit_price):
        """
        Log completed trade to history

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            entry_time: Entry timestamp
            exit_price: Exit price
        """
        # Calculate metrics
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Calculate hold duration
        try:
            entry_dt = pd.to_datetime(entry_time)
            exit_dt = pd.to_datetime(exit_time)
            hold_duration = str(exit_dt - entry_dt)
        except:
            hold_duration = 'N/A'

        # Create trade record
        trade = {
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'hold_duration': hold_duration
        }

        if self.mode == "CLOUD":
            self._log_trade_cloud(trade)
        else:
            self._log_trade_local(trade)

    # ========================================================================
    # PRIVATE: Local CSV Operations
    # ========================================================================

    def _load_positions_local(self):
        """Load positions from local CSV"""
        if not self.positions_file.exists():
            df = pd.DataFrame([
                {'ticker': t, 'has_position': False, 'entry_price': 0.0,
                 'entry_time': '', 'highest_price': 0.0}
                for t in self.trading_stocks
            ])
            df.to_csv(self.positions_file, index=False)
            return df

        df = pd.read_csv(self.positions_file)

        # Backward compatibility: add highest_price if missing
        if 'highest_price' not in df.columns:
            df['highest_price'] = 0.0
            df.to_csv(self.positions_file, index=False)

        # CRITICAL FIX: Ensure has_position is boolean type
        # Pandas may read it as string or object type from CSV
        if df['has_position'].dtype != bool:
            df['has_position'] = df['has_position'].astype(bool)

        return df

    def _save_positions_local(self, positions_df):
        """Save positions to local CSV"""
        positions_df.to_csv(self.positions_file, index=False)

    def _load_history_local(self):
        """Load trade history from local CSV"""
        if not self.trade_history_file.exists():
            df = pd.DataFrame(columns=[
                'ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                'entry_time', 'exit_time', 'hold_duration'
            ])
            df.to_csv(self.trade_history_file, index=False)
            return df

        return pd.read_csv(self.trade_history_file)

    def _log_trade_local(self, trade):
        """Log trade to local CSV"""
        history = self._load_history_local()
        new_trade = pd.DataFrame([trade])
        history = pd.concat([history, new_trade], ignore_index=True)
        history.to_csv(self.trade_history_file, index=False)

    # ========================================================================
    # PRIVATE: Google Sheets Operations
    # ========================================================================

    def _load_positions_cloud(self):
        """Load positions from Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet('positions')
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)

            # CRITICAL FIX: Convert has_position to boolean safely
            # Handle multiple possible formats: 'True', 'False', 'nan', True, False, NaN
            def safe_bool_convert(val):
                """Safely convert various formats to boolean"""
                if pd.isna(val):
                    return False
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    if val_lower in ('true', '1', 'yes'):
                        return True
                    elif val_lower in ('false', '0', 'no', 'nan', ''):
                        return False
                # Fallback: try numeric conversion
                try:
                    return bool(float(val))
                except:
                    return False

            df['has_position'] = df['has_position'].apply(safe_bool_convert)

            # Convert string numbers to float
            df['entry_price'] = df['entry_price'].astype(float)
            df['highest_price'] = df['highest_price'].astype(float)

            return df

        except Exception as e:
            print(f"Error loading from Google Sheets: {e}")
            # Fallback to local
            return self._load_positions_local()

    def _save_positions_cloud(self, positions_df):
        """Save positions to Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet('positions')

            # CRITICAL FIX: Update each cell individually to avoid locale issues
            # Google Sheets with Turkish locale has decimal separator problems
            # Solution: Update cells one-by-one, ensuring float values are preserved

            for i, (idx, row) in enumerate(positions_df.iterrows(), start=2):  # Start from row 2 (skip header)
                # Update ticker (A column)
                worksheet.update_cell(i, 1, str(row['ticker']))

                # Update has_position (B column) - CRITICAL: Ensure proper boolean format
                # Convert boolean to string explicitly to avoid pandas/gspread conversion issues
                has_pos = row['has_position']
                if pd.isna(has_pos):
                    has_pos_str = 'False'
                elif isinstance(has_pos, bool):
                    has_pos_str = 'True' if has_pos else 'False'
                else:
                    # Fallback: try to interpret as boolean
                    has_pos_str = 'True' if has_pos else 'False'
                worksheet.update_cell(i, 2, has_pos_str)

                # Update entry_price (C column) - CRITICAL: Send as number, not string
                entry_price_val = float(row['entry_price'])
                worksheet.update_cell(i, 3, entry_price_val)

                # Update entry_time (D column)
                worksheet.update_cell(i, 4, str(row['entry_time']))

                # Update highest_price (E column) - CRITICAL: Send as number, not string
                highest_price_val = float(row['highest_price'])
                worksheet.update_cell(i, 5, highest_price_val)

        except Exception as e:
            print(f"Error saving to Google Sheets: {e}")
            # Fallback to local
            self._save_positions_local(positions_df)

    def _load_history_cloud(self):
        """Load trade history from Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet('trade_history')
            data = worksheet.get_all_records()

            if not data:
                return pd.DataFrame(columns=[
                    'ticker', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                    'entry_time', 'exit_time', 'hold_duration'
                ])

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            print(f"Error loading history from Google Sheets: {e}")
            # Fallback to local
            return self._load_history_local()

    def _log_trade_cloud(self, trade):
        """Log trade to Google Sheets"""
        try:
            worksheet = self.spreadsheet.worksheet('trade_history')

            # Append row
            row = [
                trade['ticker'],
                trade['entry_price'],
                trade['exit_price'],
                trade['pnl'],
                trade['pnl_pct'],
                trade['entry_time'],
                trade['exit_time'],
                trade['hold_duration']
            ]

            worksheet.append_row(row)

        except Exception as e:
            print(f"Error logging to Google Sheets: {e}")
            # Fallback to local
            self._log_trade_local(trade)

    # ========================================================================
    # UTILITY
    # ========================================================================

    def get_mode(self):
        """Get current storage mode"""
        return self.mode

    def get_connection_error(self):
        """Get connection error message (None if no error)"""
        return self.connection_error

    def test_connection(self):
        """Test storage connection"""
        try:
            positions = self.load_positions()
            return True, f"✅ Storage OK ({self.mode}): {len(positions)} positions loaded"
        except Exception as e:
            return False, f"❌ Storage Error: {str(e)}"
