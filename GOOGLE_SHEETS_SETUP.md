# PHASE 13: GOOGLE SHEETS PERSISTENCE SETUP
==========================================

This guide explains how to configure Google Sheets persistence for Streamlit Cloud deployment.

## 🎯 Why Google Sheets?

**Problem:** Streamlit Cloud restarts containers, wiping out local CSV files.

**Solution:** Store trading data in Google Sheets for permanent persistence.

## 📋 Setup Steps

### 1. Create Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable **Google Sheets API**:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

4. Create Service Account:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Name: `trading-bot-storage`
   - Click "Create and Continue"
   - Skip role assignment (click "Continue")
   - Click "Done"

5. Create JSON Key:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" → "Create New Key"
   - Choose "JSON"
   - **DOWNLOAD THE JSON FILE** (keep it safe!)

### 2. Create Google Spreadsheet

1. Go to [Google Sheets](https://sheets.google.com/)
2. Create a new spreadsheet
3. Name it: `Trading Bot - Live Positions`
4. **Share with Service Account:**
   - Click "Share" button
   - Paste the service account email (from JSON: `client_email`)
   - Grant "Editor" access
   - Uncheck "Notify people"
   - Click "Share"

5. **Copy Spreadsheet ID:**
   - Look at the URL: `https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`
   - Copy the `SPREADSHEET_ID` part

### 3. Configure Streamlit Secrets

#### For Local Development:

Create `.streamlit/secrets.toml` in your project:

```toml
# PHASE 13: Google Sheets Persistence
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "abc123..."
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
client_email = "trading-bot-storage@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."

[sheets]
spreadsheet_id = "YOUR_SPREADSHEET_ID_HERE"
```

**IMPORTANT:** Copy values from the downloaded JSON file.

**⚠️ SECURITY:** Add `.streamlit/` to `.gitignore` to prevent committing secrets!

#### For Streamlit Cloud:

1. Go to your Streamlit Cloud app settings
2. Navigate to "Secrets" section
3. Paste the same TOML content
4. Click "Save"

### 4. Verify Setup

Run the test script:

```bash
python -c "from storage_manager import StorageManager; sm = StorageManager(); print(sm.get_mode())"
```

**Expected output:**
- Local without secrets: `FALLBACK` (uses CSV)
- Local with secrets: `CLOUD` (uses Google Sheets)
- Streamlit Cloud: `CLOUD` (uses Google Sheets)

## 🔄 How It Works

### Hybrid Storage Architecture:

```
┌─────────────────────────────────────┐
│      StorageManager.__init__()      │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Check for st.secrets  │
    │ "gcp_service_account" │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
     Found          Not Found
        │             │
        ▼             ▼
  ┌──────────┐   ┌──────────┐
  │  CLOUD   │   │ FALLBACK │
  │  MODE    │   │   MODE   │
  └────┬─────┘   └────┬─────┘
       │              │
       ▼              ▼
  Google Sheets   Local CSV
  (Persistent)    (Temporary)
```

### Storage Operations:

| Operation | Cloud Mode | Fallback Mode |
|-----------|------------|---------------|
| `load_positions()` | Read from Google Sheets | Read from CSV |
| `save_positions()` | Write to Google Sheets | Write to CSV |
| `enter_position()` | Update Google Sheets | Update CSV |
| `exit_position()` | Update Google Sheets + log trade | Update CSV + log trade |
| `load_trade_history()` | Read from Google Sheets | Read from CSV |

## 📊 Google Sheets Structure

### Worksheet 1: `positions`

| ticker | has_position | entry_price | entry_time | highest_price |
|--------|--------------|-------------|------------|---------------|
| GOOG   | False        | 0.0         |            | 0.0           |
| XOM    | False        | 0.0         |            | 0.0           |
| NVDA   | False        | 0.0         |            | 0.0           |
| JPM    | False        | 0.0         |            | 0.0           |
| KO     | False        | 0.0         |            | 0.0           |

### Worksheet 2: `trade_history`

| ticker | entry_price | exit_price | pnl | pnl_pct | entry_time | exit_time | hold_duration |
|--------|-------------|------------|-----|---------|------------|-----------|---------------|
| NVDA   | 145.30      | 148.75     | 3.45| 2.37    | 2025-01-15...| 2025-01-15...| 0:03:00 |

## 🚨 Troubleshooting

### "Storage Mode: FALLBACK" on Streamlit Cloud

**Cause:** Secrets not configured correctly.

**Fix:**
1. Check Streamlit Cloud → App Settings → Secrets
2. Verify all fields match the JSON file
3. Ensure `private_key` includes `\n` for newlines
4. Restart the app

### "Permission Denied" error

**Cause:** Service account not shared with spreadsheet.

**Fix:**
1. Open Google Sheet
2. Click "Share"
3. Add service account email
4. Grant "Editor" access

### "Spreadsheet not found"

**Cause:** Wrong spreadsheet ID.

**Fix:**
1. Check spreadsheet URL
2. Copy ID from URL
3. Update `secrets.toml` → `sheets.spreadsheet_id`

## 🎯 Best Practices

1. **Never commit secrets to Git**
   - Always use `.gitignore` for `.streamlit/`
   - Use Streamlit Cloud secrets for deployment

2. **Backup your data**
   - Google Sheets auto-saves
   - Export as CSV periodically for backup

3. **Monitor storage mode**
   - Check sidebar in dashboard
   - ☁️ = Google Sheets (good for production)
   - 💾 = Local CSV (good for development)

4. **Test locally first**
   - Configure secrets locally
   - Verify cloud mode works
   - Then deploy to Streamlit Cloud

## 📚 References

- [Google Sheets API Python Quickstart](https://developers.google.com/sheets/api/quickstart/python)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [gspread Documentation](https://docs.gspread.org/)

---

**✅ Setup complete! Your trading data will now persist across Streamlit Cloud restarts.**
