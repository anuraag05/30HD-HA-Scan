# NSE Heikin-Ashi Scanner (Dhan + Streamlit)

Scans NSE equities on **1D, 30m and 1H**, shows whether the latest Heikin-Ashi
candle is positive/negative, and flags **YES** when the Heikin-Ashi candle and the
regular candle are the same colour. Includes summary and per-stock candlestick charts.

## Run locally
```bash
pip install -r requirements.txt
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
DHAN_CLIENT_ID = "your-client-id"
DHAN_ACCESS_TOKEN = "your-access-token"
EOF
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (`secrets.toml` is git-ignored, never commit it).
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch `main`, file `app.py`.
3. In Advanced settings -> Secrets paste:
   ```toml
   DHAN_CLIENT_ID = "your-client-id"
   DHAN_ACCESS_TOKEN = "your-access-token"
   ```
4. Deploy. Dhan access tokens expire (24h unless you generate a longer-lived one), so update the secret when it does.

## Notes
- Dhan has no native 30-minute interval; 30m candles are built from 15-minute data.
- Dhan data API is rate limited (~5 req/s). Nifty 50/100/500 are quick; "All NSE EQ" takes 25+ minutes.
- Run after market close for completed candles; during market hours the latest bar is still forming.
