"""NSE Heikin-Ashi scanner (Dhan data) - Streamlit app.

Scans NSE equities on 1D, 30m and 1H and flags stocks where the latest
Heikin-Ashi candle and the latest regular candle have the same colour.
"""
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Heikin-Ashi Scanner", layout="wide")

API = "https://api.dhan.co/v2"
MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
NSE_LISTS = {
    "Nifty 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}
TFS = ["1D", "30m", "1H"]
IST = "Asia/Kolkata"


# ---------------------------------------------------------------- credentials
def get_creds():
    cid = st.secrets.get("DHAN_CLIENT_ID", "") if hasattr(st, "secrets") else ""
    tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") if hasattr(st, "secrets") else ""
    with st.sidebar:
        st.header("Dhan credentials")
        if cid and tok:
            st.success("Loaded from Streamlit secrets")
        else:
            cid = st.text_input("Client ID", type="password")
            tok = st.text_input("Access token", type="password")
    return cid, tok


# ---------------------------------------------------------------- rate limiter
class Limiter:
    def __init__(self, per_sec=4.0):
        self.gap, self.last, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            sleep = self.last + self.gap - now
            if sleep > 0:
                time.sleep(sleep)
            self.last = time.monotonic()


LIMITER = Limiter(2.5)  # Dhan data API allows 5 req/sec; stay well under it
ABORT = threading.Event()
THROTTLED = {"n": 0}
ERRS = {}
_ERR_LOCK = threading.Lock()


def note_err(msg):
    with _ERR_LOCK:
        ERRS[msg] = ERRS.get(msg, 0) + 1


# ---------------------------------------------------------------- universe
@st.cache_data(ttl=24 * 3600, show_spinner="Loading Dhan instrument master...")
def load_master():
    df = pd.read_csv(MASTER_URL, low_memory=False)
    df = df[(df["SEM_EXM_EXCH_ID"] == "NSE") & (df["SEM_SEGMENT"] == "E")]
    df = df[df["SEM_INSTRUMENT_NAME"] == "EQUITY"]
    if "SEM_SERIES" in df.columns:
        df = df[df["SEM_SERIES"] == "EQ"]
    out = df[["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]].drop_duplicates("SEM_TRADING_SYMBOL")
    out.columns = ["symbol", "security_id"]
    return out.reset_index(drop=True)


@st.cache_data(ttl=24 * 3600)
def load_index_symbols(name):
    r = requests.get(NSE_LISTS[name], headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))["Symbol"].str.strip().tolist()


# ---------------------------------------------------------------- data
def _to_df(js):
    if not js or "close" not in js or not js["close"]:
        return None
    ts = pd.Series(js["timestamp"], dtype="float64")
    if ts.iloc[0] < 946684800:  # some responses use a 1980 epoch
        ts = ts + 315513000
    idx = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(IST)
    df = pd.DataFrame(
        {k: js[k] for k in ("open", "high", "low", "close", "volume")}
    ).astype(float)
    df.index = idx
    return df


def _post(path, body, headers):
    for attempt in range(4):
        if ABORT.is_set():
            return None
        LIMITER.wait()
        try:
            r = requests.post(f"{API}{path}", json=body, headers=headers, timeout=25)
            if r.status_code == 200:
                THROTTLED["n"] = 0
                js = r.json()
                if not isinstance(js, dict) or not js.get("close"):
                    note_err(f"HTTP 200 but no candle data: {str(js)[:160]}")
                return js
            if r.status_code == 429:
                note_err("HTTP 429 DH-904 rate limit (backing off)")
                with _ERR_LOCK:
                    THROTTLED["n"] += 1
                    if THROTTLED["n"] >= 10:
                        ABORT.set()
                with LIMITER.lock:  # every worker waits behind this pause
                    LIMITER.last = max(LIMITER.last, time.monotonic() + 5 * (attempt + 1))
                continue
            note_err(f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        except (requests.RequestException, ValueError) as e:
            note_err(f"Request error: {e}")
            time.sleep(1)
    return None


def fetch_raw(headers, sec_id, kind):
    """kind 'D' = daily candles, '15' = 15-minute candles (all other TFs are built from these)."""
    now = datetime.now()
    base = {"securityId": str(sec_id), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"}
    if kind == "D":
        body = {**base, "expiryCode": 0, "oi": False,
                "fromDate": (now - timedelta(days=150)).strftime("%Y-%m-%d"),
                "toDate": (now + timedelta(days=1)).strftime("%Y-%m-%d")}
        return _to_df(_post("/charts/historical", body, headers))
    body = {**base, "interval": "15", "oi": False,
            "fromDate": (now - timedelta(days=20)).strftime("%Y-%m-%d 09:15:00"),
            "toDate": now.strftime("%Y-%m-%d 15:30:00")}
    return _to_df(_post("/charts/intraday", body, headers))


def resample(df, rule):
    # sessions start 09:15, so shift bin edges by 15 minutes
    return df.resample(rule, offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def get_frames(headers, sec_id):
    """Two API calls per stock -> 1D, 30m and 1H frames."""
    d = fetch_raw(headers, sec_id, "D")
    m15 = fetch_raw(headers, sec_id, "15")
    return {"1D": d,
            "30m": resample(m15, "30min") if m15 is not None else None,
            "1H": resample(m15, "60min") if m15 is not None else None}


@st.cache_data(ttl=300, show_spinner="Fetching candles...")
def cached_frames(cid, tok, sid):
    h = {"access-token": tok, "client-id": cid, "Content-Type": "application/json",
         "Accept": "application/json"}
    return get_frames(h, sid)


def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df.open + df.high + df.low + df.close) / 4
    opens = [(df.open.iloc[0] + df.close.iloc[0]) / 2]
    for i in range(1, len(df)):
        opens.append((opens[-1] + ha.close.iloc[i - 1]) / 2)
    ha["open"] = opens
    ha["high"] = pd.concat([df.high, ha.open, ha.close], axis=1).max(axis=1)
    ha["low"] = pd.concat([df.low, ha.open, ha.close], axis=1).min(axis=1)
    return ha


def classify(df):
    """Latest candle: HA colour, regular colour, and whether they agree."""
    if df is None or len(df) < 3:
        return "n/a", "n/a", "n/a"
    ha = heikin_ashi(df)
    ha_c = "Positive" if ha.close.iloc[-1] > ha.open.iloc[-1] else "Negative"
    rg_c = "Positive" if df.close.iloc[-1] > df.open.iloc[-1] else "Negative"
    both = "No"
    if ha_c == rg_c == "Positive":
        both = "YES - Positive"
    elif ha_c == rg_c == "Negative":
        both = "YES - Negative"
    return ha_c, rg_c, both


def scan_one(headers, symbol, sec_id):
    row = {"Symbol": symbol}
    frames = get_frames(headers, sec_id)
    for tf in TFS:
        ha_c, rg_c, both = classify(frames[tf])
        row[f"{tf} HA"], row[f"{tf} Candle"], row[f"{tf} Both"] = ha_c, rg_c, both
    boths = [row[f"{tf} Both"] for tf in TFS]
    row["All TFs"] = (
        "ALL POSITIVE" if all(b == "YES - Positive" for b in boths)
        else "ALL NEGATIVE" if all(b == "YES - Negative" for b in boths) else ""
    )
    return row


# ---------------------------------------------------------------- UI
st.title("NSE Heikin-Ashi Scanner (Dhan)")
st.caption(
    "Latest completed candle per timeframe. 'Both' = YES when the Heikin-Ashi candle and the "
    "regular candle are the same colour (green/green or red/red). 30m is built from 15m data."
)

client_id, token = get_creds()
headers = {"access-token": token, "client-id": client_id, "Content-Type": "application/json",
           "Accept": "application/json"}

with st.sidebar:
    universe = st.selectbox("Universe", ["Nifty 50", "Nifty 100", "Nifty 500", "All NSE EQ", "Custom"])
    custom = st.text_area("Custom symbols (comma separated)", "RELIANCE, TCS, INFY") if universe == "Custom" else ""
    limit = st.number_input("Max stocks (0 = no limit)", 0, 3000, 0)
    workers = st.slider("Parallel workers", 1, 4, 2)
    run = st.button("Run scan", type="primary", disabled=not (client_id and token))
    st.caption("All NSE EQ is ~2,000 stocks x 2 calls; at ~2.5 req/s that is well over an hour; start with Nifty 50/100.")
    if st.button("Test connection", disabled=not (client_id and token)):
        try:
            sid0 = load_master().set_index("symbol").security_id.get("RELIANCE")
            body = {"securityId": str(sid0), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
                    "expiryCode": 0, "oi": False,
                    "fromDate": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "toDate": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")}
            r = requests.post(f"{API}/charts/historical", json=body, headers=headers, timeout=25)
            st.code(f"HTTP {r.status_code}\n{r.text[:500]}")
        except Exception as e:
            st.code(f"Failed: {e}")

if run:
    master = load_master()
    if universe == "All NSE EQ":
        syms = master.symbol.tolist()
    elif universe == "Custom":
        syms = [s.strip().upper() for s in custom.split(",") if s.strip()]
    else:
        syms = load_index_symbols(universe)
    m = master.set_index("symbol").security_id
    syms = [s for s in syms if s in m.index]
    if limit:
        syms = syms[: int(limit)]

    ERRS.clear()
    ABORT.clear()
    THROTTLED["n"] = 0
    rows, bar = [], st.progress(0.0, text="Scanning...")
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(scan_one, headers, s_, m[s_]) for s_ in syms]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception:
                pass
            bar.progress(i / len(futs), text=f"Scanned {i}/{len(futs)}")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    bar.empty()
    if ABORT.is_set():
        st.error("Dhan kept returning 429 (rate limit), so the scan was stopped. Wait a few minutes, "
                 "then rerun with a smaller universe or 1 worker.")
    res = pd.DataFrame(rows).sort_values("Symbol").reset_index(drop=True)
    st.session_state["res"] = res
    st.session_state["errs"] = dict(ERRS)
    st.session_state["scanned_at"] = datetime.now().strftime("%d %b %Y %H:%M")

if "res" in st.session_state:
    res = st.session_state["res"]
    st.subheader(f"Results - {len(res)} stocks (scanned {st.session_state['scanned_at']})")

    ok = res[[f"{t} Both" for t in TFS]].ne("n/a").all(axis=1)
    if (~ok).any():
        st.warning(f"{(~ok).sum()} stocks had missing data on at least one timeframe (shown as n/a).")
        errs = st.session_state.get("errs", {})
        if errs:
            st.error("Dhan API problems seen during the scan (most common first):")
            for k, v in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
                st.code(f"{v}x  {k}")

    c1, c2, c3 = st.columns(3)
    pick = c1.multiselect("Filter 'Both' on timeframe", TFS)
    kind = c2.selectbox("Match type", ["Any", "YES - Positive", "YES - Negative"])
    allt = c3.selectbox("Aligned on all timeframes", ["Any", "ALL POSITIVE", "ALL NEGATIVE"])
    view = res
    for t in pick:
        view = view[view[f"{t} Both"].str.startswith("YES")] if kind == "Any" else view[view[f"{t} Both"] == kind]
    if allt != "Any":
        view = view[view["All TFs"] == allt]

    def color(v):
        if isinstance(v, str) and ("Positive" in v or "POSITIVE" in v):
            return "background-color:#c8f7c5;color:#064"
        if isinstance(v, str) and ("Negative" in v or "NEGATIVE" in v):
            return "background-color:#f9c9c9;color:#800"
        return ""

    st.dataframe(view.style.map(color), use_container_width=True, height=420)
    st.download_button("Download CSV", view.to_csv(index=False), "ha_scan.csv", "text/csv")

    # summary chart
    counts = []
    for t in TFS:
        counts.append({"TF": t, "Type": "Positive match", "Count": (res[f"{t} Both"] == "YES - Positive").sum()})
        counts.append({"TF": t, "Type": "Negative match", "Count": (res[f"{t} Both"] == "YES - Negative").sum()})
        counts.append({"TF": t, "Type": "No match", "Count": (res[f"{t} Both"] == "No").sum()})
    cd = pd.DataFrame(counts)
    fig = go.Figure()
    for typ, col in [("Positive match", "#2ca02c"), ("Negative match", "#d62728"), ("No match", "#999999")]:
        s = cd[cd.Type == typ]
        fig.add_bar(x=s.TF, y=s.Count, name=typ, marker_color=col)
    fig.update_layout(barmode="group", title="Matches by timeframe", height=350)
    st.plotly_chart(fig, use_container_width=True)


if "res" not in st.session_state:
    st.info("Choose a universe in the sidebar and press **Run scan**.")

# ---------------------------------------------------------------- charts (independent of scan)
st.divider()
st.header("Charts: regular vs Heikin-Ashi (any stock)")
master = load_master()
scanned = st.session_state["res"].Symbol.tolist() if "res" in st.session_state else []
all_syms = scanned + [s_ for s_ in master.symbol if s_ not in set(scanned)]
k1, k2 = st.columns([2, 1])
sym = k1.selectbox("Stock", all_syms)
bars = k2.slider("Candles to show", 30, 200, 100)
sid = master.set_index("symbol").security_id.get(sym)

if not (client_id and token):
    st.info("Enter Dhan credentials to load charts.")
else:
    frames = cached_frames(client_id, token, sid)
    for tab, tf in zip(st.tabs(TFS), TFS):
        with tab:
            df = frames.get(tf)
            if df is None or df.empty:
                st.warning("No data returned. Press 'Test connection' in the sidebar to see Dhan's error.")
                continue
            d, h = df.tail(bars), heikin_ashi(df).tail(bars)
            x = d.index.strftime("%d-%b %H:%M")
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                              subplot_titles=("Regular candles", "Heikin-Ashi candles"))
            f.add_trace(go.Candlestick(x=x, open=d.open, high=d.high, low=d.low, close=d.close), 1, 1)
            f.add_trace(go.Candlestick(x=x, open=h.open, high=h.high, low=h.low, close=h.close), 2, 1)
            f.update_xaxes(type="category", nticks=12, rangeslider_visible=False)
            f.update_layout(height=700, showlegend=False, title=f"{sym} - {tf}")
            st.plotly_chart(f, use_container_width=True)
