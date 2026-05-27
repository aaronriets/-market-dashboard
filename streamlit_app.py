"""
Market Pulse Dashboard - Phase 1
Real-time descriptive view of ES, NQ, YM futures.
Shows trend, volume, EMA stack, and a composite "directional pressure" score.
Phase 2 will add historical base rates from backtests.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Market Pulse",
    page_icon="📊",
    layout="wide",
)

SYMBOLS = {
    "ES": "ES=F",   # E-mini S&P 500
    "NQ": "NQ=F",   # E-mini Nasdaq-100
    "YM": "YM=F",   # E-mini Dow
}

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="data_refresh")


# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=55)
def get_data(symbol: str, period: str, interval: str):
    """Pull bars from yfinance with light caching to avoid hammering the API."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================
# INDICATORS
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def trend_label(df: pd.DataFrame) -> str:
    """Up / Down / Sideways based on price vs 20-EMA vs 50-EMA."""
    if df is None or len(df) < 50:
        return "—"
    close = df["Close"]
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    last = close.iloc[-1]
    if last > ema20 > ema50:
        return "↑ Up"
    if last < ema20 < ema50:
        return "↓ Down"
    return "→ Sideways"


def ema_stack_status(df: pd.DataFrame) -> str:
    """Bullish (8>21>50>200) / Bearish / Mixed on the supplied timeframe."""
    if df is None or len(df) < 200:
        return "—"
    close = df["Close"]
    e8 = ema(close, 8).iloc[-1]
    e21 = ema(close, 21).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    if e8 > e21 > e50 > e200:
        return "🟢 Bullish stack"
    if e8 < e21 < e50 < e200:
        return "🔴 Bearish stack"
    return "⚪ Mixed"


def relative_volume(df: pd.DataFrame) -> str:
    """Most recent bar's volume vs the last 20 bars' average."""
    if df is None or len(df) < 21:
        return "—"
    recent = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-21:-1].mean()
    if avg == 0 or pd.isna(avg):
        return "—"
    rvol = recent / avg
    if rvol >= 1.5:
        return f"🔥 High ({rvol:.1f}x)"
    if rvol <= 0.5:
        return f"💤 Low ({rvol:.1f}x)"
    return f"Normal ({rvol:.1f}x)"


def session_range_position(df: pd.DataFrame) -> str:
    """Where the current price sits inside the day's range."""
    if df is None or len(df) < 1:
        return "—"
    today = df.tail(78)  # ~ one RTH session of 5-min bars
    high = today["High"].max()
    low = today["Low"].min()
    last = df["Close"].iloc[-1]
    if high == low:
        return "—"
    pct = (last - low) / (high - low) * 100
    if pct >= 80:
        return f"⬆ Near session high ({pct:.0f}%)"
    if pct <= 20:
        return f"⬇ Near session low ({pct:.0f}%)"
    return f"Mid-range ({pct:.0f}%)"


def lean_score(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> int:
    """
    Composite directional pressure score (0-100).
    Each input votes bullish (+1) or bearish (0). Final score = average.
    Describes CURRENT STATE, not future probability.
    """
    if df_5m is None or df_1h is None:
        return 50

    votes = []

    # 5m: price above/below 20 EMA
    close_5 = df_5m["Close"]
    e20_5 = ema(close_5, 20).iloc[-1]
    votes.append(1 if close_5.iloc[-1] > e20_5 else 0)

    # 5m: price above/below 50 EMA
    if len(close_5) >= 50:
        e50_5 = ema(close_5, 50).iloc[-1]
        votes.append(1 if close_5.iloc[-1] > e50_5 else 0)

    # 1h: price above/below 20 EMA
    close_1h = df_1h["Close"]
    if len(close_1h) >= 20:
        e20_1h = ema(close_1h, 20).iloc[-1]
        votes.append(1 if close_1h.iloc[-1] > e20_1h else 0)

    # 5m momentum: close vs 20 bars ago
    if len(close_5) >= 20:
        votes.append(1 if close_5.iloc[-1] > close_5.iloc[-20] else 0)

    # 5m higher-highs: second half vs first half of last 20 bars
    if len(df_5m) >= 20:
        recent = df_5m.tail(20)
        first_half_high = recent.iloc[:10]["High"].max()
        second_half_high = recent.iloc[10:]["High"].max()
        votes.append(1 if second_half_high > first_half_high else 0)

    if not votes:
        return 50
    return int(round(sum(votes) / len(votes) * 100))


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str) -> bool:
    """Send a message to the configured Telegram chat."""
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ============================================================
# UI
# ============================================================
st.title("📊 Market Pulse")
st.caption(
    f"Updated {datetime.now().strftime('%H:%M:%S')}  •  Source: yfinance (15-min delayed)  •  Auto-refresh: 60s"
)

# Top controls
c1, c2, c3 = st.columns([1, 1, 6])
with c1:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()
with c2:
    if st.button("📨 Test alert"):
        ok = send_telegram("✅ Test alert from Market Pulse dashboard")
        st.success("Sent") if ok else st.error("Failed (check secrets)")

st.divider()

# Pull all data once
data = {}
for label, symbol in SYMBOLS.items():
    data[label] = {
        "5m": get_data(symbol, period="5d", interval="5m"),
        "1h": get_data(symbol, period="1mo", interval="1h"),
        "1d": get_data(symbol, period="6mo", interval="1d"),
    }

# Per-instrument columns
cols = st.columns(3)
leans = {}

for i, (label, _) in enumerate(SYMBOLS.items()):
    with cols[i]:
        st.header(label)
        df_5m = data[label]["5m"]
        df_1h = data[label]["1h"]
        df_1d = data[label]["1d"]

        if df_5m is None:
            st.error("No data available")
            continue

        # Price + change
        last = df_5m["Close"].iloc[-1]
        prev_close = df_1d["Close"].iloc[-2] if df_1d is not None and len(df_1d) >= 2 else last
        change_pct = (last - prev_close) / prev_close * 100 if prev_close else 0.0
        st.metric("Price", f"{last:,.2f}", f"{change_pct:+.2f}%")

        # Composite lean
        score = lean_score(df_5m, df_1h)
        leans[label] = score
        if score >= 65:
            lean_emoji, lean_word = "🟢", "Bullish"
        elif score <= 35:
            lean_emoji, lean_word = "🔴", "Bearish"
        else:
            lean_emoji, lean_word = "⚪", "Neutral"

        st.markdown(f"**{lean_emoji} Directional pressure: {score}%**")
        st.progress(score / 100)
        st.caption(f"State: {lean_word}  •  describes right now, not future probability")

        # Trend by timeframe
        st.markdown("**Trend by timeframe**")
        st.write(f"5min:  {trend_label(df_5m)}")
        st.write(f"1hr:   {trend_label(df_1h)}")
        st.write(f"Daily: {trend_label(df_1d)}")

        # EMA stack (5m)
        st.markdown("**EMA stack (5m)**")
        st.write(ema_stack_status(df_5m))

        # Volume
        st.markdown("**Volume (last 5m bar)**")
        st.write(relative_volume(df_5m))

        # Range position
        st.markdown("**Session range**")
        st.write(session_range_position(df_5m))

st.divider()

# Index alignment summary
st.subheader("Index alignment")
if len(leans) == 3:
    vals = list(leans.values())
    if all(v >= 65 for v in vals):
        st.success("🟢 All three indices aligned BULLISH")
    elif all(v <= 35 for v in vals):
        st.error("🔴 All three indices aligned BEARISH")
    else:
        spread = max(vals) - min(vals)
        st.info(
            f"Mixed  •  ES: {leans['ES']}%  •  NQ: {leans['NQ']}%  •  YM: {leans['YM']}%"
        )
        if spread >= 30:
            leader = max(leans, key=leans.get)
            laggard = min(leans, key=leans.get)
            st.warning(
                f"⚠️ Divergence: {leader} leading ({leans[leader]}%), {laggard} lagging ({leans[laggard]}%)"
            )

st.divider()
st.caption(
    "Phase 1 — descriptive only. Phase 2 adds historical base rates from backtests, "
    "shown as captions under each percentage."
)
