"""
Market Pulse Dashboard - Phase 1.5
Real-time descriptive view of ES, NQ, YM futures.
Adds: trend duration, today's points move, floor/ceiling bounce stats.
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

SESSION_BARS = 78  # ~one RTH session in 5-min bars

st_autorefresh(interval=60_000, key="data_refresh")


# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=55)
def get_data(symbol: str, period: str, interval: str):
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
    if df is None or len(df) < 1:
        return "—"
    today = df.tail(SESSION_BARS)
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


def lean_score(df_5m, df_1h):
    if df_5m is None or df_1h is None:
        return 50
    votes = []
    close_5 = df_5m["Close"]
    e20_5 = ema(close_5, 20).iloc[-1]
    votes.append(1 if close_5.iloc[-1] > e20_5 else 0)
    if len(close_5) >= 50:
        e50_5 = ema(close_5, 50).iloc[-1]
        votes.append(1 if close_5.iloc[-1] > e50_5 else 0)
    close_1h = df_1h["Close"]
    if len(close_1h) >= 20:
        e20_1h = ema(close_1h, 20).iloc[-1]
        votes.append(1 if close_1h.iloc[-1] > e20_1h else 0)
    if len(close_5) >= 20:
        votes.append(1 if close_5.iloc[-1] > close_5.iloc[-20] else 0)
    if len(df_5m) >= 20:
        recent = df_5m.tail(20)
        first_high = recent.iloc[:10]["High"].max()
        second_high = recent.iloc[10:]["High"].max()
        votes.append(1 if second_high > first_high else 0)
    if not votes:
        return 50
    return int(round(sum(votes) / len(votes) * 100))


# ============================================================
# NEW: TREND DURATION
# ============================================================
def trend_duration_bars(df: pd.DataFrame):
    """Consecutive bars in the current state (price above or below 20 EMA)."""
    if df is None or len(df) < 50:
        return None, None
    close = df["Close"]
    e20 = ema(close, 20)
    above = close > e20
    current = bool(above.iloc[-1])
    count = 0
    for i in range(len(above) - 1, -1, -1):
        if pd.isna(e20.iloc[i]):
            break
        if bool(above.iloc[i]) == current:
            count += 1
        else:
            break
    return count, current


def format_duration(bars, interval_minutes=5):
    if bars is None:
        return "—"
    minutes = bars * interval_minutes
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


# ============================================================
# NEW: POINTS STATS
# ============================================================
def points_stats(df: pd.DataFrame):
    """Session move statistics in points and %."""
    if df is None or len(df) < 1:
        return None
    today = df.tail(SESSION_BARS)
    if len(today) < 1:
        return None
    open_p = today["Open"].iloc[0]
    high_p = today["High"].max()
    low_p = today["Low"].min()
    current = today["Close"].iloc[-1]
    if open_p == 0:
        return None
    return {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "current": current,
        "from_open": current - open_p,
        "from_open_pct": (current - open_p) / open_p * 100,
        "from_high": current - high_p,
        "from_low": current - low_p,
        "range": high_p - low_p,
        "range_pct": (high_p - low_p) / open_p * 100,
    }


# ============================================================
# NEW: BOUNCE STATS
# ============================================================
def bounce_stats(df: pd.DataFrame, edge_pct: float = 15.0):
    """
    Floor/ceiling bounce analysis for the current session.
    Floor zone = bottom edge_pct% of range. Ceiling zone = top edge_pct%.
    Counts distinct visits to each zone and what % of those visits
    were followed by a visit to the opposite zone.
    """
    if df is None or len(df) < 20:
        return None
    today = df.tail(SESSION_BARS)
    sh = today["High"].max()
    sl = today["Low"].min()
    sr = sh - sl
    if sr == 0:
        return None

    floor_z = sl + sr * (edge_pct / 100)
    ceiling_z = sh - sr * (edge_pct / 100)

    zones = []
    for i in range(len(today)):
        low = today["Low"].iloc[i]
        high = today["High"].iloc[i]
        if low <= floor_z:
            zones.append("floor")
        elif high >= ceiling_z:
            zones.append("ceiling")
        else:
            zones.append("middle")

    floor_visits = []
    ceiling_visits = []
    cur = None
    start = 0
    for i, z in enumerate(zones):
        if z != cur:
            if cur == "floor":
                floor_visits.append((start, i - 1))
            elif cur == "ceiling":
                ceiling_visits.append((start, i - 1))
            cur = z
            start = i
    if cur == "floor":
        floor_visits.append((start, len(zones) - 1))
    elif cur == "ceiling":
        ceiling_visits.append((start, len(zones) - 1))

    f_to_c = 0
    for _, end in floor_visits:
        if end < len(zones) - 1 and "ceiling" in zones[end + 1:]:
            f_to_c += 1
    c_to_f = 0
    for _, end in ceiling_visits:
        if end < len(zones) - 1 and "floor" in zones[end + 1:]:
            c_to_f += 1

    floor_pct = (f_to_c / len(floor_visits) * 100) if floor_visits else None
    ceiling_pct = (c_to_f / len(ceiling_visits) * 100) if ceiling_visits else None

    total = len(zones)
    return {
        "floor_visits": len(floor_visits),
        "ceiling_visits": len(ceiling_visits),
        "floor_to_ceiling_pct": floor_pct,
        "ceiling_to_floor_pct": ceiling_pct,
        "floor_time_pct": zones.count("floor") / total * 100,
        "middle_time_pct": zones.count("middle") / total * 100,
        "ceiling_time_pct": zones.count("ceiling") / total * 100,
    }


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str) -> bool:
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

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()
with c2:
    if st.button("📨 Test alert"):
        ok = send_telegram("✅ Test alert from Market Pulse dashboard")
        st.success("Sent") if ok else st.error("Failed (check secrets)")

st.divider()

data = {}
for label, symbol in SYMBOLS.items():
    data[label] = {
        "5m": get_data(symbol, period="5d", interval="5m"),
        "1h": get_data(symbol, period="1mo", interval="1h"),
        "1d": get_data(symbol, period="6mo", interval="1d"),
    }

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

        # Price
        last = df_5m["Close"].iloc[-1]
        prev_close = df_1d["Close"].iloc[-2] if df_1d is not None and len(df_1d) >= 2 else last
        change_pct = (last - prev_close) / prev_close * 100 if prev_close else 0.0
        st.metric("Price", f"{last:,.2f}", f"{change_pct:+.2f}%")

        # Composite lean + state duration
        score = lean_score(df_5m, df_1h)
        leans[label] = score
        if score >= 65:
            le, lw = "🟢", "Bullish"
        elif score <= 35:
            le, lw = "🔴", "Bearish"
        else:
            le, lw = "⚪", "Neutral"

        dur_bars, dur_above = trend_duration_bars(df_5m)
        dur_str = format_duration(dur_bars, interval_minutes=5)
        dur_dir = "above 20-EMA" if dur_above else "below 20-EMA"

        st.markdown(f"**{le} Directional pressure: {score}%**")
        st.progress(score / 100)
        st.caption(f"State: {lw}  •  current state for {dur_str} ({dur_dir})")

        # Today's move (points + %)
        ps = points_stats(df_5m)
        st.markdown("**Today's move**")
        if ps:
            st.write(f"From open: {ps['from_open']:+.2f} pts ({ps['from_open_pct']:+.2f}%)")
            st.write(f"From session high: {ps['from_high']:+.2f} pts")
            st.write(f"From session low: {ps['from_low']:+.2f} pts")
            st.write(f"Range: {ps['range']:.2f} pts ({ps['range_pct']:.2f}%)")
        else:
            st.write("—")

        # Trend by timeframe
        st.markdown("**Trend by timeframe**")
        st.write(f"5min:  {trend_label(df_5m)}")
        st.write(f"1hr:   {trend_label(df_1h)}")
        st.write(f"Daily: {trend_label(df_1d)}")

        # EMA stack
        st.markdown("**EMA stack (5m)**")
        st.write(ema_stack_status(df_5m))

        # Volume
        st.markdown("**Volume (last 5m bar)**")
        st.write(relative_volume(df_5m))

        # Session range position
        st.markdown("**Session range position**")
        st.write(session_range_position(df_5m))

        # Floor/ceiling bounce stats (NEW)
        bs = bounce_stats(df_5m)
        st.markdown("**Floor / ceiling action (today)**")
        if bs is None:
            st.write("—")
        else:
            st.write(f"Floor visits: {bs['floor_visits']}")
            if bs["floor_to_ceiling_pct"] is not None:
                st.write(f"→ Reached ceiling after: {bs['floor_to_ceiling_pct']:.0f}%")
            st.write(f"Ceiling visits: {bs['ceiling_visits']}")
            if bs["ceiling_to_floor_pct"] is not None:
                st.write(f"→ Reached floor after: {bs['ceiling_to_floor_pct']:.0f}%")
            st.caption(
                f"Time: floor {bs['floor_time_pct']:.0f}%  •  "
                f"middle {bs['middle_time_pct']:.0f}%  •  "
                f"ceiling {bs['ceiling_time_pct']:.0f}%"
            )

st.divider()

# Index alignment
st.subheader("Index alignment")
if len(leans) == 3:
    vals = list(leans.values())
    if all(v >= 65 for v in vals):
        st.success("🟢 All three indices aligned BULLISH")
    elif all(v <= 35 for v in vals):
        st.error("🔴 All three indices aligned BEARISH")
    else:
        spread = max(vals) - min(vals)
        st.info(f"Mixed  •  ES: {leans['ES']}%  •  NQ: {leans['NQ']}%  •  YM: {leans['YM']}%")
        if spread >= 30:
            leader = max(leans, key=leans.get)
            laggard = min(leans, key=leans.get)
            st.warning(
                f"⚠️ Divergence: {leader} leading ({leans[leader]}%), {laggard} lagging ({leans[laggard]}%)"
            )

st.divider()
st.caption(
    "Phase 1.5  •  Intraday stats only. Phase 2 will add historical base rates from backtests, "
    "shown as captions under each percentage."
)
