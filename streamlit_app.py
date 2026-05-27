"""
Market Pulse Dashboard - Phase 2
Now with historical base rates from 3 years of CME futures data.
For each instrument, classifies current state and looks up historical outcomes
for matching conditions across multiple time horizons.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import requests
from streamlit_autorefresh import st_autorefresh

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Market Pulse", page_icon="📊", layout="wide")

SYMBOLS = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
}

# Map dashboard label → key in base_rates_5m.json
BASE_RATE_KEY = {"ES": "ES", "NQ": "MNQ", "YM": "MYM"}

SESSION_BARS = 78
BARS_1H = 12
BARS_4H = 48

st_autorefresh(interval=60_000, key="data_refresh")


# ============================================================
# BASE RATES (loaded once)
# ============================================================
@st.cache_data
def load_base_rates():
    path = Path(__file__).parent / "base_rates_5m.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


BASE_RATES = load_base_rates()


# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=55)
def get_data(symbol, period, interval):
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
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def trend_label(df):
    if df is None or len(df) < 50:
        return "—"
    close = df["Close"]
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    last = close.iloc[-1]
    if last > e20 > e50:
        return "↑ Up"
    if last < e20 < e50:
        return "↓ Down"
    return "→ Sideways"


def ema_stack_status(df):
    if df is None or len(df) < 200:
        return "—"
    c = df["Close"]
    e8 = ema(c, 8).iloc[-1]; e21 = ema(c, 21).iloc[-1]
    e50 = ema(c, 50).iloc[-1]; e200 = ema(c, 200).iloc[-1]
    if e8 > e21 > e50 > e200:
        return "🟢 Bullish stack"
    if e8 < e21 < e50 < e200:
        return "🔴 Bearish stack"
    return "⚪ Mixed"


def relative_volume(df):
    if df is None or len(df) < 21:
        return "—", 1.0
    recent = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-21:-1].mean()
    if avg == 0 or pd.isna(avg):
        return "—", 1.0
    rvol = recent / avg
    if rvol >= 1.5:
        return f"🔥 High ({rvol:.1f}x)", rvol
    if rvol <= 0.5:
        return f"💤 Low ({rvol:.1f}x)", rvol
    return f"Normal ({rvol:.1f}x)", rvol


def session_range_pct(df):
    if df is None or len(df) < 1:
        return None
    today = df.tail(SESSION_BARS)
    high = today["High"].max(); low = today["Low"].min()
    last = df["Close"].iloc[-1]
    if high == low:
        return None
    return (last - low) / (high - low) * 100


def session_range_position(df):
    pct = session_range_pct(df)
    if pct is None:
        return "—"
    if pct >= 80:
        return f"⬆ Near session high ({pct:.0f}%)"
    if pct <= 20:
        return f"⬇ Near session low ({pct:.0f}%)"
    return f"Mid-range ({pct:.0f}%)"


def lean_score(df_5m, df_1h):
    if df_5m is None or df_1h is None:
        return 50
    votes = []
    c5 = df_5m["Close"]
    votes.append(1 if c5.iloc[-1] > ema(c5, 20).iloc[-1] else 0)
    if len(c5) >= 50:
        votes.append(1 if c5.iloc[-1] > ema(c5, 50).iloc[-1] else 0)
    c1h = df_1h["Close"]
    if len(c1h) >= 20:
        votes.append(1 if c1h.iloc[-1] > ema(c1h, 20).iloc[-1] else 0)
    if len(c5) >= 20:
        votes.append(1 if c5.iloc[-1] > c5.iloc[-20] else 0)
    if len(df_5m) >= 20:
        r = df_5m.tail(20)
        votes.append(1 if r.iloc[10:]["High"].max() > r.iloc[:10]["High"].max() else 0)
    return int(round(sum(votes) / len(votes) * 100)) if votes else 50


def trend_duration_bars(df):
    if df is None or len(df) < 50:
        return None, None
    c = df["Close"]; e20 = ema(c, 20)
    above = c > e20
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
    m = bars * interval_minutes
    if m < 60:
        return f"{m}m"
    h, mm = m // 60, m % 60
    return f"{h}h" if mm == 0 else f"{h}h {mm}m"


def window_stats(df, n_bars):
    if df is None or len(df) < 2:
        return None
    n = min(n_bars, len(df))
    win = df.tail(n)
    open_p = win["Open"].iloc[0]
    if open_p == 0:
        return None
    return {
        "change": win["Close"].iloc[-1] - open_p,
        "range": win["High"].max() - win["Low"].min(),
    }


def avg_daily_range(df_1d, lookback=20):
    if df_1d is None or len(df_1d) < 2:
        return None
    n = min(lookback, len(df_1d))
    return float((df_1d.tail(n)["High"] - df_1d.tail(n)["Low"]).mean())


# ============================================================
# BASE RATE FEATURE COMPUTATION (must match offline script)
# ============================================================
def current_macd_state(close):
    if len(close) < 30:
        return None
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    return int((macd.iloc[-1] > 0)) * 2 + int((hist.iloc[-1] > 0))


def current_candle_type(df):
    if df is None or len(df) < 1:
        return None
    o, h, l, c = df.iloc[-1][["Open", "High", "Low", "Close"]]
    body = c - o
    rng = h - l
    if rng == 0:
        return 0
    body_pct = abs(body / rng)
    if body > 0 and body_pct >= 0.5:
        return 1
    if body < 0 and body_pct >= 0.5:
        return -1
    return 0


def current_session(df):
    if df is None or len(df) < 1:
        return None
    ts = df.index[-1]
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    et = ts.tz_convert("US/Eastern")
    m = et.hour * 60 + et.minute
    if 4 * 60 <= m < 9 * 60 + 30:
        return 1
    if 9 * 60 + 30 <= m < 12 * 60:
        return 2
    if 12 * 60 <= m < 16 * 60:
        return 3
    return 0


def current_vol_regime(df):
    if df is None or len(df) < 21:
        return 0
    recent = df["Volume"].iloc[-1]
    avg = df["Volume"].iloc[-21:-1].mean()
    if avg == 0 or pd.isna(avg):
        return 0
    rvol = recent / avg
    if rvol > 1.5:
        return 1
    if rvol < 0.5:
        return -1
    return 0


def macd_state_label(s):
    return {0: "strong bear", 1: "weak bear", 2: "weak bull", 3: "strong bull"}.get(s, "—")


def candle_label(c):
    return {-1: "bearish", 0: "neutral", 1: "bullish"}.get(c, "—")


def session_label(s):
    return {0: "overnight", 1: "premarket", 2: "RTH AM", 3: "RTH PM"}.get(s, "—")


def vol_label(v):
    return {-1: "low", 0: "normal", 1: "high"}.get(v, "—")


def lookup_base_rate(market_key, state, horizon):
    """
    Try full 4-feature match first, fall back to 3-feature, then 2-feature.
    Returns (data, granularity_label) or (None, None) if nothing available.
    """
    if BASE_RATES is None or market_key not in BASE_RATES:
        return None, None
    m, c, s, v = state
    rates = BASE_RATES[market_key]
    k_full = f"{m}|{c}|{s}|{v}"
    k_3 = f"{m}|{c}|{s}"
    k_2 = f"{m}|{c}"
    if k_full in rates.get(f"h{horizon}_full", {}):
        return rates[f"h{horizon}_full"][k_full], "full match (MACD+candle+session+vol)"
    if k_3 in rates.get(f"h{horizon}_macd_candle_session", {}):
        return rates[f"h{horizon}_macd_candle_session"][k_3], "3-feature match (MACD+candle+session)"
    if k_2 in rates.get(f"h{horizon}_macd_candle", {}):
        return rates[f"h{horizon}_macd_candle"][k_2], "2-feature match (MACD+candle)"
    return None, None


# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def quick_summary(score, df_5m, df_1d, dur_str):
    parts = []
    if score >= 65:
        emoji, state = "📈", "bullish"
    elif score <= 35:
        emoji, state = "📉", "bearish"
    else:
        emoji, state = "↔️", "neutral"
    if df_5m is not None and df_1d is not None and len(df_1d) >= 2:
        last = df_5m["Close"].iloc[-1]; prev = df_1d["Close"].iloc[-2]
        pct = (last - prev) / prev * 100 if prev else 0
        if pct > 0.3:
            parts.append(f"Up {pct:.2f}%")
        elif pct < -0.3:
            parts.append(f"Down {abs(pct):.2f}%")
        else:
            parts.append("Flat")
    rng_pct = session_range_pct(df_5m)
    if rng_pct is not None:
        if rng_pct >= 80:
            parts.append("near session high")
        elif rng_pct <= 20:
            parts.append("near session low")
        else:
            parts.append("mid-range")
    parts.append(f"{state} for {dur_str}")
    _, rvol = relative_volume(df_5m)
    if rvol >= 1.5:
        parts.append("high vol")
    elif rvol <= 0.5:
        parts.append("quiet vol")
    return f"{emoji} " + ", ".join(parts)


# ============================================================
# UI
# ============================================================
st.title("📊 Market Pulse")
st.caption(
    f"Updated {datetime.now().strftime('%H:%M:%S')}  •  Live data: yfinance (15-min delayed)  "
    f"•  Base rates: 3 years CME data (Apr 2023 – May 2026)  •  Auto-refresh: 60s"
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

if BASE_RATES is None:
    st.warning("base_rates_5m.json not found — historical base rates disabled. Live data still works.")

st.divider()

data = {}
for label, symbol in SYMBOLS.items():
    data[label] = {
        "5m": get_data(symbol, "5d", "5m"),
        "1h": get_data(symbol, "1mo", "1h"),
        "1d": get_data(symbol, "6mo", "1d"),
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
            st.error("No data")
            continue

        # Score
        score = lean_score(df_5m, df_1h)
        leans[label] = score
        dur_bars, _ = trend_duration_bars(df_5m)
        dur_str = format_duration(dur_bars)

        # Quick summary
        st.info(quick_summary(score, df_5m, df_1d, dur_str))

        # Price
        last = df_5m["Close"].iloc[-1]
        prev_close = df_1d["Close"].iloc[-2] if df_1d is not None and len(df_1d) >= 2 else last
        change_pct = (last - prev_close) / prev_close * 100 if prev_close else 0.0
        st.metric("Price", f"{last:,.2f}", f"{change_pct:+.2f}%")

        # Directional pressure
        if score >= 65:
            le, lw = "🟢", "Bullish"
        elif score <= 35:
            le, lw = "🔴", "Bearish"
        else:
            le, lw = "⚪", "Neutral"
        st.markdown(f"**{le} Directional pressure: {score}%**")
        st.progress(score / 100)
        st.caption(f"State: {lw}  •  current state for {dur_str}")

        # ============================================
        # HISTORICAL BASE RATES (NEW)
        # ============================================
        st.markdown("**📊 Historical base rates**")
        macd_s = current_macd_state(df_5m["Close"])
        candle_s = current_candle_type(df_5m)
        session_s = current_session(df_5m)
        vol_s = current_vol_regime(df_5m)

        if None in (macd_s, candle_s, session_s, vol_s):
            st.write("—")
        else:
            st.caption(
                f"Current state: **{macd_state_label(macd_s)}** MACD · "
                f"**{candle_label(candle_s)}** candle · "
                f"**{session_label(session_s)}** · "
                f"**{vol_label(vol_s)}** volume"
            )

            market_key = BASE_RATE_KEY[label]
            state = (macd_s, candle_s, session_s, vol_s)
            granularity_shown = None
            rows = []
            for horizon in ["5m", "15m", "60m", "240m", "1440m"]:
                r, gran = lookup_base_rate(market_key, state, horizon)
                if r is None:
                    rows.append({"Horizon": horizon, "n": "—", "% up": "—", "Median": "—", "90% band": "—"})
                else:
                    if granularity_shown is None:
                        granularity_shown = gran
                    horizon_label = {"5m": "5 min", "15m": "15 min", "60m": "1 hr", "240m": "4 hr", "1440m": "24 hr"}[horizon]
                    rows.append({
                        "Horizon": horizon_label,
                        "n": f"{r['n']:,}",
                        "% up": f"{r['pct_up']:.0f}%",
                        "Median": f"{r['median']:+.1f}",
                        "90% band": f"{r['p5']:+.0f} to {r['p95']:+.0f}",
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if granularity_shown:
                st.caption(f"_{granularity_shown}_")

        # Move snapshot
        st.markdown("**Move snapshot**")
        w1h = window_stats(df_5m, BARS_1H)
        w4h = window_stats(df_5m, BARS_4H)
        wd = window_stats(df_5m, SESSION_BARS)
        adr = avg_daily_range(df_1d)
        if w1h:
            st.write(f"1hr:   {w1h['change']:+.2f} pts  •  range {w1h['range']:.2f}")
        if w4h:
            st.write(f"4hr:   {w4h['change']:+.2f} pts  •  range {w4h['range']:.2f}")
        if wd:
            avg_txt = f", avg daily: {adr:.2f}" if adr else ""
            st.write(f"Today: {wd['change']:+.2f} pts  •  range {wd['range']:.2f}{avg_txt}")

        # Trend / EMA / Volume / Range
        st.markdown("**Trend by timeframe**")
        st.write(f"5min:  {trend_label(df_5m)}")
        st.write(f"1hr:   {trend_label(df_1h)}")
        st.write(f"Daily: {trend_label(df_1d)}")

        st.markdown("**EMA stack (5m)**")
        st.write(ema_stack_status(df_5m))

        st.markdown("**Volume (last 5m bar)**")
        vol_str, _ = relative_volume(df_5m)
        st.write(vol_str)

        st.markdown("**Session range position**")
        st.write(session_range_position(df_5m))

st.divider()

# Alignment
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
    "Phase 2  •  Base rates from 3 years of CME 5-min OHLCV data via Databento. "
    "Probabilities reflect historical frequency under matching MACD/candle/session/volume conditions. "
    "Past performance ≠ future results."
)
