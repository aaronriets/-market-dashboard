# Market Pulse Dashboard

Phase 1 — descriptive real-time dashboard for ES, NQ, YM futures.

## What it shows
- Current price and % change
- Composite "directional pressure" score (current state, not prediction)
- Trend reading on 5min / 1hr / Daily timeframes
- EMA stack status (8/21/50/200)
- Relative volume vs last 20 bars
- Position within session range
- Index alignment / divergence summary

## Data source
yfinance (15-min delayed). Swappable later for a real-time feed.

## Phase 2 (next)
Historical base rates from backtests. Each percentage will gain a caption
showing how that condition has resolved historically.
