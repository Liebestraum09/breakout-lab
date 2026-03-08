import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from flask import Flask, render_template_string

app = Flask(__name__)

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

BACKTEST_YEARS = 10

# --- Turtle System parameters ---
# System 1: faster, more trades, smaller wins
S1_ENTRY  = 20      # 20-day high breakout
S1_EXIT   = 10      # 10-day low exit
# System 2: slower, fewer trades, catches bigger trends
S2_ENTRY  = 55      # 55-day high breakout
S2_EXIT   = 20      # 20-day low exit

ATR_WINDOW = 20     # ATR period (shared)

# --- Portfolio limits ---
INITIAL_CAPITAL = 10_000
RISK_PER_TRADE  = 0.01      # 1% of current equity per unit
MAX_POSITIONS   = 12        # max simultaneous positions (across all systems)
MAX_PER_TICKER  = 1         # max 1 unit per ticker (no pyramiding in V3)

# --- Cost model ---
COMMISSION    = 0.001
SLIPPAGE      = 0.001
ROUND_TRIP    = (COMMISSION + SLIPPAGE) * 2

# =============================================================================
# SECTION 2: UNIVERSE — EXPANDED MULTI-SECTOR
#
# Original turtle universe was: commodities, bonds, FX, equities.
# We approximate with ETFs covering 11 S&P sectors + bonds + commodities + gold.
# Nasdaq 100 top constituents retained for equity exposure.
#
# SECTOR ETFs:
#   XLK  Technology       XLF  Financials      XLE  Energy
#   XLV  Healthcare       XLI  Industrials     XLY  Consumer Disc.
#   XLP  Consumer Staples XLU  Utilities       XLRE Real Estate
#   XLB  Materials        XLC  Communication
#
# BONDS / MACRO:
#   TLT  20yr Treasury    IEF  7-10yr Treasury  HYG  High Yield Corp
#
# COMMODITIES / ALTERNATIVES:
#   GLD  Gold             SLV  Silver           USO  Oil
#   DBA  Agriculture      PDBC Broad Commodity
#
# NASDAQ 100 TOP NAMES (equity alpha):
#   AAPL MSFT NVDA AMZN META GOOGL TSLA AVGO COST NFLX
#   AMD  QCOM ADBE INTU AMAT MU   LRCX CDNS SNPS MRVL
# =============================================================================

SECTOR_ETFS = [
    # US Equity Sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
    # Bonds
    "TLT", "IEF", "HYG",
    # Commodities & Alternatives
    "GLD", "SLV", "USO", "DBA", "PDBC",
    # Nasdaq 100 individual names
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "NFLX",
    "AMD",  "QCOM", "ADBE", "INTU", "AMAT", "MU",   "LRCX", "CDNS", "SNPS", "MRVL",
]

SECTOR_LABELS = {
    "XLK":"Tech","XLF":"Financials","XLE":"Energy","XLV":"Healthcare",
    "XLI":"Industrials","XLY":"Cons.Disc","XLP":"Cons.Staples","XLU":"Utilities",
    "XLRE":"Real Estate","XLB":"Materials","XLC":"Comm.",
    "TLT":"Bond 20yr","IEF":"Bond 7-10yr","HYG":"High Yield",
    "GLD":"Gold","SLV":"Silver","USO":"Oil","DBA":"Agriculture","PDBC":"Commodity",
}

# =============================================================================
# SECTION 3: DATA FETCHING & CACHING
# =============================================================================

def fetch_and_cache(ticker, years=BACKTEST_YEARS):
    cache_file = f"cache/{ticker}_{years}yr_v3.pkl"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)

    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.to_pickle(cache_file)
    time.sleep(0.1)
    return df

# =============================================================================
# SECTION 4: INDICATOR CALCULATION
#
# Both System 1 and System 2 indicators computed for every ticker.
# All values shifted(1): only yesterday's data used → no lookahead.
#
# S1: entry on 20d high break, exit on 10d low break
# S2: entry on 55d high break, exit on 20d low break
# ATR: shared 20-day ATR for position sizing on both systems
# =============================================================================

def calculate_indicators(df):
    # System 1
    df['s1_high'] = df['high'].rolling(S1_ENTRY).max().shift(1)
    df['s1_low']  = df['low'].rolling(S1_EXIT).min().shift(1)
    df['s1_entry'] = df['high'] > df['s1_high']
    df['s1_exit']  = df['low']  < df['s1_low']

    # System 2
    df['s2_high'] = df['high'].rolling(S2_ENTRY).max().shift(1)
    df['s2_low']  = df['low'].rolling(S2_EXIT).min().shift(1)
    df['s2_entry'] = df['high'] > df['s2_high']
    df['s2_exit']  = df['low']  < df['s2_low']

    # Shared ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low']  - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW).mean().shift(1)

    return df.dropna()

# =============================================================================
# SECTION 5: PORTFOLIO-LEVEL BACKTEST ENGINE
#
# Dual-system logic:
#   Each ticker can have AT MOST one open position total (S1 OR S2, not both).
#   Priority: S2 takes precedence when both fire on the same day
#             (S2 is the "stronger" signal — longer breakout).
#   Exit: each position uses its own system's exit rule.
#         S1 position → exits on 10d low break
#         S2 position → exits on 20d low break
#
# Compounding: equity updates after every closed trade.
# Position sizing: 1% of current equity / ATR (same as V2).
# Capital guard: skip entry if cost > 50% of current equity.
# =============================================================================

def run_portfolio_backtest(all_data):
    all_dates = sorted(set(
        d for df in all_data.values() for d in df.index
    ))

    equity        = INITIAL_CAPITAL
    positions     = {}        # ticker → {system, entry_date, entry_price, shares, atr}
    closed_trades = []
    equity_curve  = []

    for date in all_dates:

        # ── STEP 1: Process exits ─────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            df = all_data.get(ticker)
            if df is None or date not in df.index:
                continue
            prev_dates = df.index[df.index < date]
            if len(prev_dates) == 0:
                continue
            prev = df.loc[prev_dates[-1]]
            row  = df.loc[date]

            # Use the exit rule that matches the system that opened the trade
            exit_triggered = (
                prev['s1_exit'] if pos['system'] == 'S1' else prev['s2_exit']
            )

            if exit_triggered:
                exit_price = row['open']
                entry_p    = pos['entry_price']
                shares     = pos['shares']
                net        = (exit_price / entry_p - 1) - ROUND_TRIP
                pnl        = net * entry_p * shares
                equity    += pnl
                closed_trades.append({
                    'ticker':       ticker,
                    'system':       pos['system'],
                    'entry_date':   pos['entry_date'],
                    'exit_date':    date,
                    'entry_price':  round(entry_p, 2),
                    'exit_price':   round(exit_price, 2),
                    'shares':       shares,
                    'return_pct':   round(net * 100, 2),
                    'pnl':          round(pnl, 2),
                    'hold_days':    (date - pos['entry_date']).days,
                    'equity_after': round(equity, 2),
                })
                to_close.append(ticker)

        for t in to_close:
            del positions[t]

        # ── STEP 2: Process entries ───────────────────────────────────────
        if len(positions) < MAX_POSITIONS:
            for ticker, df in all_data.items():
                if len(positions) >= MAX_POSITIONS:
                    break
                if ticker in positions:
                    continue
                if date not in df.index:
                    continue

                prev_dates = df.index[df.index < date]
                if len(prev_dates) == 0:
                    continue
                prev = df.loc[prev_dates[-1]]
                row  = df.loc[date]

                # S2 takes priority over S1 when both fire
                if prev['s2_entry']:
                    system = 'S2'
                elif prev['s1_entry']:
                    system = 'S1'
                else:
                    continue

                atr = row['atr']
                if atr <= 0:
                    continue

                shares = int((equity * RISK_PER_TRADE) / atr)
                if shares < 1:
                    continue

                entry_price = row['open']
                if entry_price * shares > equity * 0.5:
                    continue

                positions[ticker] = {
                    'system':       system,
                    'entry_date':   date,
                    'entry_price':  entry_price,
                    'shares':       shares,
                    'atr':          atr,
                }

        # Mark-to-market for equity curve
        mtm = equity
        for ticker, pos in positions.items():
            df = all_data.get(ticker)
            if df is not None and date in df.index:
                mtm += (df.loc[date, 'close'] - pos['entry_price']) * pos['shares']
        equity_curve.append({'date': date, 'equity': round(mtm, 2)})

    # Close remaining positions at last price
    for ticker, pos in positions.items():
        df   = all_data[ticker]
        last = df.iloc[-1]
        net  = (last['close'] / pos['entry_price'] - 1) - ROUND_TRIP
        pnl  = net * pos['entry_price'] * pos['shares']
        equity += pnl
        closed_trades.append({
            'ticker':       ticker,
            'system':       pos['system'],
            'entry_date':   pos['entry_date'],
            'exit_date':    last.name,
            'entry_price':  round(pos['entry_price'], 2),
            'exit_price':   round(last['close'], 2),
            'shares':       pos['shares'],
            'return_pct':   round(net * 100, 2),
            'pnl':          round(pnl, 2),
            'hold_days':    (last.name - pos['entry_date']).days,
            'equity_after': round(equity, 2),
        })

    return closed_trades, equity_curve, equity

# =============================================================================
# SECTION 6: SUMMARY STATISTICS
# =============================================================================

def summarize(trades, initial, final):
    if not trades:
        return {}
    rets    = [t['return_pct'] for t in trades]
    pnls    = [t['pnl'] for t in trades]
    winners = [r for r in rets if r > 0]
    losers  = [r for r in rets if r <= 0]
    avg_win  = round(sum(winners)/len(winners), 2) if winners else 0
    avg_loss = round(sum(losers) /len(losers),  2) if losers  else 0
    payoff   = round(abs(avg_win/avg_loss), 2) if avg_loss else 0
    holds    = [t['hold_days'] for t in trades]

    s1 = [t for t in trades if t['system'] == 'S1']
    s2 = [t for t in trades if t['system'] == 'S2']

    return {
        'total_trades':  len(trades),
        'win_rate':      round(len(winners)/len(trades)*100, 1),
        'total_return':  round((final/initial - 1)*100, 1),
        'cagr':          round(((final/initial)**(1/BACKTEST_YEARS)-1)*100, 1),
        'final_equity':  round(final, 0),
        'avg_win':       avg_win,
        'avg_loss':      avg_loss,
        'payoff':        payoff,
        'avg_hold':      round(sum(holds)/len(holds), 1),
        'best':          round(max(rets), 2),
        'worst':         round(min(rets), 2),
        's1_trades':     len(s1),
        's2_trades':     len(s2),
        's1_pnl':        round(sum(t['pnl'] for t in s1), 0),
        's2_pnl':        round(sum(t['pnl'] for t in s2), 0),
        's1_wr':         round(sum(1 for t in s1 if t['return_pct']>0)/len(s1)*100,1) if s1 else 0,
        's2_wr':         round(sum(1 for t in s2 if t['return_pct']>0)/len(s2)*100,1) if s2 else 0,
    }

def per_ticker_summary(trades):
    by_ticker = {}
    for t in trades:
        by_ticker.setdefault(t['ticker'], []).append(t)
    result = []
    for ticker, ts in by_ticker.items():
        rets  = [t['return_pct'] for t in ts]
        pnls  = [t['pnl'] for t in ts]
        wins  = [r for r in rets if r > 0]
        result.append({
            'ticker':     ticker,
            'label':      SECTOR_LABELS.get(ticker, ''),
            'trades':     len(ts),
            'win_rate':   round(len(wins)/len(ts)*100, 1),
            'total_pnl':  round(sum(pnls), 0),
            'avg_return': round(sum(rets)/len(rets), 2),
        })
    result.sort(key=lambda x: x['total_pnl'], reverse=True)
    return result

# =============================================================================
# SECTION 7: FLASK DASHBOARD
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Turtle V3 — Dual System + Multi-Sector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>
        :root {
            --bg:#f0f2f5; --surface:#fff; --border:#e4e8ef;
            --text:#111827; --muted:#6b7280; --faint:#9ca3af;
            --green:#16a34a; --green-bg:#f0fdf4;
            --red:#dc2626;   --red-bg:#fef2f2;
            --blue:#2563eb;  --blue-bg:#eff6ff;
            --amber:#d97706; --amber-bg:#fffbeb;
            --purple:#7c3aed;--purple-bg:#f5f3ff;
            --teal:#0d9488;  --teal-bg:#f0fdfa;
        }
        * { box-sizing:border-box; margin:0; padding:0; }
        body { font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); padding:32px; font-size:13px; }

        .header {
            background:var(--surface); border:1px solid var(--border);
            border-radius:12px; padding:24px 28px; margin-bottom:20px;
            display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
        }
        .header h1 { font-size:18px; font-weight:700; letter-spacing:-0.02em; }
        .header p  { color:var(--muted); margin-top:4px; font-size:12px; }
        .badges { display:flex; gap:8px; flex-wrap:wrap; }
        .badge { font-size:11px; font-weight:600; padding:4px 12px; border-radius:20px; font-family:'JetBrains Mono',monospace; }
        .b-blue   { background:var(--blue-bg);   color:var(--blue);   border:1px solid #bfdbfe; }
        .b-purple { background:var(--purple-bg); color:var(--purple); border:1px solid #ddd6fe; }
        .b-green  { background:var(--green-bg);  color:var(--green);  border:1px solid #bbf7d0; }
        .b-amber  { background:var(--amber-bg);  color:var(--amber);  border:1px solid #fde68a; }
        .b-teal   { background:var(--teal-bg);   color:var(--teal);   border:1px solid #99f6e4; }

        /* KPI */
        .kpi-bar { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:12px; margin-bottom:20px; }
        .kpi { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
        .kpi-label { font-size:11px; color:var(--muted); font-weight:500; margin-bottom:4px; }
        .kpi-value { font-size:20px; font-weight:700; font-family:'JetBrains Mono',monospace; }
        .kpi-value.pos { color:var(--green); }
        .kpi-value.neg { color:var(--red); }

        /* System comparison */
        .sys-bar { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:20px; }
        .sys-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }
        .sys-title { font-size:12px; font-weight:700; margin-bottom:10px; }
        .sys-s1 { color:var(--blue); }
        .sys-s2 { color:var(--purple); }
        .sys-row { display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px solid #f3f4f6; }
        .sys-row:last-child { border-bottom:none; }
        .sys-key { color:var(--muted); }

        /* Chart */
        .chart-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:20px; }
        .chart-title { font-size:12px; font-weight:600; color:var(--muted); margin-bottom:14px; }
        .chart-wrap { height:240px; }

        /* Two col */
        .two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }
        @media(max-width:900px){ .two-col { grid-template-columns:1fr; } }

        .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
        .card-header { padding:13px 18px; border-bottom:1px solid var(--border); font-size:13px; font-weight:600; color:var(--muted); }

        table { width:100%; border-collapse:collapse; }
        th {
            padding:7px 12px; text-align:center;
            font-size:10px; font-weight:600; color:var(--faint);
            background:#fafbfc; border-bottom:1px solid var(--border);
            text-transform:uppercase; letter-spacing:0.05em; white-space:nowrap;
        }
        th.left { text-align:left; }
        td { padding:7px 12px; text-align:center; font-size:12px; border-bottom:1px solid #f3f4f6; font-family:'JetBrains Mono',monospace; }
        td.label { text-align:left; font-family:'Inter',sans-serif; font-weight:600; }
        td.sub   { text-align:left; font-family:'Inter',sans-serif; color:var(--muted); font-size:11px; }
        td.muted { text-align:left; font-family:'Inter',sans-serif; color:var(--muted); }
        tr:last-child td { border-bottom:none; }
        tr:hover td { background:#fafbfc; }

        .pos { color:var(--green); font-weight:600; }
        .neg { color:var(--red);   font-weight:600; }
        .pill-pos { background:var(--green-bg); color:var(--green); font-weight:600; padding:2px 8px; border-radius:4px; display:inline-block; }
        .pill-neg { background:var(--red-bg);   color:var(--red);   font-weight:600; padding:2px 8px; border-radius:4px; display:inline-block; }
        .pill-s1  { background:var(--blue-bg);   color:var(--blue);   font-size:10px; padding:1px 6px; border-radius:3px; font-family:'Inter',sans-serif; font-weight:600; }
        .pill-s2  { background:var(--purple-bg); color:var(--purple); font-size:10px; padding:1px 6px; border-radius:3px; font-family:'Inter',sans-serif; font-weight:600; }

        .trade-log-wrap { max-height:500px; overflow-y:auto; }
    </style>
</head>
<body>

<div class="header">
    <div>
        <h1>Turtle Trading System 1+2 — Multi-Sector Portfolio V3</h1>
        <p>${{ "{:,.0f}".format(initial) }} starting · Compounding · Max {{ max_pos }} positions · Nasdaq100 + Sector ETFs + Commodities + Bonds</p>
    </div>
    <div class="badges">
        <span class="badge b-blue">S1: 20d/10d</span>
        <span class="badge b-purple">S2: 55d/20d</span>
        <span class="badge b-teal">{{ universe_size }} instruments</span>
        <span class="badge b-amber">ATR Sizing + Compounding</span>
        <span class="badge b-green">{{ years }}yr backtest</span>
    </div>
</div>

<!-- KPI -->
<div class="kpi-bar">
    <div class="kpi">
        <div class="kpi-label">Starting Capital</div>
        <div class="kpi-value">${{ "{:,.0f}".format(initial) }}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Final Equity</div>
        <div class="kpi-value {{ 'pos' if s.final_equity > initial else 'neg' }}">${{ "{:,.0f}".format(s.final_equity) }}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Total Return</div>
        <div class="kpi-value {{ 'pos' if s.total_return > 0 else 'neg' }}">{{ s.total_return }}%</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">CAGR</div>
        <div class="kpi-value {{ 'pos' if s.cagr > 0 else 'neg' }}">{{ s.cagr }}%</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value">{{ s.total_trades }}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value {{ 'pos' if s.win_rate >= 50 else 'neg' }}">{{ s.win_rate }}%</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Payoff Ratio</div>
        <div class="kpi-value {{ 'pos' if s.payoff >= 1 else 'neg' }}">{{ s.payoff }}x</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Avg Hold</div>
        <div class="kpi-value">{{ s.avg_hold }}d</div>
    </div>
</div>

<!-- System comparison -->
<div class="sys-bar">
    <div class="sys-card">
        <div class="sys-title sys-s1">System 1 — Fast (20d entry / 10d exit)</div>
        <div class="sys-row"><span class="sys-key">Trades</span><span>{{ s.s1_trades }}</span></div>
        <div class="sys-row"><span class="sys-key">Win Rate</span><span class="{{ 'pos' if s.s1_wr >= 50 else 'neg' }}">{{ s.s1_wr }}%</span></div>
        <div class="sys-row"><span class="sys-key">Total P&L</span><span class="{{ 'pos' if s.s1_pnl >= 0 else 'neg' }}">${{ "{:,.0f}".format(s.s1_pnl) }}</span></div>
    </div>
    <div class="sys-card">
        <div class="sys-title sys-s2">System 2 — Slow (55d entry / 20d exit)</div>
        <div class="sys-row"><span class="sys-key">Trades</span><span>{{ s.s2_trades }}</span></div>
        <div class="sys-row"><span class="sys-key">Win Rate</span><span class="{{ 'pos' if s.s2_wr >= 50 else 'neg' }}">{{ s.s2_wr }}%</span></div>
        <div class="sys-row"><span class="sys-key">Total P&L</span><span class="{{ 'pos' if s.s2_pnl >= 0 else 'neg' }}">${{ "{:,.0f}".format(s.s2_pnl) }}</span></div>
    </div>
</div>

<!-- Equity curve -->
<div class="chart-card">
    <div class="chart-title">PORTFOLIO EQUITY CURVE</div>
    <div class="chart-wrap"><canvas id="eqChart"></canvas></div>
</div>

<!-- Ticker + Trade log -->
<div class="two-col">
    <div class="card">
        <div class="card-header">Per-Instrument Performance</div>
        <table>
            <thead><tr>
                <th class="left">Ticker</th>
                <th class="left">Sector</th>
                <th>Trades</th>
                <th>Win%</th>
                <th>Total P&L</th>
            </tr></thead>
            <tbody>
            {% for t in ticker_summary %}
            <tr>
                <td class="label">{{ t.ticker }}</td>
                <td class="sub">{{ t.label }}</td>
                <td>{{ t.trades }}</td>
                <td class="{{ 'pos' if t.win_rate >= 50 else 'neg' }}">{{ t.win_rate }}%</td>
                <td><span class="{{ 'pill-pos' if t.total_pnl >= 0 else 'pill-neg' }}">${{ "{:,.0f}".format(t.total_pnl) }}</span></td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="card">
        <div class="card-header">Trade Log (most recent 50)</div>
        <div class="trade-log-wrap">
        <table>
            <thead><tr>
                <th class="left">Ticker</th>
                <th>Sys</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Days</th>
                <th>Return</th>
                <th>P&L</th>
                <th>Equity</th>
            </tr></thead>
            <tbody>
            {% for t in recent_trades %}
            <tr>
                <td class="label">{{ t.ticker }}</td>
                <td><span class="pill-{{ t.system.lower() }}">{{ t.system }}</span></td>
                <td class="muted">{{ t.entry_date.strftime('%y-%m-%d') }}</td>
                <td class="muted">{{ t.exit_date.strftime('%y-%m-%d') }}</td>
                <td>{{ t.hold_days }}</td>
                <td><span class="{{ 'pill-pos' if t.return_pct > 0 else 'pill-neg' }}">{{ t.return_pct }}%</span></td>
                <td class="{{ 'pos' if t.pnl > 0 else 'neg' }}">${{ "{:,.0f}".format(t.pnl) }}</td>
                <td>${{ "{:,.0f}".format(t.equity_after) }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        </div>
    </div>
</div>

<script>
const labels = {{ equity_dates | tojson }};
const values = {{ equity_values | tojson }};
new Chart(document.getElementById('eqChart').getContext('2d'), {
    type: 'line',
    data: {
        labels,
        datasets: [{
            data: values,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37,99,235,0.06)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0.1,
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { ticks: { maxTicksLimit:12, font:{family:'JetBrains Mono',size:10}, color:'#9ca3af' }, grid:{color:'#f3f4f6'} },
            y: { ticks: { font:{family:'JetBrains Mono',size:10}, color:'#9ca3af', callback: v=>'$'+v.toLocaleString() }, grid:{color:'#f3f4f6'} }
        }
    }
});
</script>
</body>
</html>
"""

# =============================================================================
# SECTION 8: APP ENTRY POINT
# =============================================================================

@app.route("/")
def dashboard():
    all_data = {}
    for ticker in SECTOR_ETFS:
        df = fetch_and_cache(ticker)
        if df is None or df.empty:
            continue
        df = calculate_indicators(df)
        if df.empty:
            continue
        all_data[ticker] = df

    trades, equity_curve, final_equity = run_portfolio_backtest(all_data)

    s            = summarize(trades, INITIAL_CAPITAL, final_equity)
    ticker_summ  = per_ticker_summary(trades)
    recent_trades = list(reversed(trades[-50:]))

    eq_df    = pd.DataFrame(equity_curve).set_index('date')
    step     = max(1, len(eq_df) // 500)
    eq_s     = eq_df.iloc[::step]
    eq_dates = [d.strftime('%Y-%m-%d') for d in eq_s.index]
    eq_vals  = [round(v, 2) for v in eq_s['equity'].tolist()]

    return render_template_string(
        HTML_TEMPLATE,
        s=s,
        ticker_summary=ticker_summ,
        recent_trades=recent_trades,
        equity_dates=eq_dates,
        equity_values=eq_vals,
        initial=INITIAL_CAPITAL,
        max_pos=MAX_POSITIONS,
        years=BACKTEST_YEARS,
        universe_size=len(all_data),
    )

if __name__ == "__main__":
    app.run(debug=False, port=5001)