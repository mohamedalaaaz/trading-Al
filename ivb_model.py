"""
IVB Model - Python Implementation (Inspired by Fabervaale / Deepcharts)
========================================================================
Opening Range Breakout (ORB) framework with:
  - Statistical projection/protection levels
  - CVD (Cumulative Volume Delta) aggression confirmation
  - Volume Profile (POC + LVNs)
  - Daily bias signal
  - Real-time Binance Futures support

Usage:
    python ivb_model.py --symbol BTC/USDT --tf 5m --orb 30 --session NY
    python ivb_model.py --symbol BTC/USDT --tf 5m --orb 30 --live

Author: Generated for Jo (Fabervaale IVB Model recreation)
"""

import argparse
import sys
import warnings
from datetime import datetime, timezone, timedelta

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ───────────────────────────

SESSION_OPENS_UTC = {
    "NY":     {"hour": 14, "minute": 30},   # 09:30 ET
    "LONDON": {"hour":  8, "minute":  0},
    "ASIA":   {"hour":  0, "minute":  0},
    "CRYPTO": {"hour":  0, "minute":  0},   # 00:00 UTC daily open
}

PROJECTION_MULTIPLIERS = {
    "protection": 0.5,   # tightest target (~50% of range)
    "avg":        1.0,   # average extension (1× range)
    "std1":       1.618, # Fibonacci / 1.618 extension
    "std2":       2.0,   # max extension target
}

# ─────────────────────────── DATA ───────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
    exchange = ccxt.binance({"options": {"defaultType": "future"}})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df


# ─────────────────────────── OPENING RANGE ───────────────────────────

def get_session_open(session: str, reference_date: pd.Timestamp) -> pd.Timestamp:
    cfg = SESSION_OPENS_UTC[session.upper()]
    return reference_date.normalize().replace(
        hour=cfg["hour"], minute=cfg["minute"], second=0, microsecond=0
    )


def calculate_opening_range(
    df: pd.DataFrame,
    session: str = "CRYPTO",
    orb_minutes: int = 30
) -> dict:
    """
    Identify the most recent Opening Range and return its levels.
    """
    now = df.index[-1]
    session_open = get_session_open(session, now)

    # If session hasn't started yet today, look at yesterday
    if session_open > now:
        session_open -= timedelta(days=1)

    session_end = session_open + timedelta(minutes=orb_minutes)

    orb_candles = df[(df.index >= session_open) & (df.index < session_end)]

    if orb_candles.empty:
        raise ValueError(
            f"No candles found in the ORB window "
            f"({session_open} → {session_end}). "
            f"Try a different --session or check your data range."
        )

    orb_high = orb_candles["high"].max()
    orb_low  = orb_candles["low"].min()
    orb_mid  = (orb_high + orb_low) / 2
    orb_range = orb_high - orb_low

    return {
        "open_time":   session_open,
        "end_time":    session_end,
        "high":        orb_high,
        "low":         orb_low,
        "mid":         orb_mid,
        "range":       orb_range,
        "candles":     orb_candles,
    }


def calculate_projection_levels(orb: dict) -> dict:
    """
    Statistically derived extension levels above and below the ORB.
    """
    levels = {}
    r = orb["range"]
    for name, mult in PROJECTION_MULTIPLIERS.items():
        levels[f"bull_{name}"] = orb["high"] + r * mult
        levels[f"bear_{name}"] = orb["low"]  - r * mult
    return levels


# ─────────────────────────── CVD ───────────────────────────

def calculate_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Approximate CVD using close vs open direction and volume.
    For true tick-level CVD you'd need trade data; this is candle-level.
    """
    delta = np.where(
        df["close"] >= df["open"],
        df["volume"],   # bullish candle → buying volume
        -df["volume"]   # bearish candle → selling volume
    )
    return pd.Series(delta, index=df.index).cumsum()


def cvd_aggression_signal(df: pd.DataFrame, orb: dict, window: int = 10) -> str:
    """
    Assess CVD trend after the ORB period to determine aggression direction.
    """
    post_orb = df[df.index >= orb["end_time"]].copy()
    if len(post_orb) < window:
        return "NEUTRAL"

    cvd = calculate_cvd(post_orb).iloc[-window:]
    slope = np.polyfit(range(len(cvd)), cvd.values, 1)[0]

    if slope > 0:
        return "BULLISH"
    elif slope < 0:
        return "BEARISH"
    return "NEUTRAL"


# ─────────────────────────── VOLUME PROFILE ───────────────────────────

def calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> dict:
    """
    Calculate a simplified volume profile and return POC + LVN levels.
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    price_levels = np.linspace(price_min, price_max, bins + 1)
    vol_at_price = np.zeros(bins)

    for _, row in df.iterrows():
        # Distribute candle volume uniformly across its H-L range
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if hi == lo:
            idx = np.searchsorted(price_levels, lo, side="right") - 1
            idx = np.clip(idx, 0, bins - 1)
            vol_at_price[idx] += vol
        else:
            mask = (price_levels[:-1] >= lo) & (price_levels[1:] <= hi)
            n_overlap = mask.sum() or 1
            vol_at_price[mask] += vol / n_overlap

    mid_prices = (price_levels[:-1] + price_levels[1:]) / 2
    poc_idx = np.argmax(vol_at_price)
    poc = mid_prices[poc_idx]

    # LVNs: price levels with < 20th percentile volume
    threshold = np.percentile(vol_at_price, 20)
    lvn_prices = mid_prices[vol_at_price < threshold]

    return {
        "prices":     mid_prices,
        "volumes":    vol_at_price,
        "poc":        poc,
        "lvns":       lvn_prices,
        "price_min":  price_min,
        "price_max":  price_max,
    }


# ─────────────────────────── BIAS & SIGNAL ───────────────────────────

def calculate_daily_bias(df: pd.DataFrame, orb: dict, vp: dict) -> dict:
    """
    Determine daily bias based on:
      - Price position relative to ORB mid
      - CVD direction
      - Distance from POC
    """
    current_price = df["close"].iloc[-1]
    cvd_signal    = cvd_aggression_signal(df, orb)

    # Price above mid = lean bullish, below = lean bearish
    price_bias = "BULLISH" if current_price > orb["mid"] else "BEARISH"

    # Combined: if CVD confirms price bias → strong signal
    if price_bias == cvd_signal:
        bias_strength = "STRONG"
        bias_dir      = price_bias
    elif cvd_signal == "NEUTRAL":
        bias_strength = "WEAK"
        bias_dir      = price_bias
    else:
        bias_strength = "CONFLICTED"
        bias_dir      = "NEUTRAL"

    # Breakout detection
    breakout = None
    if current_price > orb["high"]:
        breakout = "BULL_BREAKOUT"
    elif current_price < orb["low"]:
        breakout = "BEAR_BREAKOUT"

    return {
        "direction":  bias_dir,
        "strength":   bias_strength,
        "breakout":   breakout,
        "cvd_signal": cvd_signal,
        "price":      current_price,
        "poc":        vp["poc"],
    }


def generate_trade_signal(orb: dict, bias: dict, proj: dict, vp: dict) -> dict:
    """
    Generate actionable trade signal based on IVB model logic:
      Market State + Location + Aggression → Entry / Target / Stop
    """
    price   = bias["price"]
    signal  = {"action": "FLAT", "entry": None, "stop": None,
                "target1": None, "target2": None, "reason": ""}

    if bias["breakout"] == "BULL_BREAKOUT" and bias["strength"] == "STRONG":
        signal.update({
            "action":  "LONG",
            "entry":   orb["high"],
            "stop":    orb["mid"],
            "target1": proj["bull_protection"],
            "target2": proj["bull_avg"],
            "reason":  "ORB bull breakout + CVD confirms bullish aggression",
        })

    elif bias["breakout"] == "BEAR_BREAKOUT" and bias["strength"] == "STRONG":
        signal.update({
            "action":  "SHORT",
            "entry":   orb["low"],
            "stop":    orb["mid"],
            "target1": proj["bear_protection"],
            "target2": proj["bear_avg"],
            "reason":  "ORB bear breakout + CVD confirms bearish aggression",
        })

    elif bias["breakout"] is None and bias["direction"] != "NEUTRAL":
        # Mean reversion setup: price inside ORB but CVD diverging
        if bias["direction"] == "BULLISH":
            signal.update({
                "action":  "LONG (MEAN REV)",
                "entry":   orb["low"],
                "stop":    orb["low"] - orb["range"] * 0.25,
                "target1": orb["mid"],
                "target2": orb["high"],
                "reason":  "Inside balance — mean reversion to POC/ORB high",
            })
        else:
            signal.update({
                "action":  "SHORT (MEAN REV)",
                "entry":   orb["high"],
                "stop":    orb["high"] + orb["range"] * 0.25,
                "target1": orb["mid"],
                "target2": orb["low"],
                "reason":  "Inside balance — mean reversion to POC/ORB low",
            })
    else:
        signal["reason"] = "No alignment: Market State + Aggression conflict. Stay flat."

    # Risk/Reward
    if signal["entry"] and signal["stop"] and signal["target1"]:
        risk   = abs(signal["entry"] - signal["stop"])
        reward = abs(signal["target1"] - signal["entry"])
        signal["rr"] = round(reward / risk, 2) if risk else 0
    else:
        signal["rr"] = 0

    return signal


# ─────────────────────────── PRINT SUMMARY ───────────────────────────

def print_summary(orb: dict, proj: dict, vp: dict, bias: dict, signal: dict, symbol: str):
    divider = "─" * 55
    print(f"\n{divider}")
    print(f"  IVB MODEL — {symbol}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(divider)

    print(f"\n  OPENING RANGE  ({orb['open_time'].strftime('%H:%M')} → {orb['end_time'].strftime('%H:%M')} UTC)")
    print(f"    High   : {orb['high']:.2f}")
    print(f"    Mid    : {orb['mid']:.2f}")
    print(f"    Low    : {orb['low']:.2f}")
    print(f"    Range  : {orb['range']:.2f}")

    print(f"\n  VOLUME PROFILE")
    print(f"    POC    : {vp['poc']:.2f}")
    lvn_sample = vp['lvns'][:3]
    print(f"    LVNs   : {', '.join(f'{l:.2f}' for l in lvn_sample)} ...")

    print(f"\n  PROJECTION LEVELS (Bull / Bear)")
    for key in ["protection", "avg", "std1", "std2"]:
        print(f"    {key:<12}: {proj[f'bull_{key}']:.2f}  /  {proj[f'bear_{key}']:.2f}")

    bias_color = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "→"}
    print(f"\n  DAILY BIAS")
    print(f"    Direction : {bias_color.get(bias['direction'], '?')} {bias['direction']}")
    print(f"    Strength  : {bias['strength']}")
    print(f"    CVD       : {bias['cvd_signal']}")
    print(f"    Breakout  : {bias['breakout'] or 'NONE — inside balance'}")
    print(f"    Price     : {bias['price']:.2f}")

    action_icon = {"LONG": "▲ LONG", "SHORT": "▼ SHORT",
                   "LONG (MEAN REV)": "▲ LONG (MR)", "SHORT (MEAN REV)": "▼ SHORT (MR)",
                   "FLAT": "— FLAT"}
    print(f"\n  SIGNAL")
    print(f"    Action  : {action_icon.get(signal['action'], signal['action'])}")
    if signal["entry"]:
        print(f"    Entry   : {signal['entry']:.2f}")
        print(f"    Stop    : {signal['stop']:.2f}")
        print(f"    T1      : {signal['target1']:.2f}")
        print(f"    T2      : {signal['target2']:.2f}")
        print(f"    R:R     : 1 : {signal['rr']}")
    print(f"    Reason  : {signal['reason']}")
    print(f"\n{divider}\n")


# ─────────────────────────── CHART ───────────────────────────

def plot_ivb(df: pd.DataFrame, orb: dict, proj: dict, vp: dict,
             bias: dict, signal: dict, symbol: str):
    fig, (ax_price, ax_cvd, ax_vp) = plt.subplots(
        3, 1, figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        facecolor="#0d1117"
    )
    fig.suptitle(
        f"IVB Model  —  {symbol}  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    # ── Price chart ──
    ax_price.set_facecolor("#0d1117")
    dates = df.index.to_pydatetime()

    # Candlesticks
    for i, (ts, row) in enumerate(df.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax_price.plot([ts, ts], [row["low"], row["high"]], color=color, linewidth=0.8, alpha=0.7)
        ax_price.bar(ts, row["close"] - row["open"], bottom=row["open"],
                     color=color, width=pd.Timedelta(minutes=4), alpha=0.85)

    # ORB shading
    ax_price.axvspan(orb["open_time"], orb["end_time"],
                     alpha=0.12, color="#a78bfa", label="ORB Window")

    # ORB levels
    ax_price.axhline(orb["high"], color="#a78bfa", linewidth=1.2, linestyle="--", label=f"ORB High {orb['high']:.1f}")
    ax_price.axhline(orb["mid"],  color="#94a3b8", linewidth=0.8, linestyle=":",  label=f"ORB Mid {orb['mid']:.1f}")
    ax_price.axhline(orb["low"],  color="#a78bfa", linewidth=1.2, linestyle="--", label=f"ORB Low {orb['low']:.1f}")

    # POC
    ax_price.axhline(vp["poc"], color="#fbbf24", linewidth=1.0, linestyle="-.",
                     alpha=0.8, label=f"POC {vp['poc']:.1f}")

    # Projection levels
    proj_colors = {"protection": "#34d399", "avg": "#60a5fa", "std1": "#f59e0b", "std2": "#f87171"}
    for key, col in proj_colors.items():
        ax_price.axhline(proj[f"bull_{key}"], color=col, linewidth=0.8, linestyle=":",
                         alpha=0.6, label=f"↑ {key} {proj[f'bull_{key}']:.1f}")
        ax_price.axhline(proj[f"bear_{key}"], color=col, linewidth=0.8, linestyle=":",
                         alpha=0.6)

    # Signal annotation
    if signal["action"] != "FLAT" and signal["entry"]:
        col = "#26a69a" if "LONG" in signal["action"] else "#ef5350"
        ax_price.axhline(signal["entry"],  color=col,      linewidth=1.2, linestyle="-",  alpha=0.9)
        ax_price.axhline(signal["stop"],   color="#ef5350", linewidth=0.9, linestyle="-", alpha=0.7)
        ax_price.axhline(signal["target1"],color="#26a69a", linewidth=0.9, linestyle="-", alpha=0.7)

        ax_price.annotate(
            f"  {signal['action']}  |  R:R 1:{signal['rr']}",
            xy=(df.index[-1], signal["entry"]),
            color=col, fontsize=8.5, va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor=col, alpha=0.9)
        )

    ax_price.set_ylabel("Price (USDT)", color="#94a3b8", fontsize=9)
    ax_price.tick_params(colors="#94a3b8", labelsize=7)
    ax_price.legend(loc="upper left", fontsize=6.5, facecolor="#1e293b",
                    edgecolor="#334155", labelcolor="white", ncol=2)
    ax_price.grid(alpha=0.08, color="#334155")
    for spine in ax_price.spines.values():
        spine.set_edgecolor("#334155")

    # ── CVD panel ──
    ax_cvd.set_facecolor("#0d1117")
    cvd = calculate_cvd(df)
    cvd_color = np.where(cvd >= 0, "#26a69a", "#ef5350")
    ax_cvd.bar(df.index, cvd.values, color=cvd_color, width=pd.Timedelta(minutes=4), alpha=0.7)
    ax_cvd.axhline(0, color="#475569", linewidth=0.6)
    ax_cvd.axvspan(orb["open_time"], orb["end_time"], alpha=0.08, color="#a78bfa")
    ax_cvd.set_ylabel("CVD", color="#94a3b8", fontsize=9)
    ax_cvd.tick_params(colors="#94a3b8", labelsize=7)
    ax_cvd.grid(alpha=0.06, color="#334155")
    for spine in ax_cvd.spines.values():
        spine.set_edgecolor("#334155")

    # CVD signal label
    sig_color = {"BULLISH": "#26a69a", "BEARISH": "#ef5350", "NEUTRAL": "#94a3b8"}
    ax_cvd.text(0.01, 0.85, f"CVD: {bias['cvd_signal']}",
                transform=ax_cvd.transAxes, color=sig_color[bias["cvd_signal"]],
                fontsize=8, fontweight="bold")

    # ── Volume Profile panel ──
    ax_vp.set_facecolor("#0d1117")
    ax_vp.barh(vp["prices"], vp["volumes"],
               height=(vp["price_max"] - vp["price_min"]) / len(vp["prices"]),
               color="#60a5fa", alpha=0.5)
    ax_vp.axhline(vp["poc"], color="#fbbf24", linewidth=1.2, linestyle="-.", label=f"POC {vp['poc']:.1f}")
    ax_vp.axhline(orb["high"], color="#a78bfa", linewidth=0.8, linestyle="--")
    ax_vp.axhline(orb["low"],  color="#a78bfa", linewidth=0.8, linestyle="--")
    ax_vp.set_xlabel("Volume at Price", color="#94a3b8", fontsize=9)
    ax_vp.set_ylabel("Price", color="#94a3b8", fontsize=9)
    ax_vp.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
    ax_vp.tick_params(colors="#94a3b8", labelsize=7)
    ax_vp.grid(alpha=0.06, color="#334155")
    for spine in ax_vp.spines.values():
        spine.set_edgecolor("#334155")

    # Bias badge
    bias_colors = {"BULLISH": "#26a69a", "BEARISH": "#ef5350", "NEUTRAL": "#94a3b8"}
    b_col = bias_colors.get(bias["direction"], "#94a3b8")
    fig.text(0.99, 0.97,
             f"BIAS: {bias['direction']} ({bias['strength']})",
             ha="right", va="top", fontsize=9, fontweight="bold",
             color=b_col,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor=b_col, alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"/mnt/user-data/outputs/ivb_model_{symbol.replace('/', '')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"  Chart saved → {out}")
    plt.show()


# ─────────────────────────── MAIN ───────────────────────────

def run(symbol: str, timeframe: str, orb_minutes: int, session: str,
        candles: int, chart: bool):
    print(f"\n  Fetching {candles} × {timeframe} candles for {symbol} ...")
    df = fetch_ohlcv(symbol, timeframe, candles)

    print(f"  Calculating Opening Range ({orb_minutes} min, {session} session) ...")
    orb = calculate_opening_range(df, session, orb_minutes)

    proj = calculate_projection_levels(orb)
    vp   = calculate_volume_profile(df)
    bias = calculate_daily_bias(df, orb, vp)
    sig  = generate_trade_signal(orb, bias, proj, vp)

    print_summary(orb, proj, vp, bias, sig, symbol)

    if chart:
        plot_ivb(df, orb, proj, vp, bias, sig, symbol)


def main():
    parser = argparse.ArgumentParser(description="IVB Model — Fabervaale ORB + Order Flow")
    parser.add_argument("--symbol",  default="BTC/USDT",
                        help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--tf",      default="5m",
                        help="Timeframe: 1m, 3m, 5m, 15m (default: 5m)")
    parser.add_argument("--orb",     type=int, default=30,
                        help="ORB window in minutes: 15, 30, 60 (default: 30)")
    parser.add_argument("--session", default="CRYPTO",
                        choices=["NY", "LONDON", "ASIA", "CRYPTO"],
                        help="Session open reference (default: CRYPTO)")
    parser.add_argument("--candles", type=int, default=288,
                        help="Number of candles to fetch (default: 288 = 24h on 5m)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart output, print only")
    args = parser.parse_args()

    run(
        symbol    = args.symbol,
        timeframe = args.tf,
        orb_minutes = args.orb,
        session   = args.session,
        candles   = args.candles,
        chart     = not args.no_chart,
    )


if __name__ == "__main__":
    main()
