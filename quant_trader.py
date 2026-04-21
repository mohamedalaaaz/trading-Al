"""
=============================================================================
  QUANT AUTO-TRADER  |  BTC/USDT Binance Futures
=============================================================================
  Math Engine:
    - Kalman Filter          → dynamic price mean tracking
    - Ornstein-Uhlenbeck     → mean-reversion speed & fair value
    - Hurst Exponent         → regime detection (trending vs mean-rev)
    - Z-Score signals        → entry / exit thresholds
    - Kelly Criterion        → optimal position sizing
    - GARCH(1,1)             → realized + forecast volatility
    - CVaR / VaR             → tail-risk position limits
    - VWAP ± σ bands         → institutional reference levels
    - RSI + MACD             → momentum confirmation
    - Ensemble signal score  → weighted vote across all signals

  Usage:
    pip install websocket-client requests numpy pandas scipy

    python quant_trader.py --paper --account 10000 --symbol BTCUSDT --tf 1m
    python quant_trader.py --paper --account 10000 --symbol BTCUSDT --tf 5m
=============================================================================
"""

import argparse
import json
import math
import time
import threading
import logging
import sys
import os
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import websocket

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("QuantTrader")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_REST   = "https://fapi.binance.com"
BINANCE_WS     = "wss://fstream.binance.com/ws"
MIN_BARS       = 60          # bars needed before any trade
MAX_LEVERAGE   = 5
MAX_RISK_PCT   = 0.02        # max 2% account per trade
KELLY_FRACTION = 0.25        # use 25% of Kelly (conservative)
HURST_WINDOW   = 50          # bars for Hurst calculation
ZSCORE_ENTRY   = 2.0         # enter when |z| > 2.0
ZSCORE_EXIT    = 0.4         # exit when |z| < 0.4
MAKER_FEE      = 0.0002      # 0.02% Binance Futures maker
TAKER_FEE      = 0.0005      # 0.05% Binance Futures taker

# ─────────────────────────────────────────────────────────────────────────────
#  1.  MATH LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    1-D Kalman filter for price mean estimation.
    State: [price, velocity]
    """
    def __init__(self, process_noise: float = 0.01, obs_noise: float = 1.0):
        self.Q = process_noise   # process noise variance
        self.R = obs_noise       # observation noise variance
        self.x = None            # state estimate [price, vel]
        self.P = np.eye(2) * 1.0 # error covariance

        self.A = np.array([[1, 1],   # state transition
                           [0, 1]])
        self.H = np.array([[1, 0]])  # observation matrix

    def update(self, price: float) -> float:
        """Returns Kalman-filtered price estimate."""
        if self.x is None:
            self.x = np.array([price, 0.0])
            return price

        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + np.eye(2) * self.Q

        # Update
        y  = price - (self.H @ x_pred)[0]
        S  = (self.H @ P_pred @ self.H.T + self.R)[0, 0]
        K  = (P_pred @ self.H.T) / S

        self.x = x_pred + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ P_pred

        return float(self.x[0])   # filtered price


def hurst_exponent(series: np.ndarray) -> float:
    """
    Hurst Exponent via R/S analysis.
      H < 0.45  →  mean-reverting
      H ≈ 0.50  →  random walk
      H > 0.55  →  trending
    """
    n = len(series)
    if n < 20:
        return 0.5

    lags    = range(2, min(n // 2, 20))
    tau     = []
    rs_vals = []

    for lag in lags:
        segments = [series[i:i + lag] for i in range(0, n - lag, lag)]
        rs_list  = []
        for seg in segments:
            mean  = np.mean(seg)
            devs  = np.cumsum(seg - mean)
            r     = np.max(devs) - np.min(devs)
            s     = np.std(seg, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            tau.append(lag)

    if len(tau) < 2:
        return 0.5

    log_tau = np.log(tau)
    log_rs  = np.log(rs_vals)
    H, _    = np.polyfit(log_tau, log_rs, 1)
    return float(np.clip(H, 0.01, 0.99))


def ou_parameters(prices: np.ndarray, dt: float = 1.0):
    """
    Fit Ornstein-Uhlenbeck process to price series.
    dX = κ(μ - X)dt + σ dW
    Returns: kappa (mean-reversion speed), mu (long-run mean), sigma (vol)
    """
    x   = prices[:-1]
    y   = prices[1:]
    n   = len(x)

    # OLS regression: y = a + b*x
    sx  = np.sum(x)
    sy  = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, float(np.mean(prices)), float(np.std(prices))

    a = (sy * sxx - sx * sxy) / denom
    b = (n  * sxy - sx * sy)  / denom

    kappa = -math.log(b) / dt if b > 0 else 0.001
    mu    = a / (1 - b) if abs(1 - b) > 1e-12 else float(np.mean(prices))
    resid = y - (a + b * x)
    sigma = float(np.std(resid, ddof=1)) * math.sqrt(2 * kappa / (1 - b**2)) \
            if b > 0 else float(np.std(resid, ddof=1))

    return max(kappa, 0.0), float(mu), max(sigma, 1e-9)


def garch11_volatility(returns: np.ndarray,
                        omega: float = 1e-6,
                        alpha: float = 0.1,
                        beta:  float = 0.85) -> float:
    """
    GARCH(1,1) one-step-ahead volatility forecast.
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    """
    if len(returns) < 5:
        return float(np.std(returns)) if len(returns) > 1 else 0.01

    var = float(np.var(returns))
    for r in returns:
        var = omega + alpha * r**2 + beta * var
    return math.sqrt(max(var, 1e-12))


def zscore(series: np.ndarray, window: int = 20) -> float:
    """Rolling z-score of last value vs window."""
    if len(series) < window:
        return 0.0
    sub  = series[-window:]
    mean = np.mean(sub)
    std  = np.std(sub, ddof=1)
    return float((series[-1] - mean) / std) if std > 1e-12 else 0.0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: f* = (p·b - q) / b
      p = win_rate, q = 1-p, b = avg_win/avg_loss
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f = (win_rate * b - q) / b
    return float(np.clip(f * KELLY_FRACTION, 0.0, 1.0))


def compute_var_cvar(returns: np.ndarray, confidence: float = 0.95):
    """
    Historical VaR and CVaR at given confidence level.
    Returns: (VaR, CVaR) as positive loss fractions
    """
    if len(returns) < 10:
        return 0.05, 0.10
    sorted_r = np.sort(returns)
    idx      = int((1 - confidence) * len(sorted_r))
    var      = float(-sorted_r[idx])
    cvar     = float(-np.mean(sorted_r[:idx + 1]))
    return max(var, 0.0), max(cvar, 0.0)


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains)
    avg_l  = np.mean(losses)
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return float(100 - 100 / (1 + rs))


def vwap_bands(closes: np.ndarray, volumes: np.ndarray, n_std: float = 2.0):
    """
    VWAP with standard deviation bands.
    Returns: (vwap, upper_band, lower_band)
    """
    if len(closes) == 0:
        return 0.0, 0.0, 0.0
    cum_vol  = np.cumsum(volumes)
    cum_tp_v = np.cumsum(closes * volumes)
    vwap     = cum_tp_v[-1] / (cum_vol[-1] + 1e-12)
    variance = np.sum(volumes * (closes - vwap) ** 2) / (cum_vol[-1] + 1e-12)
    sigma    = math.sqrt(max(variance, 0.0))
    return float(vwap), float(vwap + n_std * sigma), float(vwap - n_std * sigma)


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and signal line. Returns (macd_line, signal_line)."""
    if len(prices) < slow + signal:
        return 0.0, 0.0
    p     = pd.Series(prices)
    ema_f = p.ewm(span=fast, adjust=False).mean()
    ema_s = p.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  2.  SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Combines all quantitative signals into a single composite score [-1, +1].
    Positive → Long bias  |  Negative → Short bias  |  Near zero → Flat
    """

    WEIGHTS = {
        "zscore_mr"   : 0.30,   # mean-reversion z-score
        "ou_deviation": 0.20,   # OU deviation from fair value
        "hurst"       : 0.15,   # regime weight modifier
        "rsi"         : 0.15,   # momentum filter
        "macd"        : 0.10,   # momentum confirmation
        "vwap"        : 0.10,   # institutional level
    }

    def __init__(self):
        self.kalman  = KalmanFilter(process_noise=0.005, obs_noise=1.5)
        self.history : deque = deque(maxlen=200)  # (close, volume)

    def update(self, close: float, volume: float) -> dict:
        """Feed one bar. Returns signal dict."""
        filtered = self.kalman.update(close)
        self.history.append((close, volume))

        n = len(self.history)
        result = {
            "score"         : 0.0,
            "hurst"         : 0.5,
            "zscore"        : 0.0,
            "ou_mu"         : close,
            "garch_vol"     : 0.01,
            "rsi"           : 50.0,
            "macd"          : 0.0,
            "vwap"          : close,
            "regime"        : "RANDOM",
            "filtered_price": filtered,
            "bars"          : n,
        }

        if n < MIN_BARS:
            return result

        closes  = np.array([x[0] for x in self.history])
        volumes = np.array([x[1] for x in self.history])
        returns = np.diff(np.log(closes))

        # ── Hurst exponent ──────────────────────────────────────────────────
        h = hurst_exponent(closes[-HURST_WINDOW:])
        if   h < 0.45: regime = "MEAN-REV"
        elif h > 0.55: regime = "TRENDING"
        else:          regime = "RANDOM"
        result["hurst"]  = h
        result["regime"] = regime

        # ── GARCH volatility ────────────────────────────────────────────────
        garch_vol = garch11_volatility(returns[-60:])
        result["garch_vol"] = garch_vol

        # ── Z-score mean-reversion ──────────────────────────────────────────
        z = zscore(closes, window=20)
        result["zscore"] = z

        # ── Ornstein-Uhlenbeck ──────────────────────────────────────────────
        kappa, mu, ou_sigma = ou_parameters(closes[-40:])
        ou_z = (close - mu) / (ou_sigma + 1e-12)
        result["ou_mu"] = mu

        # ── RSI ─────────────────────────────────────────────────────────────
        rsi_val = rsi(closes)
        result["rsi"] = rsi_val

        # ── MACD ────────────────────────────────────────────────────────────
        macd_line, sig_line = macd(closes)
        result["macd"] = macd_line - sig_line

        # ── VWAP ────────────────────────────────────────────────────────────
        vwap_val, vwap_up, vwap_dn = vwap_bands(closes[-50:], volumes[-50:])
        result["vwap"] = vwap_val

        # ─── Composite score ────────────────────────────────────────────────
        W = self.WEIGHTS

        # Mean-reversion z-score signal: fade extremes
        mr_signal   = float(np.clip(-z / 3.0, -1.0, 1.0))

        # OU deviation signal
        ou_signal   = float(np.clip(-ou_z / 3.0, -1.0, 1.0))

        # Hurst modifier: amplify MR signals when H < 0.45, trending when > 0.55
        if regime == "MEAN-REV":
            hurst_mod =  1.2
        elif regime == "TRENDING":
            hurst_mod = -0.5  # flip: follow trend, not mean-rev
        else:
            hurst_mod =  0.8

        # RSI signal: oversold → +1, overbought → -1
        rsi_signal  = float(np.clip(-(rsi_val - 50.0) / 40.0, -1.0, 1.0))

        # MACD signal: normalise by recent GARCH vol
        macd_signal = float(np.clip((macd_line - sig_line) / (close * garch_vol + 1e-9),
                                    -1.0, 1.0))

        # VWAP signal: below lower band → buy, above upper → sell
        if close < vwap_dn:
            vwap_signal =  1.0
        elif close > vwap_up:
            vwap_signal = -1.0
        else:
            vwap_signal =  0.0

        score = (W["zscore_mr"]    * mr_signal * hurst_mod +
                 W["ou_deviation"] * ou_signal              +
                 W["hurst"]        * 0.0                    +  # weight slot
                 W["rsi"]          * rsi_signal             +
                 W["macd"]         * macd_signal            +
                 W["vwap"]         * vwap_signal)

        result["score"] = float(np.clip(score, -1.0, 1.0))
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  3.  RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """Kelly + VaR/CVaR position sizing with daily drawdown guard."""

    def __init__(self, account_size: float):
        self.account    = account_size
        self.equity     = account_size
        self.peak_eq    = account_size
        self.trade_log  : list = []   # list of pnl values

    def max_drawdown(self) -> float:
        if self.equity < self.peak_eq:
            return (self.peak_eq - self.equity) / self.peak_eq
        return 0.0

    def position_size(self, price: float, signal_strength: float,
                      garch_vol: float) -> float:
        """
        Returns position size in base units (BTC).
        0 → no trade.
        """
        if self.max_drawdown() > 0.10:
            log.warning("Max drawdown 10% hit — trading halted")
            return 0.0

        if len(self.trade_log) >= 20:
            wins   = [p for p in self.trade_log[-50:] if p > 0]
            losses = [abs(p) for p in self.trade_log[-50:] if p < 0]
            wr     = len(wins) / (len(wins) + len(losses) + 1e-9)
            aw     = float(np.mean(wins))   if wins   else 0.0
            al     = float(np.mean(losses)) if losses else 0.01
            k      = kelly_fraction(wr, aw, al)
        else:
            k = 0.02   # conservative bootstrap fraction

        returns_arr = np.array(self.trade_log[-100:]) / self.equity \
                      if self.trade_log else np.zeros(10)
        var, cvar   = compute_var_cvar(returns_arr)

        # Risk budget: min(kelly, max_risk_pct, 1 - CVaR)
        risk_budget = min(k, MAX_RISK_PCT, max(0.005, 1.0 - cvar))

        # Volatility-scale: higher GARCH vol → smaller size
        vol_scale   = max(0.2, 1.0 - garch_vol * 50)

        # Signal-strength scale (|score| 0→1)
        sig_scale   = min(1.0, abs(signal_strength) * 2.0)

        dollar_risk = self.equity * risk_budget * vol_scale * sig_scale
        leverage    = min(MAX_LEVERAGE, max(1, int(2 + abs(signal_strength) * 3)))
        notional    = dollar_risk * leverage
        qty         = notional / price

        return float(max(qty, 0.0))

    def record_trade(self, pnl: float):
        self.trade_log.append(pnl)
        self.equity += pnl
        if self.equity > self.peak_eq:
            self.peak_eq = self.equity


# ─────────────────────────────────────────────────────────────────────────────
#  4.  PAPER TRADING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Position:
    def __init__(self, side: str, qty: float, entry: float, ts: str):
        self.side  = side    # "LONG" | "SHORT"
        self.qty   = qty
        self.entry = entry
        self.ts    = ts

    def pnl(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry) * self.qty
        else:
            return (self.entry - price) * self.qty

    def __repr__(self):
        return f"<{self.side} qty={self.qty:.6f} @ {self.entry:.2f}>"


class PaperTrader:
    def __init__(self, account_size: float):
        self.risk       = RiskManager(account_size)
        self.position   : Position | None = None
        self.total_pnl  = 0.0
        self.n_trades   = 0
        self.wins       = 0
        self.losses     = 0

    def _fee(self, price: float, qty: float) -> float:
        return price * qty * TAKER_FEE

    def open_position(self, side: str, price: float, qty: float):
        fee           = self._fee(price, qty)
        self.position = Position(side, qty, price,
                                 datetime.now(timezone.utc).strftime("%H:%M:%S"))
        self.risk.equity -= fee
        log.info(f"  OPEN  {side:5s}  qty={qty:.6f} BTC  @ {price:.2f}  "
                 f"notional=${price*qty:,.0f}  fee=${fee:.2f}")

    def close_position(self, price: float):
        if not self.position:
            return
        p   = self.position
        pnl = p.pnl(price) - self._fee(price, p.qty)
        self.risk.record_trade(pnl)
        self.total_pnl += pnl
        self.n_trades  += 1
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        log.info(f"  CLOSE {p.side:5s}  @ {price:.2f}  "
                 f"PnL=${pnl:+.2f}  Equity=${self.risk.equity:,.2f}  "
                 f"DD={self.risk.max_drawdown():.1%}")
        self.position = None

    def process(self, signal: dict, price: float):
        score    = signal["score"]
        garch_v  = signal["garch_vol"]
        n_bars   = signal["bars"]
        z        = signal["zscore"]
        regime   = signal["regime"]

        if n_bars < MIN_BARS:
            return

        # ── Exit logic ───────────────────────────────────────────────────────
        if self.position:
            exit_flag = False
            if self.position.side == "LONG"  and abs(z) < ZSCORE_EXIT:
                exit_flag = True
            if self.position.side == "SHORT" and abs(z) < ZSCORE_EXIT:
                exit_flag = True
            if self.position.side == "LONG"  and score < -0.3:
                exit_flag = True
            if self.position.side == "SHORT" and score >  0.3:
                exit_flag = True
            # Stop-loss: 1.5× GARCH vol
            sl = price * garch_v * 1.5 * math.sqrt(24)
            if self.position.pnl(price) < -abs(sl * self.position.qty):
                log.warning(f"  STOP LOSS hit  pnl={self.position.pnl(price):+.2f}")
                exit_flag = True

            if exit_flag:
                self.close_position(price)
                return

        # ── Entry logic ──────────────────────────────────────────────────────
        if not self.position:
            if score > 0.25 and abs(z) > ZSCORE_ENTRY:
                side = "LONG"
            elif score < -0.25 and abs(z) > ZSCORE_ENTRY:
                side = "SHORT"
            else:
                return

            qty = self.risk.position_size(price, score, garch_v)
            if qty * price < 10:   # min $10 notional
                return
            self.open_position(side, price, qty)

    def stats(self):
        wr = self.wins / self.n_trades if self.n_trades else 0
        return {
            "trades"    : self.n_trades,
            "win_rate"  : wr,
            "total_pnl" : self.total_pnl,
            "equity"    : self.risk.equity,
            "drawdown"  : self.risk.max_drawdown(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  5.  DATA FEED  (Binance Futures WebSocket — kline/candlestick)
# ─────────────────────────────────────────────────────────────────────────────

class BinanceFeed:
    def __init__(self, symbol: str, interval: str, on_candle):
        self.symbol    = symbol.lower()
        self.interval  = interval
        self.on_candle = on_candle
        self._ws       = None
        self._thread   = None
        self._running  = False

    def _on_message(self, ws, msg):
        data = json.loads(msg)
        k    = data.get("k", {})
        if not k.get("x"):   # only closed candles
            return
        candle = {
            "ts"    : k["t"],
            "open"  : float(k["o"]),
            "high"  : float(k["h"]),
            "low"   : float(k["l"]),
            "close" : float(k["c"]),
            "volume": float(k["v"]),
        }
        self.on_candle(candle)

    def _on_error(self, ws, err):
        log.error(f"WS error: {err}")

    def _on_close(self, ws, code, msg):
        log.warning("WS closed — reconnecting in 5 s")
        if self._running:
            time.sleep(5)
            self.start()

    def _on_open(self, ws):
        log.info(f"WS connected: {self.symbol}@kline_{self.interval}")

    def _load_history(self):
        """Bootstrap with REST history before WebSocket."""
        url    = f"{BINANCE_REST}/fapi/v1/klines"
        params = {"symbol": self.symbol.upper(),
                  "interval": self.interval,
                  "limit": 200}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            rows = r.json()
            log.info(f"Loaded {len(rows)} historical bars from REST")
            for row in rows[:-1]:   # exclude last (not closed yet)
                self.on_candle({
                    "ts"    : row[0],
                    "open"  : float(row[1]),
                    "high"  : float(row[2]),
                    "low"   : float(row[3]),
                    "close" : float(row[4]),
                    "volume": float(row[5]),
                })
        except Exception as e:
            log.error(f"REST history load failed: {e}")

    def start(self):
        self._running = True
        self._load_history()
        stream  = f"{self.symbol}@kline_{self.interval}"
        url     = f"{BINANCE_WS}/{stream}"
        self._ws = websocket.WebSocketApp(
            url,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
            on_open    = self._on_open,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"reconnect": 5},
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()


# ─────────────────────────────────────────────────────────────────────────────
#  6.  MAIN CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class QuantTrader:
    STAT_INTERVAL = 30   # print stats every N candles

    def __init__(self, symbol: str, interval: str, account: float, paper: bool):
        self.symbol    = symbol
        self.interval  = interval
        self.paper     = paper
        self.engine    = SignalEngine()
        self.trader    = PaperTrader(account) if paper else None
        self.feed      = BinanceFeed(symbol, interval, self._on_candle)
        self._candle_n = 0

    def _on_candle(self, c: dict):
        self._candle_n += 1
        price  = c["close"]
        vol    = c["volume"]
        ts_str = datetime.fromtimestamp(c["ts"] / 1000,
                                        tz=timezone.utc).strftime("%H:%M")

        # Compute signals
        sig = self.engine.update(price, vol)

        # Print bar info
        if sig["bars"] >= MIN_BARS:
            log.info(
                f"[{ts_str}] {self.symbol} {price:,.2f} | "
                f"score={sig['score']:+.3f}  z={sig['zscore']:+.2f}  "
                f"H={sig['hurst']:.3f}({sig['regime']})  "
                f"GARCH={sig['garch_vol']:.4f}  RSI={sig['rsi']:.1f}"
            )

        # Paper trade
        if self.paper and self.trader:
            self.trader.process(sig, price)

        # Periodic stats
        if self._candle_n % self.STAT_INTERVAL == 0 and self.paper:
            self._print_stats()

    def _print_stats(self):
        s = self.trader.stats()
        log.info(
            f"\n{'='*60}\n"
            f"  ACCOUNT STATS\n"
            f"  Trades    : {s['trades']}\n"
            f"  Win Rate  : {s['win_rate']:.1%}\n"
            f"  Total PnL : ${s['total_pnl']:+,.2f}\n"
            f"  Equity    : ${s['equity']:,.2f}\n"
            f"  Max DD    : {s['drawdown']:.2%}\n"
            f"{'='*60}"
        )

    def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        log.info(f"\n{'='*60}")
        log.info(f"  QUANT TRADER  |  {mode} MODE")
        log.info(f"  Symbol    : {self.symbol}")
        log.info(f"  Timeframe : {self.interval}")
        log.info(f"  Account   : ${self.trader.risk.equity:,.2f}")
        log.info(f"  Signals   : Kalman · OU · Hurst · Z-Score · GARCH")
        log.info(f"              Kelly · VaR/CVaR · RSI · MACD · VWAP")
        log.info(f"{'='*60}\n")

        self.feed.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutting down...")
            self.feed.stop()
            if self.paper:
                self._print_stats()


# ─────────────────────────────────────────────────────────────────────────────
#  7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Quant Auto-Trader")
    p.add_argument("--paper",   action="store_true",  help="Paper trading mode (safe)")
    p.add_argument("--symbol",  default="BTCUSDT",    help="Trading pair (default BTCUSDT)")
    p.add_argument("--tf",      default="5m",         help="Timeframe: 1m 3m 5m 15m 1h")
    p.add_argument("--account", type=float, default=5000.0, help="Starting account in USDT")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.paper:
        log.warning("LIVE mode not implemented — add your Binance API keys and order routing.")
        log.warning("Re-run with --paper for safe simulation.")
        sys.exit(1)

    bot = QuantTrader(
        symbol   = args.symbol,
        interval = args.tf,
        account  = args.account,
        paper    = args.paper,
    )
    bot.run()
