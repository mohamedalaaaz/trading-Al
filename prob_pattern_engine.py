"""
Probability Statistics Pattern Engine for Crypto Trading
=========================================================
BTC/USDT focused | Binance Futures compatible
Integrates with elite_ultra_v6.py pipeline

Patterns detected via pure statistical inference:
  - Candlestick sequences (Markov chain transition probabilities)
  - Volume-price correlation breaks (Pearson + Spearman)
  - Momentum divergence (Bayesian posterior updating)
  - Volatility regime shifts (CUSUM + chi-square)
  - Multi-bar conditional probability matrices
  - Z-score extreme events (fat-tail adjusted)
  - Binomial win-rate significance (Wilson interval)
  - Sequential probability ratio test (SPRT) for live edge detection

Usage:
    engine = ProbPatternEngine(lookback=200, alpha=0.05)
    result = engine.analyze(df)          # df = OHLCV DataFrame
    print(result.summary())
    
    # Live bar update
    engine.update(new_bar)
    signal = engine.get_signal()
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, binom
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PatternSignal:
    name: str
    direction: int          # +1 long, -1 short, 0 neutral
    probability: float      # P(win | pattern)
    edge: float             # probability - 0.5 (net edge over random)
    confidence: float       # statistical confidence [0,1]
    p_value: float          # null hypothesis p-value
    expected_value: float   # EV in % move
    sample_n: int           # historical occurrences
    z_score: float = 0.0
    notes: str = ""

@dataclass
class EngineResult:
    timestamp: Optional[pd.Timestamp]
    signals: list = field(default_factory=list)
    composite_direction: int = 0
    composite_probability: float = 0.5
    composite_confidence: float = 0.0
    regime: str = "unknown"
    volatility_z: float = 0.0
    dominant_pattern: Optional[PatternSignal] = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  PROB PATTERN ENGINE  |  {self.timestamp}",
            f"{'='*60}",
            f"  Regime        : {self.regime}",
            f"  Vol Z-Score   : {self.volatility_z:+.2f}",
            f"  Composite Dir : {'LONG' if self.composite_direction > 0 else 'SHORT' if self.composite_direction < 0 else 'FLAT'}",
            f"  Comp Prob     : {self.composite_probability:.1%}",
            f"  Confidence    : {self.composite_confidence:.1%}",
            f"{'─'*60}",
            f"  {'PATTERN':<28} {'DIR':<6} {'PROB':>6} {'EDGE':>6} {'N':>5} {'p-val':>7}",
            f"{'─'*60}",
        ]
        for s in sorted(self.signals, key=lambda x: abs(x.edge), reverse=True):
            d = "LONG" if s.direction > 0 else "SHORT" if s.direction < 0 else "FLAT"
            lines.append(
                f"  {s.name:<28} {d:<6} {s.probability:>5.1%} {s.edge:>+5.1%} {s.sample_n:>5} {s.p_value:>7.4f}"
            )
        if self.dominant_pattern:
            lines.append(f"{'─'*60}")
            lines.append(f"  DOMINANT: {self.dominant_pattern.name} | EV={self.dominant_pattern.expected_value:+.2f}%")
        lines.append("="*60)
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  CORE STATISTICAL PRIMITIVES
# ─────────────────────────────────────────────

class StatCore:
    """Pure probability math utilities."""

    @staticmethod
    def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple:
        """Wilson confidence interval for binary win rate."""
        if n == 0:
            return (0.5, 0.5, 0.5)
        p = wins / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        return (center, max(0, center - margin), min(1, center + margin))

    @staticmethod
    def bayesian_update(prior: float, likelihood_hit: float, likelihood_miss: float) -> float:
        """Bayes theorem: P(pattern | data) given prior."""
        p_data = likelihood_hit * prior + likelihood_miss * (1 - prior)
        if p_data == 0:
            return prior
        return (likelihood_hit * prior) / p_data

    @staticmethod
    def binom_significance(wins: int, n: int, p0: float = 0.5) -> float:
        """One-sided binomial test p-value against null p0."""
        if n == 0:
            return 1.0
        return binom.sf(wins - 1, n, p0)

    @staticmethod
    def rolling_zscore(series: np.ndarray, window: int = 20) -> np.ndarray:
        """Z-score of each value vs rolling window."""
        result = np.zeros(len(series))
        for i in range(window, len(series)):
            w = series[i-window:i]
            mu, sd = w.mean(), w.std()
            result[i] = (series[i] - mu) / sd if sd > 0 else 0.0
        return result

    @staticmethod
    def sprt(wins: int, n: int, p0: float = 0.5, p1: float = 0.55,
             alpha: float = 0.05, beta: float = 0.2) -> str:
        """Sequential Probability Ratio Test — stops early with evidence."""
        if n == 0:
            return "continue"
        A = np.log((1 - beta) / alpha)
        B = np.log(beta / (1 - alpha))
        p = wins / n
        lr = wins * np.log(p1/p0) + (n-wins) * np.log((1-p1)/(1-p0))
        if lr >= A:
            return "reject_null"    # Edge confirmed
        elif lr <= B:
            return "accept_null"    # No edge
        return "continue"


# ─────────────────────────────────────────────
#  PATTERN DETECTORS
# ─────────────────────────────────────────────

class CandlePatternProb:
    """
    Markov chain + conditional probability for candlestick sequences.
    Builds transition matrices from historical data.
    """

    def __init__(self, forward_bars: int = 3):
        self.forward_bars = forward_bars
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.pattern_outcomes = defaultdict(list)

    def _encode_candle(self, o, h, l, c) -> str:
        """Encode a candle as a categorical state."""
        body = c - o
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        range_ = h - l if h > l else 1e-9

        body_pct = abs(body) / range_
        upper_pct = upper_wick / range_
        lower_pct = lower_wick / range_

        direction = "B" if body > 0 else "R" if body < 0 else "D"  # Bull/Bear/Doji

        if body_pct < 0.1:
            size = "D"   # Doji
        elif body_pct > 0.6:
            size = "L"   # Large
        else:
            size = "M"   # Medium

        wick = "LW" if lower_pct > 0.3 else "UW" if upper_pct > 0.3 else "NW"

        return f"{direction}{size}_{wick}"

    def fit(self, df: pd.DataFrame):
        """Build transition and outcome matrices from OHLCV history."""
        self.pattern_outcomes.clear()
        self.transition_counts.clear()

        states = []
        for _, row in df.iterrows():
            states.append(self._encode_candle(row.open, row.high, row.low, row.close))

        for i in range(2, len(states) - self.forward_bars):
            seq = (states[i-2], states[i-1], states[i])
            key = "->".join(seq)

            future_ret = (df.close.iloc[i + self.forward_bars] - df.close.iloc[i]) / df.close.iloc[i]
            self.pattern_outcomes[key].append(future_ret)

            if i < len(states) - 1:
                self.transition_counts[states[i]][states[i+1]] += 1

    def get_pattern_signal(self, df: pd.DataFrame, min_n: int = 15) -> Optional[PatternSignal]:
        """Get probability signal for current 3-candle sequence."""
        if len(df) < 3:
            return None

        last = df.tail(3)
        states = [
            self._encode_candle(r.open, r.high, r.low, r.close)
            for _, r in last.iterrows()
        ]
        key = "->".join(states)
        outcomes = self.pattern_outcomes.get(key, [])

        if len(outcomes) < min_n:
            return None

        outcomes = np.array(outcomes)
        wins = int(np.sum(outcomes > 0))
        n = len(outcomes)
        p_win, lo, hi = StatCore.wilson_interval(wins, n)
        p_val = StatCore.binom_significance(wins, n)
        direction = +1 if p_win > 0.5 else -1 if p_win < 0.5 else 0
        edge = p_win - 0.5
        ev = float(outcomes.mean() * 100)

        sprt = StatCore.sprt(wins, n)
        confidence = 1.0 if sprt == "reject_null" else 0.5 if sprt == "continue" else 0.0

        return PatternSignal(
            name=f"Candle:{key[:30]}",
            direction=direction,
            probability=p_win,
            edge=edge,
            confidence=confidence,
            p_value=p_val,
            expected_value=ev,
            sample_n=n,
            notes=f"Wilson CI=[{lo:.2%},{hi:.2%}]"
        )

    def get_transition_prob(self, df: pd.DataFrame) -> Optional[PatternSignal]:
        """Markov next-state probability from current candle."""
        if len(df) < 1:
            return None
        last = df.iloc[-1]
        state = self._encode_candle(last.open, last.high, last.low, last.close)
        transitions = self.transition_counts[state]
        if not transitions:
            return None

        total = sum(transitions.values())
        bull_states = sum(v for k, v in transitions.items() if k.startswith("B"))
        bear_states = sum(v for k, v in transitions.items() if k.startswith("R"))

        p_bull = bull_states / total
        wins = bull_states
        p_win, _, _ = StatCore.wilson_interval(wins, total)
        p_val = StatCore.binom_significance(wins, total)
        direction = +1 if p_bull > 0.52 else -1 if p_bull < 0.48 else 0

        return PatternSignal(
            name=f"Markov:{state[:20]}",
            direction=direction,
            probability=p_win,
            edge=p_win - 0.5,
            confidence=min(1.0, total / 50),
            p_value=p_val,
            expected_value=0.0,
            sample_n=total,
        )


class VolumePatternProb:
    """
    Volume-price divergence detection via correlation statistics.
    High-probability reversal when volume and price decorrelate.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def analyze(self, df: pd.DataFrame, alpha: float = 0.05) -> Optional[PatternSignal]:
        if len(df) < self.window + 5:
            return None

        prices = df.close.values[-self.window:]
        volumes = df.volume.values[-self.window:]
        price_chg = np.diff(prices)
        vol_chg = np.diff(volumes)

        # Spearman rank correlation (non-parametric, robust to outliers)
        rho, p_val = stats.spearmanr(price_chg, vol_chg)

        # Historical baseline: full lookback correlation
        all_pc = np.diff(df.close.values)
        all_vc = np.diff(df.volume.values)
        n = min(len(all_pc), len(all_vc))
        hist_rho, _ = stats.spearmanr(all_pc[:n], all_vc[:n])

        # Z-score of current correlation vs historical distribution
        # Using Fisher Z-transform for correlation comparison
        def fisher_z(r):
            r = np.clip(r, -0.9999, 0.9999)
            return 0.5 * np.log((1 + r) / (1 - r))

        z_curr = fisher_z(rho)
        z_hist = fisher_z(hist_rho)
        z_diff = abs(z_curr - z_hist)

        is_divergence = z_diff > 1.0  # >1 std dev divergence

        if not is_divergence:
            return None

        # If price up but volume falling → bearish divergence
        # If price down but volume falling → bullish (selling exhaustion)
        recent_price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        recent_vol_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]

        if recent_price_trend > 0 and recent_vol_trend < 0:
            direction = -1  # Bearish divergence
            prob = 0.5 + min(0.35, z_diff * 0.1)
        elif recent_price_trend < 0 and recent_vol_trend < 0:
            direction = +1  # Bullish (exhaustion)
            prob = 0.5 + min(0.25, z_diff * 0.07)
        else:
            return None

        return PatternSignal(
            name="VolumePrice:Divergence",
            direction=direction,
            probability=prob,
            edge=prob - 0.5,
            confidence=min(1.0, z_diff / 3),
            p_value=p_val,
            expected_value=direction * z_diff * 0.15,
            sample_n=self.window,
            z_score=z_diff,
            notes=f"rho={rho:.2f} hist={hist_rho:.2f} z={z_diff:.2f}"
        )


class MomentumDivergenceProb:
    """
    Bayesian momentum divergence: price makes new extreme but momentum doesn't.
    Posterior probability updated from historical hit rate.
    """

    def __init__(self, rsi_period: int = 14, lookback: int = 50):
        self.rsi_period = rsi_period
        self.lookback = lookback
        # Prior: momentum divergence has ~55% reversal rate from literature
        self.prior_bull_div = 0.55
        self.prior_bear_div = 0.55

    def _rsi(self, prices: np.ndarray) -> np.ndarray:
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(self.rsi_period)/self.rsi_period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(self.rsi_period)/self.rsi_period, mode='valid')
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        return 100 - (100 / (1 + rs))

    def _fit_prior(self, df: pd.DataFrame) -> tuple:
        """Estimate likelihood from historical divergence outcomes."""
        if len(df) < self.lookback + 20:
            return self.prior_bull_div, self.prior_bear_div

        prices = df.close.values
        rsi = self._rsi(prices)
        rsi_prices = prices[self.rsi_period:]

        bull_wins, bull_n = 0, 0
        bear_wins, bear_n = 0, 0
        fwd = 5

        for i in range(20, len(rsi) - fwd):
            # Bullish divergence: price lower low, RSI higher low
            if (rsi_prices[i] < rsi_prices[i-10] and rsi[i] > rsi[i-10]):
                bull_n += 1
                if rsi_prices[i+fwd] > rsi_prices[i]:
                    bull_wins += 1
            # Bearish divergence: price higher high, RSI lower high
            if (rsi_prices[i] > rsi_prices[i-10] and rsi[i] < rsi[i-10]):
                bear_n += 1
                if rsi_prices[i+fwd] < rsi_prices[i]:
                    bear_wins += 1

        prior_bull = (bull_wins / bull_n) if bull_n > 10 else self.prior_bull_div
        prior_bear = (bear_wins / bear_n) if bear_n > 10 else self.prior_bear_div
        return prior_bull, prior_bear

    def analyze(self, df: pd.DataFrame, alpha: float = 0.05) -> Optional[PatternSignal]:
        if len(df) < self.rsi_period + 20:
            return None

        prices = df.close.values
        rsi = self._rsi(prices)
        rsi_prices = prices[self.rsi_period:]

        if len(rsi) < 15:
            return None

        prior_bull, prior_bear = self._fit_prior(df)
        lookback = 10

        price_now = rsi_prices[-1]
        price_prev = rsi_prices[-1-lookback]
        rsi_now = rsi[-1]
        rsi_prev = rsi[-1-lookback]

        # Bayesian update based on RSI extreme conditions
        # Likelihood of divergence being real given RSI in oversold/overbought
        direction = 0
        probability = 0.5
        pattern_name = ""

        if price_now < price_prev and rsi_now > rsi_prev and rsi_now < 40:
            # Bullish divergence in oversold territory
            likelihood_if_real = 0.75   # RSI < 40 increases reliability
            likelihood_if_noise = 0.30
            posterior = StatCore.bayesian_update(prior_bull, likelihood_if_real, likelihood_if_noise)
            direction = +1
            probability = posterior
            pattern_name = "MomDiv:Bullish(RSI)"

        elif price_now > price_prev and rsi_now < rsi_prev and rsi_now > 60:
            # Bearish divergence in overbought territory
            likelihood_if_real = 0.72
            likelihood_if_noise = 0.32
            posterior = StatCore.bayesian_update(prior_bear, likelihood_if_real, likelihood_if_noise)
            direction = -1
            probability = posterior
            pattern_name = "MomDiv:Bearish(RSI)"
        else:
            return None

        edge = probability - 0.5
        if abs(edge) < 0.03:
            return None

        # Statistical significance via binomial test against historical base
        wins_est = int(probability * 30)
        p_val = StatCore.binom_significance(wins_est, 30)

        return PatternSignal(
            name=pattern_name,
            direction=direction,
            probability=probability,
            edge=edge,
            confidence=min(1.0, abs(edge) * 4),
            p_value=p_val,
            expected_value=direction * abs(edge) * 100 * 0.5,
            sample_n=30,
            notes=f"RSI={rsi_now:.1f} price_chg={((price_now/price_prev)-1)*100:.2f}%"
        )


class VolatilityRegimeProb:
    """
    CUSUM-based volatility regime change detection.
    Chi-square test for distribution shift.
    """

    def __init__(self, window: int = 20, cusum_threshold: float = 4.0):
        self.window = window
        self.cusum_threshold = cusum_threshold

    def analyze(self, df: pd.DataFrame) -> tuple:
        """Returns (regime_label, z_score, PatternSignal or None)."""
        if len(df) < self.window * 3:
            return ("unknown", 0.0, None)

        returns = df.close.pct_change().dropna().values
        vol = pd.Series(returns).rolling(self.window).std().dropna().values

        if len(vol) < self.window:
            return ("unknown", 0.0, None)

        # Current volatility vs long-term baseline
        current_vol = vol[-self.window:].mean()
        baseline_vol = vol[:-self.window].mean() if len(vol) > self.window * 2 else vol.mean()
        baseline_std = vol[:-self.window].std() if len(vol) > self.window * 2 else vol.std()

        vol_z = (current_vol - baseline_vol) / baseline_std if baseline_std > 0 else 0.0

        # Regime classification
        if vol_z > 2.0:
            regime = "high_vol"
        elif vol_z < -1.0:
            regime = "low_vol"
        else:
            regime = "normal"

        # CUSUM test for structural break
        cusum_pos = 0.0
        cusum_neg = 0.0
        mu = baseline_vol
        sd = baseline_std if baseline_std > 0 else 1e-9
        cusum_signal = None

        for v in vol[-self.window:]:
            cusum_pos = max(0, cusum_pos + (v - mu) / sd - 0.5)
            cusum_neg = max(0, cusum_neg - (v - mu) / sd - 0.5)

        if cusum_pos > self.cusum_threshold:
            # Volatility expansion breakout — tend to trend strongly
            chi2_stat, p_val, _, _ = chi2_contingency(
                [[int(cusum_pos * 10), 100], [int(cusum_neg * 10), 100]]
            )
            cusum_signal = PatternSignal(
                name="Regime:VolExpansion(CUSUM)",
                direction=0,   # Regime signal, not directional
                probability=0.5 + min(0.3, cusum_pos / 20),
                edge=cusum_pos / 20,
                confidence=min(1.0, cusum_pos / self.cusum_threshold),
                p_value=p_val,
                expected_value=0.0,
                sample_n=self.window,
                z_score=vol_z,
                notes=f"CUSUM+={cusum_pos:.1f} vol_z={vol_z:.2f}"
            )
        elif cusum_neg > self.cusum_threshold:
            chi2_stat, p_val, _, _ = chi2_contingency(
                [[int(cusum_neg * 10), 100], [int(cusum_pos * 10), 100]]
            )
            cusum_signal = PatternSignal(
                name="Regime:VolContraction(CUSUM)",
                direction=0,
                probability=0.5 + min(0.2, cusum_neg / 20),
                edge=cusum_neg / 20,
                confidence=min(1.0, cusum_neg / self.cusum_threshold),
                p_value=p_val,
                expected_value=0.0,
                sample_n=self.window,
                z_score=vol_z,
                notes=f"CUSUM-={cusum_neg:.1f} vol_z={vol_z:.2f}"
            )

        return (regime, vol_z, cusum_signal)


class ExtremeMoveProb:
    """
    Fat-tail adjusted z-score for detecting extreme moves.
    Uses Student-t distribution (not Gaussian) for crypto tails.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame, alpha: float = 0.01) -> Optional[PatternSignal]:
        if len(df) < self.lookback + 5:
            return None

        returns = df.close.pct_change().dropna().values
        hist = returns[-self.lookback:-1]
        current = returns[-1]

        # Fit Student-t for fat tails (crypto has df ~3-5)
        try:
            df_t, loc, scale = stats.t.fit(hist)
            p_val = 2 * stats.t.sf(abs((current - loc) / scale), df=df_t)
        except Exception:
            return None

        if p_val > alpha:
            return None

        # Extreme move: test for mean-reversion probability
        # Empirical: extreme moves in crypto revert ~60% within 3 bars
        z = (current - loc) / scale
        p_revert = 0.5 + min(0.35, (abs(z) - 2) * 0.05)
        direction = -1 if current > 0 else +1  # Fade the extreme

        return PatternSignal(
            name=f"ExtMove:t-tail(z={z:.1f})",
            direction=direction,
            probability=p_revert,
            edge=p_revert - 0.5,
            confidence=1 - p_val,
            p_value=p_val,
            expected_value=direction * abs(current) * 100 * 0.4,
            sample_n=self.lookback,
            z_score=z,
            notes=f"t-df={df_t:.1f} move={current*100:.2f}%"
        )


class ConditionalProbMatrix:
    """
    Multi-condition probability matrix:
    P(direction | RSI_zone, Vol_regime, Candle_type, Session)
    Built entirely from historical data.
    """

    def __init__(self):
        self.matrix = defaultdict(lambda: {"wins": 0, "n": 0, "returns": []})

    def _encode_state(self, row_series: pd.Series, rsi_val: float,
                      vol_z: float, session: str) -> str:
        body = row_series.close - row_series.open
        candle = "bull" if body > 0 else "bear"

        rsi_zone = "OS" if rsi_val < 35 else "OB" if rsi_val > 65 else "MID"
        vol_state = "HV" if vol_z > 1.5 else "LV" if vol_z < -1.0 else "NV"

        return f"{candle}|{rsi_zone}|{vol_state}|{session}"

    def fit(self, df: pd.DataFrame, rsi_vals: np.ndarray, vol_zs: np.ndarray,
            forward_bars: int = 3):
        self.matrix.clear()
        hours = pd.DatetimeIndex(df.index).hour if hasattr(df.index, 'hour') else [0] * len(df)

        for i in range(len(df) - forward_bars):
            hour = hours[i] if i < len(hours) else 0
            session = "Asia" if 0 <= hour < 8 else "London" if 8 <= hour < 16 else "NY"

            rsi = rsi_vals[i] if i < len(rsi_vals) else 50
            vz = vol_zs[i] if i < len(vol_zs) else 0

            key = self._encode_state(df.iloc[i], rsi, vz, session)
            fwd_ret = (df.close.iloc[i + forward_bars] - df.close.iloc[i]) / df.close.iloc[i]

            self.matrix[key]["n"] += 1
            self.matrix[key]["returns"].append(fwd_ret)
            if fwd_ret > 0:
                self.matrix[key]["wins"] += 1

    def get_signal(self, df: pd.DataFrame, rsi_val: float, vol_z: float,
                   min_n: int = 20) -> Optional[PatternSignal]:
        if len(df) < 1:
            return None

        hours = pd.DatetimeIndex(df.index).hour if hasattr(df.index, 'hour') else [0]
        hour = hours[-1] if len(hours) > 0 else 0
        session = "Asia" if 0 <= hour < 8 else "London" if 8 <= hour < 16 else "NY"

        key = self._encode_state(df.iloc[-1], rsi_val, vol_z, session)
        cell = self.matrix.get(key)

        if cell is None or cell["n"] < min_n:
            return None

        wins = cell["wins"]
        n = cell["n"]
        returns = np.array(cell["returns"])
        p_win, lo, hi = StatCore.wilson_interval(wins, n)
        p_val = StatCore.binom_significance(wins, n)
        direction = +1 if p_win > 0.52 else -1 if p_win < 0.48 else 0
        sprt = StatCore.sprt(wins, n)
        confidence = 1.0 if sprt == "reject_null" else 0.5 if sprt == "continue" else 0.0

        return PatternSignal(
            name=f"CondProb:{key}",
            direction=direction,
            probability=p_win,
            edge=p_win - 0.5,
            confidence=confidence,
            p_value=p_val,
            expected_value=float(returns.mean() * 100),
            sample_n=n,
            notes=f"session={session} sprt={sprt}"
        )


# ─────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────

class ProbPatternEngine:
    """
    Unified probability statistics engine.
    Aggregates all detectors into a single composite signal.
    """

    def __init__(self,
                 lookback: int = 200,
                 alpha: float = 0.05,
                 min_edge: float = 0.03,
                 min_confidence: float = 0.40):
        self.lookback = lookback
        self.alpha = alpha
        self.min_edge = min_edge
        self.min_confidence = min_confidence

        # Detectors
        self.candle_prob = CandlePatternProb(forward_bars=3)
        self.volume_prob = VolumePatternProb(window=20)
        self.momentum_prob = MomentumDivergenceProb()
        self.volatility_prob = VolatilityRegimeProb()
        self.extreme_prob = ExtremeMoveProb(lookback=100)
        self.cond_matrix = ConditionalProbMatrix()

        self._fitted = False
        self._last_result: Optional[EngineResult] = None

    def _rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0.0)
        loss = np.where(deltas < 0, -deltas, 0.0)
        ag = np.convolve(gain, np.ones(period)/period, mode='valid')
        al = np.convolve(loss, np.ones(period)/period, mode='valid')
        rs = np.where(al != 0, ag/al, 100)
        rsi = 100 - 100/(1+rs)
        return np.concatenate([np.full(period, 50.0), rsi])

    def fit(self, df: pd.DataFrame):
        """
        Train all statistical models on historical OHLCV data.
        Call once before live updates.
        df columns: open, high, low, close, volume
        df index: DatetimeIndex
        """
        assert len(df) >= 50, "Need at least 50 bars to fit."
        df = df.copy()
        df.columns = df.columns.str.lower()

        print(f"[ProbEngine] Fitting on {len(df)} bars...")

        # Fit candle Markov model
        self.candle_prob.fit(df)

        # Build RSI and vol arrays for conditional matrix
        rsi_arr = self._rsi(df.close.values)
        returns = df.close.pct_change().fillna(0).values
        vol = pd.Series(np.abs(returns)).rolling(20).std().fillna(0).values
        vol_baseline = np.nanmean(vol)
        vol_std = np.nanstd(vol)
        vol_z = (vol - vol_baseline) / vol_std if vol_std > 0 else vol * 0

        # Fit conditional probability matrix
        self.cond_matrix.fit(df, rsi_arr, vol_z)

        self._df = df
        self._fitted = True
        print(f"[ProbEngine] Fit complete. {len(self.cond_matrix.matrix)} conditional states learned.")
        print(f"[ProbEngine] {len(self.candle_prob.pattern_outcomes)} candle patterns indexed.")

    def analyze(self, df: Optional[pd.DataFrame] = None) -> EngineResult:
        """
        Run all detectors. Returns EngineResult with all signals.
        If df is None, uses stored data from fit().
        """
        if df is None:
            df = self._df
        df = df.copy()
        df.columns = df.columns.str.lower()

        ts = df.index[-1] if hasattr(df.index, '__getitem__') else None
        result = EngineResult(timestamp=ts)

        # --- Volatility Regime ---
        regime, vol_z, regime_signal = self.volatility_prob.analyze(df)
        result.regime = regime
        result.volatility_z = vol_z
        if regime_signal:
            result.signals.append(regime_signal)

        # --- RSI for downstream use ---
        rsi_arr = self._rsi(df.close.values)
        current_rsi = rsi_arr[-1]

        # --- Candle Pattern (Markov sequence) ---
        candle_sig = self.candle_prob.get_pattern_signal(df)
        if candle_sig and candle_sig.p_value < self.alpha:
            result.signals.append(candle_sig)

        markov_sig = self.candle_prob.get_transition_prob(df)
        if markov_sig and markov_sig.p_value < self.alpha:
            result.signals.append(markov_sig)

        # --- Volume Divergence ---
        vol_sig = self.volume_prob.analyze(df, self.alpha)
        if vol_sig:
            result.signals.append(vol_sig)

        # --- Momentum Divergence (Bayesian) ---
        mom_sig = self.momentum_prob.analyze(df, self.alpha)
        if mom_sig:
            result.signals.append(mom_sig)

        # --- Extreme Move (Fat Tail) ---
        extreme_sig = self.extreme_prob.analyze(df, alpha=0.01)
        if extreme_sig:
            result.signals.append(extreme_sig)

        # --- Conditional Probability Matrix ---
        cond_sig = self.cond_matrix.get_signal(df, current_rsi, vol_z)
        if cond_sig and cond_sig.p_value < self.alpha:
            result.signals.append(cond_sig)

        # --- Filter by minimum edge and confidence ---
        valid = [
            s for s in result.signals
            if abs(s.edge) >= self.min_edge and s.confidence >= self.min_confidence
        ]
        result.signals = valid

        # --- Composite Signal (confidence-weighted probability) ---
        if valid:
            weights = np.array([s.confidence for s in valid])
            probs = np.array([s.probability for s in valid])
            dirs = np.array([s.direction for s in valid])

            # Weighted average probability
            w_sum = weights.sum()
            comp_prob = float(np.dot(weights, probs) / w_sum) if w_sum > 0 else 0.5
            comp_dir_score = float(np.dot(weights, dirs) / w_sum) if w_sum > 0 else 0

            result.composite_probability = comp_prob
            result.composite_direction = +1 if comp_dir_score > 0.1 else -1 if comp_dir_score < -0.1 else 0
            result.composite_confidence = float(np.mean(weights))

            # Dominant: highest absolute edge with good confidence
            result.dominant_pattern = max(valid, key=lambda s: abs(s.edge) * s.confidence)

        self._last_result = result
        return result

    def update(self, new_bar: pd.Series):
        """Append a new bar and re-analyze (for live trading loop)."""
        if self._fitted and hasattr(self, '_df'):
            self._df = pd.concat([self._df, new_bar.to_frame().T])
            if len(self._df) > self.lookback * 3:
                self._df = self._df.iloc[-self.lookback * 3:]

    def get_signal(self) -> EngineResult:
        """Get last analysis result (call after update())."""
        return self.analyze() if self._fitted else EngineResult(timestamp=None)

    def report_pattern_stats(self, top_n: int = 10):
        """Print top patterns by historical edge from fitted data."""
        print(f"\n{'='*70}")
        print(f"  TOP {top_n} CANDLESTICK PATTERNS BY EDGE (min 15 occurrences)")
        print(f"{'='*70}")
        print(f"  {'PATTERN':<40} {'WIN%':>6} {'EDGE':>6} {'N':>5} {'EV%':>7}")
        print(f"{'─'*70}")

        ranked = []
        for key, outcomes in self.candle_prob.pattern_outcomes.items():
            if len(outcomes) < 15:
                continue
            outcomes_arr = np.array(outcomes)
            wins = int(np.sum(outcomes_arr > 0))
            n = len(outcomes_arr)
            p_win, _, _ = StatCore.wilson_interval(wins, n)
            edge = p_win - 0.5
            ev = float(outcomes_arr.mean() * 100)
            ranked.append((key, p_win, edge, n, ev))

        ranked.sort(key=lambda x: abs(x[2]), reverse=True)

        for key, p_win, edge, n, ev in ranked[:top_n]:
            k = key[:38]
            print(f"  {k:<40} {p_win:>5.1%} {edge:>+5.1%} {n:>5} {ev:>+6.2f}%")
        print("="*70)

    def report_conditional_stats(self, top_n: int = 10):
        """Print top conditional probability states."""
        print(f"\n{'='*70}")
        print(f"  TOP {top_n} CONDITIONAL STATES (min 20 occurrences)")
        print(f"  Format: candle|RSI_zone|Vol|Session")
        print(f"{'='*70}")
        print(f"  {'STATE':<35} {'WIN%':>6} {'EDGE':>6} {'N':>5} {'EV%':>7} {'SPRT'}")
        print(f"{'─'*70}")

        ranked = []
        for key, cell in self.cond_matrix.matrix.items():
            if cell["n"] < 20:
                continue
            wins = cell["wins"]
            n = cell["n"]
            returns = np.array(cell["returns"])
            p_win, _, _ = StatCore.wilson_interval(wins, n)
            edge = p_win - 0.5
            ev = float(returns.mean() * 100)
            sprt = StatCore.sprt(wins, n)
            ranked.append((key, p_win, edge, n, ev, sprt))

        ranked.sort(key=lambda x: abs(x[2]), reverse=True)

        for key, p_win, edge, n, ev, sprt in ranked[:top_n]:
            k = key[:33]
            sprt_short = "EDGE" if sprt == "reject_null" else "CONT" if sprt == "continue" else "NONE"
            print(f"  {k:<35} {p_win:>5.1%} {edge:>+5.1%} {n:>5} {ev:>+6.2f}% {sprt_short}")
        print("="*70)


# ─────────────────────────────────────────────
#  QUICK DEMO / STANDALONE TEST
# ─────────────────────────────────────────────

def _generate_synthetic_btc(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic BTC OHLCV for testing without live data."""
    rng = np.random.default_rng(seed)
    price = 65000.0
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    rows = []
    for i in range(n):
        vol_mult = 1.5 if i % 200 < 30 else 1.0  # Occasional vol spikes
        ret = rng.normal(0.0001, 0.004 * vol_mult)
        price *= (1 + ret)
        o = price
        c = price * (1 + rng.normal(0, 0.001))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0015)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0015)))
        vol = abs(rng.normal(500, 200)) * vol_mult
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": vol})
    return pd.DataFrame(rows, index=dates)


if __name__ == "__main__":
    print("Generating synthetic BTC/USDT 5m data...")
    df = _generate_synthetic_btc(n=1500)

    engine = ProbPatternEngine(
        lookback=300,
        alpha=0.05,
        min_edge=0.02,
        min_confidence=0.35
    )

    # Fit on first 1000 bars
    engine.fit(df.iloc[:1000])

    # Analyze last 300 bars
    result = engine.analyze(df.iloc[-300:])
    print(result.summary())

    # Print pattern stats
    engine.report_pattern_stats(top_n=10)
    engine.report_conditional_stats(top_n=10)

    # Simulate live bar update
    print("\n[LIVE] Simulating single bar update...")
    new_bar = df.iloc[-1]
    engine.update(new_bar)
    live_result = engine.get_signal()
    print(f"Live signal: {live_result.composite_direction:+d} | "
          f"Prob: {live_result.composite_probability:.1%} | "
          f"Regime: {live_result.regime}")
