import java.util.*;
import java.util.stream.*;
import java.text.*;
import java.time.*;
import java.time.format.*;

/**
 * ============================================================
 *   QUANTITATIVE AUTOMATED TRADING SYSTEM  — Java Edition
 * ============================================================
 *  Strategies implemented:
 *   1. Exponential Moving Average (EMA) Crossover
 *   2. Relative Strength Index (RSI) Mean-Reversion
 *   3. Bollinger Bands Breakout
 *   4. MACD (Moving Average Convergence Divergence)
 *   5. Momentum / Rate-of-Change (ROC)
 *
 *  Risk Management:
 *   • Kelly Criterion position sizing
 *   • ATR-based stop-loss / take-profit
 *   • Maximum drawdown circuit-breaker
 *   • Portfolio heat limit
 *
 *  Engine:
 *   • Event-driven back-test loop (tick-by-tick)
 *   • Real-time P&L, Sharpe ratio, win-rate reporting
 * ============================================================
 */
public class QuantTrader {

    // ─────────────────────────────────────────────────────────
    //  CONSTANTS & CONFIGURATION
    // ─────────────────────────────────────────────────────────
    static final double INITIAL_CAPITAL      = 100_000.0;
    static final double MAX_POSITION_PCT     = 0.10;   // 10 % of portfolio per trade
    static final double MAX_PORTFOLIO_HEAT   = 0.30;   // 30 % total risk at once
    static final double MAX_DRAWDOWN_LIMIT   = 0.20;   // 20 % → halt trading
    static final int WARMUP = 50;
    static final double COMMISSION_PCT       = 0.001;  // 0.10 % per side
    static final double ATR_STOP_MULTIPLIER  = 1.5;
    static final double ATR_TARGET_MULTIPLIER= 2.5;

    // ─────────────────────────────────────────────────────────
    //  DATA STRUCTURES
    // ─────────────────────────────────────────────────────────

    /** A single OHLCV bar */
    record Bar(String date, double open, double high, double low,
               double close, long volume) {}

    /** An open position */
    static class Position {
        String  symbol;
        String  strategy;
        int     direction;    //  +1 long / -1 short
        double  entryPrice;
        double  stopLoss;
        double  takeProfit;
        int     qty;
        String  entryDate;
        double  riskAmount;

        Position(String symbol, String strategy, int direction,
                 double entry, double stop, double target,
                 int qty, String date, double risk) {
            this.symbol     = symbol;  this.strategy  = strategy;
            this.direction  = direction; this.entryPrice = entry;
            this.stopLoss   = stop;    this.takeProfit = target;
            this.qty        = qty;     this.entryDate  = date;
            this.riskAmount = risk;
        }
    }

    /** Closed trade record */
    record Trade(String symbol, String strategy, String entryDate,
                 String exitDate, int direction, double entryPrice,
                 double exitPrice, int qty, double pnl, String exitReason) {}

    // ─────────────────────────────────────────────────────────
    //  MATH LIBRARY  (all pure-Java, no external deps)
    // ─────────────────────────────────────────────────────────
    static class MathLib {

        /** Simple Moving Average */
        static double sma(double[] data, int period) {
            if (data.length < period) return Double.NaN;
            double sum = 0;
            for (int i = data.length - period; i < data.length; i++) sum += data[i];
            return sum / period;
        }

        /** Exponential Moving Average — full series */
        static double[] emaFull(double[] data, int period) {
            double[] ema = new double[data.length];
            double k = 2.0 / (period + 1);
            ema[0] = data[0];
            for (int i = 1; i < data.length; i++)
                ema[i] = data[i] * k + ema[i - 1] * (1 - k);
            return ema;
        }

        /** Latest EMA value */
        static double ema(double[] data, int period) {
            double[] full = emaFull(data, period);
            return full[full.length - 1];
        }

        /**
         * RSI — Wilder smoothing
         */
        static double rsi(double[] closes, int period) {
            if (closes.length <= period) return 50.0;
            double avgGain = 0, avgLoss = 0;
            for (int i = 1; i <= period; i++) {
                double chg = closes[i] - closes[i - 1];
                if (chg > 0) avgGain += chg; else avgLoss -= chg;
            }
            avgGain /= period; avgLoss /= period;
            for (int i = period + 1; i < closes.length; i++) {
                double chg = closes[i] - closes[i - 1];
                double gain = Math.max(chg, 0), loss = Math.max(-chg, 0);
                avgGain = (avgGain * (period - 1) + gain) / period;
                avgLoss = (avgLoss * (period - 1) + loss) / period;
            }
            if (avgLoss == 0) return 100;
            double rs = avgGain / avgLoss;
            return 100 - 100 / (1 + rs);
        }

        /** Bollinger Bands: [middle, upper, lower] */
        static double[] bollingerBands(double[] data, int period, double numStd) {
            double mid = sma(data, period);
            double[] slice = Arrays.copyOfRange(data, data.length - period, data.length);
            double variance = 0;
            for (double v : slice) variance += (v - mid) * (v - mid);
            double std = Math.sqrt(variance / period);
            return new double[]{mid, mid + numStd * std, mid - numStd * std};
        }

        /** MACD: [macdLine, signalLine, histogram] */
        static double[] macd(double[] data, int fast, int slow, int signal) {
            double[] fastEma = emaFull(data, fast);
            double[] slowEma = emaFull(data, slow);
            double[] macdLine = new double[data.length];
            for (int i = 0; i < data.length; i++)
                macdLine[i] = fastEma[i] - slowEma[i];
            double[] sig = emaFull(macdLine, signal);
            double macdVal = macdLine[macdLine.length - 1];
            double sigVal  = sig[sig.length - 1];
            return new double[]{macdVal, sigVal, macdVal - sigVal};
        }

        /** Average True Range */
        static double atr(double[] highs, double[] lows, double[] closes, int period) {
            int n = closes.length;
            if (n < 2) return 0;
            double[] tr = new double[n - 1];
            for (int i = 1; i < n; i++) {
                double hl  = highs[i]  - lows[i];
                double hcp = Math.abs(highs[i]  - closes[i - 1]);
                double lcp = Math.abs(lows[i]   - closes[i - 1]);
                tr[i - 1] = Math.max(hl, Math.max(hcp, lcp));
            }
            return sma(tr, Math.min(period, tr.length));
        }

        /** Rate of Change (momentum %) */
        static double roc(double[] data, int period) {
            int n = data.length;
            if (n <= period) return 0;
            return (data[n - 1] - data[n - 1 - period]) / data[n - 1 - period] * 100;
        }

        /** Kelly Criterion fraction f* = (bp - q) / b */
        static double kellyCriterion(double winRate, double avgWin, double avgLoss) {
            if (avgLoss == 0) return 0;
            double b = avgWin / avgLoss;      // win/loss ratio
            double p = winRate, q = 1 - p;
            return Math.max(0, (b * p - q) / b);
        }

        /** Sharpe Ratio (daily returns → annualised) */
        static double sharpeRatio(double[] returns) {
            if (returns.length < 2) return 0;
            double mean = Arrays.stream(returns).average().orElse(0);
            double var  = Arrays.stream(returns).map(r -> (r - mean) * (r - mean))
                                .average().orElse(0);
            double std  = Math.sqrt(var);
            if (std == 0) return 0;
            return (mean / std) * Math.sqrt(252);
        }

        /** Max drawdown of equity curve */
        static double maxDrawdown(double[] equity) {
            double peak = equity[0], maxDD = 0;
            for (double e : equity) {
                if (e > peak) peak = e;
                double dd = (peak - e) / peak;
                if (dd > maxDD) maxDD = dd;
            }
            return maxDD;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  STRATEGY SIGNALS  (return +1 buy / -1 sell / 0 neutral)
    // ─────────────────────────────────────────────────────────
    static class Strategies {

        /** EMA Crossover: fast EMA crosses above/below slow EMA */
        static int emaCrossover(double[] closes, int fast, int slow) {
            if (closes.length < slow + 1) return 0;
            double[] prev = Arrays.copyOf(closes, closes.length - 1);
            double fastNow  = MathLib.ema(closes, fast),
                   slowNow  = MathLib.ema(closes, slow),
                   fastPrev = MathLib.ema(prev,   fast),
                   slowPrev = MathLib.ema(prev,   slow);
            if (fastPrev <= slowPrev && fastNow > slowNow) return  1;  // golden cross
            if (fastPrev >= slowPrev && fastNow < slowNow) return -1;  // death  cross
            return 0;
        }

        /** RSI Mean-Reversion: oversold → buy, overbought → sell */
        static int rsiMeanReversion(double[] closes, int period,
                                    double oversold, double overbought) {
            double rsiVal = MathLib.rsi(closes, period);
            if (rsiVal <= oversold)  return  1;
            if (rsiVal >= overbought)return -1;
            return 0;
        }

        /** Bollinger Band breakout */
        static int bollingerBreakout(double[] closes, int period, double numStd) {
            double price = closes[closes.length - 1];
            double[] bb  = MathLib.bollingerBands(closes, period, numStd);
            if (price > bb[1]) return  1;   // above upper → momentum long
            if (price < bb[2]) return -1;   // below lower → momentum short
            return 0;
        }

        /** MACD histogram sign change */
        static int macdSignal(double[] closes, int fast, int slow, int signal) {
            if (closes.length < slow + signal + 1) return 0;
            double[] prev    = Arrays.copyOf(closes, closes.length - 1);
            double histNow   = MathLib.macd(closes, fast, slow, signal)[2];
            double histPrev  = MathLib.macd(prev,   fast, slow, signal)[2];
            if (histPrev < 0 && histNow >= 0) return  1;
            if (histPrev > 0 && histNow <= 0) return -1;
            return 0;
        }

        /** Momentum / ROC threshold */
        static int momentumROC(double[] closes, int period, double threshold) {
            double roc = MathLib.roc(closes, period);
            if (roc >  threshold) return  1;
            if (roc < -threshold) return -1;
            return 0;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  PORTFOLIO & RISK MANAGER
    // ─────────────────────────────────────────────────────────
    static class PortfolioManager {
        double              cash;
        double              peakEquity;
        List<Position>      openPositions = new ArrayList<>();
        List<Trade>         closedTrades  = new ArrayList<>();
        List<Double>        equityCurve   = new ArrayList<>();
        boolean             haltTrading   = false;

        // running statistics for Kelly
        int    wins = 0, losses = 0;
        double totalWin = 0, totalLoss = 0;

        PortfolioManager(double capital) {
            this.cash = capital;
            this.peakEquity = capital;
            equityCurve.add(capital);
        }

        double equity(Map<String, Double> prices) {
            double posVal = openPositions.stream()
                .mapToDouble(p -> p.qty * prices.getOrDefault(p.symbol, p.entryPrice))
                .sum();
            return cash + posVal;
        }

        double totalRisk() {
            return openPositions.stream().mapToDouble(p -> p.riskAmount).sum();
        }

        /**
         * Open a new position if risk rules allow.
         */
        boolean openPosition(String symbol, String strategy, int direction,
                             double price, double atrVal,
                             double portfolioEquity, String date) {
            if (haltTrading) return false;

            double stop   = price - direction * ATR_STOP_MULTIPLIER   * atrVal;
            double target = price + direction * ATR_TARGET_MULTIPLIER  * atrVal;
            double riskPerShare = Math.abs(price - stop);
            if (riskPerShare == 0) return false;

            // Kelly position sizing
            double winRate = (wins + losses > 10)
                ? (double) wins / (wins + losses) : 0.5;
            double avgW = (wins > 0)    ? totalWin  / wins   : riskPerShare * ATR_TARGET_MULTIPLIER;
            double avgL = (losses > 0)  ? totalLoss / losses : riskPerShare;
            double kelly = MathLib.kellyCriterion(winRate, avgW, avgL);
            kelly = Math.min(kelly, 0.25); // cap at 25 % of Kelly
            double positionSize = portfolioEquity * Math.min(kelly, MAX_POSITION_PCT);

            // portfolio heat check
            double newRisk = positionSize * (riskPerShare / price);
            if (totalRisk() + newRisk > portfolioEquity * MAX_PORTFOLIO_HEAT)
                return false;

            int qty = (int)(positionSize / price);
            if (qty < 1) return false;

            double cost = qty * price * (1 + COMMISSION_PCT);
            if (cost > cash) return false;

            cash -= cost;
            openPositions.add(new Position(symbol, strategy, direction,
                                           price, stop, target, qty, date, newRisk));
            return true;
        }

        /**
         * Check each open position against stop / target on the latest bar.
         */
        void checkExits(String symbol, Bar bar, String date,
                        Map<String, Double> prices) {
            Iterator<Position> it = openPositions.iterator();
            while (it.hasNext()) {
                Position p = it.next();
                if (!p.symbol.equals(symbol)) continue;

                double exitPrice  = 0;
                String exitReason = "";

                // stop-loss
                if ((p.direction ==  1 && bar.low()  <= p.stopLoss) ||
                    (p.direction == -1 && bar.high() >= p.stopLoss)) {
                    exitPrice  = p.stopLoss;
                    exitReason = "STOP-LOSS";
                }
                // take-profit
                else if ((p.direction ==  1 && bar.high() >= p.takeProfit) ||
                         (p.direction == -1 && bar.low()  <= p.takeProfit)) {
                    exitPrice  = p.takeProfit;
                    exitReason = "TAKE-PROFIT";
                }

                if (exitPrice > 0) {
                    double pnl = (exitPrice - p.entryPrice) * p.direction * p.qty
                                 - exitPrice * p.qty * COMMISSION_PCT;
                    cash += exitPrice * p.qty * (1 - COMMISSION_PCT);
                    recordTrade(p, date, exitPrice, pnl, exitReason);
                    it.remove();
                }
            }

            // drawdown circuit-breaker
            double eq = equity(prices);
            if (eq > peakEquity) peakEquity = eq;
            double dd = (peakEquity - eq) / peakEquity;
            if (dd >= MAX_DRAWDOWN_LIMIT) {
                System.out.printf("%n⚠  MAX DRAWDOWN %.1f%% HIT — TRADING HALTED%n",
                                  dd * 100);
                haltTrading = true;
            }
        }

        private void recordTrade(Position p, String exitDate,
                                 double exitPrice, double pnl, String reason) {
            closedTrades.add(new Trade(p.symbol, p.strategy, p.entryDate,
                                       exitDate, p.direction, p.entryPrice,
                                       exitPrice, p.qty, pnl, reason));
            if (pnl > 0) { wins++;    totalWin  += pnl; }
            else          { losses++;  totalLoss -= pnl; }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  SYNTHETIC MARKET DATA GENERATOR
    //  (GBM + momentum + mean-reversion regimes)
    // ─────────────────────────────────────────────────────────
    static List<Bar> generateBars(String symbol, int days, double startPrice,
                                  long seed) {
        Random rng = new Random(seed);
        List<Bar> bars = new ArrayList<>();
        double price = startPrice;
        LocalDate date = LocalDate.of(2023, 1, 3);

        double drift = 0.0003, vol = 0.018, momentum = 0;

        for (int i = 0; i < days; i++) {
            // skip weekends
            while (date.getDayOfWeek() == DayOfWeek.SATURDAY ||
                   date.getDayOfWeek() == DayOfWeek.SUNDAY)
                date = date.plusDays(1);

            // GBM with momentum regime
            momentum = 0.85 * momentum + 0.15 * (rng.nextGaussian() * 0.5);
            double ret   = drift + (vol + Math.abs(momentum) * 0.01)
                           * rng.nextGaussian() + momentum * 0.003;
            double open  = price;
            double close = price * Math.exp(ret);
            double intraRange = price * vol * (0.5 + rng.nextDouble());
            double high  = Math.max(open, close) + intraRange * rng.nextDouble();
            double low   = Math.min(open, close) - intraRange * rng.nextDouble();
            long   vol_  = (long)(1_000_000 + rng.nextInt(5_000_000));

            bars.add(new Bar(date.toString(), open, high, low, close, vol_));
            price = close;
            date  = date.plusDays(1);
        }
        return bars;
    }

    // ─────────────────────────────────────────────────────────
    //  BACK-TEST ENGINE
    // ─────────────────────────────────────────────────────────
    static class BacktestEngine {
        String            symbol;
        List<Bar>         bars;
        PortfolioManager  pm;
        int               warmup = 50;    // bars needed before trading

        BacktestEngine(String symbol, List<Bar> bars, PortfolioManager pm) {
            this.symbol = symbol;
            this.bars   = bars;
            this.pm     = pm;
        }

        void run() {
            for (int i = warmup; i < bars.size(); i++) {
                Bar bar = bars.get(i);
                String date = bar.date();

                // build sub-arrays up to and including bar i
                double[] closes = new double[i + 1];
                double[] highs  = new double[i + 1];
                double[] lows   = new double[i + 1];
                for (int j = 0; j <= i; j++) {
                    closes[j] = bars.get(j).close();
                    highs[j]  = bars.get(j).high();
                    lows[j]   = bars.get(j).low();
                }

                double atrVal = MathLib.atr(highs, lows, closes, 14);
                Map<String, Double> prices = Map.of(symbol, bar.close());

                // ── EXIT CHECK (before entry) ────────────────
                pm.checkExits(symbol, bar, date, prices);

                if (pm.haltTrading) break;

                double equity = pm.equity(prices);
                pm.equityCurve.add(equity);

                // 1. EMA Crossover (9/21)
                int sig1 = Strategies.emaCrossover(closes, 9, 21);

                // 2. RSI Mean-Reversion (14, 30/70)
                int sig2 = Strategies.rsiMeanReversion(closes, 14, 30, 70);

                // 3. Bollinger Band (20, 2σ)
                int sig3 = Strategies.bollingerBreakout(closes, 20, 2.0);

                // 4. MACD (12/26/9)
                int sig4 = Strategies.macdSignal(closes, 12, 26, 9);

                // 5. Momentum ROC (10, 2 %)
                int sig5 = Strategies.momentumROC(closes, 10, 2.0);

                // ── ENSEMBLE VOTE ────────────────────────────
                int vote = sig1 + sig2 + sig3 + sig4 + sig5;

                String stratName;
                int    direction;
                if      (vote >= 1)  { direction =  1; stratName = "ENSEMBLE-LONG";  }
                else if (vote <= -1) { direction = -1; stratName = "ENSEMBLE-SHORT"; }
                // single-strategy high-confidence signals
                else if (sig1 != 0 && sig4 != 0 && sig1 == sig4) {
                    direction = sig1; stratName = "EMA+MACD";
                }
                else if (sig2 != 0 && sig5 != 0 && sig2 == sig5) {
                    direction = sig2; stratName = "RSI+ROC";
                }
                else if (sig1 != 0 && sig3 != 0 && sig1 == sig3) {
                    direction = sig1; stratName = "EMA+BOLL";
                }
                else if (sig2 != 0 && sig4 != 0 && sig2 == sig4) {
                    direction = sig2; stratName = "RSI+MACD";
                }
                else continue;

                boolean opened = pm.openPosition(symbol, stratName, direction,
                                                 bar.close(), atrVal, equity, date);
                if (opened) {
                    System.out.printf("  [%s] %-14s %-12s %-6s @ $%8.2f  qty=%d%n",
                        date, symbol, stratName,
                        direction == 1 ? "BUY ↑" : "SELL ↓",
                        bar.close(),
                        pm.openPositions.stream()
                            .filter(p -> p.symbol.equals(symbol))
                            .mapToInt(p -> p.qty).sum());
                }
            }

            // close any remaining positions at last price
            if (!bars.isEmpty()) {
                Bar last = bars.get(bars.size() - 1);
                pm.openPositions.removeIf(p -> {
                    if (!p.symbol.equals(symbol)) return false;
                    double pnl = (last.close() - p.entryPrice) * p.direction * p.qty
                                 - last.close() * p.qty * COMMISSION_PCT;
                    pm.cash += last.close() * p.qty;
                    pm.recordTrade(p, last.date(), last.close(), pnl, "END-OF-DATA");
                    return true;
                });
            }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  REPORTING
    // ─────────────────────────────────────────────────────────
    static void printReport(PortfolioManager pm) {
        List<Trade> trades = pm.closedTrades;
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║           QUANTITATIVE TRADER — BACK-TEST REPORT         ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        if (trades.isEmpty()) {
            System.out.println("  No trades executed.");
            return;
        }

        double[] equity = pm.equityCurve.stream()
                            .mapToDouble(Double::doubleValue).toArray();
        double finalEquity = equity[equity.length - 1];
        double totalReturn = (finalEquity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100;
        double maxDD       = MathLib.maxDrawdown(equity) * 100;

        // daily returns
        double[] dailyRet = new double[equity.length - 1];
        for (int i = 1; i < equity.length; i++)
            dailyRet[i - 1] = (equity[i] - equity[i - 1]) / equity[i - 1];
        double sharpe = MathLib.sharpeRatio(dailyRet);

        long wins   = trades.stream().filter(t -> t.pnl() > 0).count();
        long losses = trades.stream().filter(t -> t.pnl() <= 0).count();
        double winRate    = (double) wins / trades.size() * 100;
        double totalPnL   = trades.stream().mapToDouble(Trade::pnl).sum();
        double avgWin     = trades.stream().filter(t -> t.pnl() > 0)
                                  .mapToDouble(Trade::pnl).average().orElse(0);
        double avgLoss    = trades.stream().filter(t -> t.pnl() <= 0)
                                  .mapToDouble(t -> -t.pnl()).average().orElse(0);
        double profitFactor = (avgLoss > 0 && losses > 0)
                ? (avgWin * wins) / (avgLoss * losses) : Double.POSITIVE_INFINITY;

        DecimalFormat df2 = new DecimalFormat("#,##0.00");
        DecimalFormat df1 = new DecimalFormat("0.0");

        System.out.printf("  %-30s %s%n", "Initial Capital:",   "$" + df2.format(INITIAL_CAPITAL));
        System.out.printf("  %-30s %s%n", "Final Equity:",      "$" + df2.format(finalEquity));
        System.out.printf("  %-30s %s%%%n","Total Return:",      df1.format(totalReturn));
        System.out.printf("  %-30s %s%n", "Total P&L:",         "$" + df2.format(totalPnL));
        System.out.println("  ─────────────────────────────────────────────────────────");
        System.out.printf("  %-30s %d%n",  "Total Trades:",      trades.size());
        System.out.printf("  %-30s %d  (%.1f%%)%n", "Winning Trades:", wins, winRate);
        System.out.printf("  %-30s %d%n",  "Losing Trades:",     losses);
        System.out.printf("  %-30s $%s%n", "Avg Win:",           df2.format(avgWin));
        System.out.printf("  %-30s $%s%n", "Avg Loss:",          df2.format(avgLoss));
        System.out.printf("  %-30s %s%n",  "Profit Factor:",     df1.format(profitFactor));
        System.out.println("  ─────────────────────────────────────────────────────────");
        System.out.printf("  %-30s %s%n",  "Sharpe Ratio:",      df1.format(sharpe));
        System.out.printf("  %-30s %s%%%n","Max Drawdown:",      df1.format(maxDD));

        // Strategy breakdown
        System.out.println("\n  ┌──────────────────────────────────────────────────────┐");
        System.out.println("  │                  STRATEGY BREAKDOWN                  │");
        System.out.println("  └──────────────────────────────────────────────────────┘");
        trades.stream()
            .collect(Collectors.groupingBy(Trade::strategy))
            .forEach((strat, ts) -> {
                long w  = ts.stream().filter(t -> t.pnl() > 0).count();
                double pnl = ts.stream().mapToDouble(Trade::pnl).sum();
                System.out.printf("  %-18s trades=%3d  wins=%3d  P&L=$%s%n",
                    strat, ts.size(), w, df2.format(pnl));
            });

        // Last 10 trades
        System.out.println("\n  ┌──────────────────────────────────────────────────────────────────┐");
        System.out.println("  │                     LAST 10 CLOSED TRADES                        │");
        System.out.println("  └──────────────────────────────────────────────────────────────────┘");
        System.out.printf("  %-12s %-10s %-14s %-6s %-10s %-10s %s%n",
            "Date-Exit","Symbol","Strategy","Dir","Entry","Exit","P&L");
        System.out.println("  " + "─".repeat(80));
        int start = Math.max(0, trades.size() - 10);
        for (int i = start; i < trades.size(); i++) {
            Trade t = trades.get(i);
            System.out.printf("  %-12s %-10s %-14s %-6s %-10s %-10s $%s%n",
                t.exitDate(), t.symbol(), t.strategy(),
                t.direction() == 1 ? "LONG" : "SHORT",
                "$" + df2.format(t.entryPrice()),
                "$" + df2.format(t.exitPrice()),
                df2.format(t.pnl()));
        }

        // ASCII equity curve
        System.out.println("\n  ┌─ EQUITY CURVE ─────────────────────────────────────────┐");
        double[] eq = pm.equityCurve.stream().mapToDouble(Double::doubleValue).toArray();
        int barCount = 60, height = 12;
        double eqMin = Arrays.stream(eq).min().orElse(0);
        double eqMax = Arrays.stream(eq).max().orElse(1);
        int step = Math.max(1, eq.length / barCount);
        double[] sampled = new double[barCount];
        for (int i = 0; i < barCount; i++) {
            int idx = Math.min((int)((double) i / barCount * eq.length), eq.length - 1);
            sampled[i] = eq[idx];
        }
        for (int row = height; row >= 0; row--) {
            System.out.print("  │");
            double threshold = eqMin + (eqMax - eqMin) * row / height;
            for (double v : sampled) System.out.print(v >= threshold ? "█" : " ");
            if (row == height)  System.out.printf(" $%s%n", df2.format(eqMax));
            else if (row == 0)  System.out.printf(" $%s%n", df2.format(eqMin));
            else                System.out.println();
        }
        System.out.println("  └" + "─".repeat(barCount + 1) + "┘");
    }

    // ─────────────────────────────────────────────────────────
    //  MAIN
    // ─────────────────────────────────────────────────────────
    public static void main(String[] args) {
        System.out.println("══════════════════════════════════════════════════════════");
        System.out.println("   QUANTITATIVE AUTOMATED TRADING SYSTEM  —  v2.0         ");
        System.out.println("══════════════════════════════════════════════════════════");
        System.out.printf("  Strategies : EMA-Cross · RSI · Bollinger · MACD · ROC%n");
        System.out.printf("  Sizing     : Kelly Criterion%n");
        System.out.printf("  Risk Mgmt  : ATR Stop/Target · Max-DD Breaker · Heat Limit%n");
        System.out.printf("  Capital    : $%,.0f%n%n", INITIAL_CAPITAL);

        PortfolioManager pm = new PortfolioManager(INITIAL_CAPITAL);

        // ── Multi-symbol portfolio ───────────────────────────
        String[][] instruments = {
            {"AAPL",  "500", "42"},
            {"MSFT",  "400", "7" },
            {"GOOGL", "280", "13"},
            {"TSLA",  "250", "99"},
            {"SPY",   "450", "3" },
        };

        System.out.println("  Generating market data & running back-test …");
        System.out.println("  " + "─".repeat(70));

        for (String[] inst : instruments) {
            String sym   = inst[0];
            double start = Double.parseDouble(inst[1]);
            long   seed  = Long.parseLong(inst[2]);
            List<Bar> bars = generateBars(sym, 504, start, seed);  // ~2 years
            BacktestEngine engine = new BacktestEngine(sym, bars, pm);
            System.out.printf("%n  ► %s  (start=$%.2f, bars=%d)%n", sym, start, bars.size());
            engine.run();
        }

        printReport(pm);

        System.out.println();
        System.out.println("  Math used: EMA/SMA · RSI · Bollinger Bands · MACD ·");
        System.out.println("             ATR · ROC · Kelly Criterion · Sharpe Ratio");
        System.out.println("             Max Drawdown · GBM simulation");
        System.out.println("══════════════════════════════════════════════════════════");
    }
}
