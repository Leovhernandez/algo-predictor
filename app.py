"""
Prediction Market Trading System
================================
A quantitative approach to identifying +EV opportunities across prediction markets.

MATHEMATICAL FOUNDATIONS (First Principles):
============================================

1. EXPECTED VALUE (EV)
   - The average outcome if you repeated the bet infinitely
   - EV = (Probability of Win × Profit) - (Probability of Loss × Loss)
   - For a binary bet at price p with true probability q:
     EV = q × (1 - p) - (1 - q) × p = q - p
   - Positive EV means profitable in the long run

2. KELLY CRITERION
   - Optimal bet sizing to maximize long-term growth
   - Derived by John Kelly (Bell Labs, 1956) for information transmission
   - Full Kelly: f* = (p × b - q) / b  where:
     - p = probability of winning
     - q = probability of losing (1 - p)
     - b = odds received on the bet (net profit per $1 wagered)
   - For binary markets at price c with true prob p:
     f* = (p - c) / (1 - c)  [betting YES]
     f* = ((1-p) - c) / (1 - c)  [betting NO at price c]

3. WHY FRACTIONAL KELLY?
   - Full Kelly maximizes growth but has HIGH VARIANCE
   - Drawdowns can exceed 50% even with edge
   - Half Kelly: 75% of growth rate, 50% of variance
   - Quarter Kelly: 50% of growth rate, 25% of variance
   - RECOMMENDATION: Start with 0.25 (Quarter Kelly) for $100 bankroll
   - Increase to 0.5 (Half Kelly) once you verify your edge is real

4. VARIANCE AND STANDARD DEVIATION
   - Variance = E[(X - μ)²] = average squared deviation from mean
   - Standard Deviation = √Variance = same units as the data
   - For a binary bet: Var = p(1-p)
   - Higher variance = more uncertainty = wider confidence intervals

5. SHARPE RATIO (Risk-Adjusted Returns)
   - Sharpe = (Expected Return - Risk-Free Rate) / Standard Deviation
   - Higher is better: more return per unit of risk
   - >1.0 is good, >2.0 is excellent, >3.0 is exceptional
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm
from scipy.optimize import minimize_scalar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """
    Central configuration for the trading system.

    WHY a dataclass?
    - Immutable by default (safer than dicts)
    - Type hints provide documentation
    - Easy serialization to JSON for logging

    Q: What happens if you change bankroll mid-session?
    Try: Create two configs and compare their Kelly recommendations
    """
    # Capital Management
    bankroll: float = 100.0  # Starting capital in USD
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative start)

    # Edge Requirements
    min_ev: float = 0.03  # Minimum 3% edge to consider a bet
    min_ev_after_fees: float = 0.02  # After accounting for trading fees

    # Risk Limits
    max_position_pct: float = 0.20  # Never bet more than 20% on single position
    max_correlated_exposure: float = 0.30  # Max 30% in correlated positions
    daily_loss_limit: float = 0.15  # Stop trading if down 15% in a day

    # Liquidity Filters
    min_volume: int = 5000  # Minimum USD volume
    max_volume: int = 500000  # Avoid manipulated thin markets
    min_open_interest: int = 100  # Minimum contracts outstanding

    # Platform Fees (as decimal)
    kalshi_fee: float = 0.01  # 1% fee on profits
    polymarket_fee: float = 0.02  # 2% fee estimate

    # Backtesting
    simulation_runs: int = 10000  # Monte Carlo iterations
    confidence_level: float = 0.95  # For confidence intervals

    # Mode
    live_trading_enabled: bool = False  # SAFETY: Must explicitly enable


# Initialize default config
config = TradingConfig()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Platform(Enum):
    """Supported prediction market platforms."""
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    DKPREDICTIONS = "dkpredictions"


class BetSide(Enum):
    """Which side of a binary bet."""
    YES = "yes"
    NO = "no"


@dataclass
class Market:
    """
    Represents a single prediction market.

    ANATOMY OF A BINARY MARKET:
    - Two outcomes: YES or NO
    - Prices sum to ~$1.00 (minus spread)
    - YES price = market's implied probability of YES
    - Your edge = Your probability - Market price
    """
    ticker: str
    title: str
    platform: Platform

    # Prices (as decimals, 0-1)
    yes_price: float  # Cost to buy YES contract
    no_price: float   # Cost to buy NO contract
    yes_ask: float = None  # Best ask for YES
    no_ask: float = None   # Best ask for NO

    # Liquidity metrics
    volume: float = 0  # 24h volume in USD
    open_interest: int = 0  # Total contracts outstanding

    # Metadata
    category: str = ""
    expiry: datetime = None

    @property
    def spread(self) -> float:
        """
        Bid-ask spread as a measure of market efficiency.

        WHY does spread matter?
        - Wide spread = illiquid = hard to enter/exit
        - Your effective edge is reduced by spread
        - Spread > 5% is usually too expensive
        """
        return (self.yes_price + self.no_price) - 1.0

    @property
    def implied_prob_yes(self) -> float:
        """Market's implied probability of YES outcome."""
        # Adjust for overround (spread)
        total = self.yes_price + self.no_price
        return self.yes_price / total if total > 0 else 0.5


@dataclass
class Prediction:
    """
    Your model's prediction for a market outcome.

    The GAP between your prediction and market price is your EDGE.
    No edge = no bet, regardless of confidence.
    """
    market: Market
    model_prob_yes: float  # Your estimated probability (0-1)
    confidence: float = 1.0  # How confident in your model (0-1)
    model_name: str = "manual"  # Which model generated this
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def edge_yes(self) -> float:
        """Edge if betting YES = your prob - market price."""
        return self.model_prob_yes - self.market.yes_price

    @property
    def edge_no(self) -> float:
        """Edge if betting NO = (1 - your prob) - market NO price."""
        return (1 - self.model_prob_yes) - self.market.no_price

    @property
    def best_side(self) -> Tuple[BetSide, float]:
        """Returns the side with higher edge and that edge value."""
        if self.edge_yes > self.edge_no:
            return BetSide.YES, self.edge_yes
        return BetSide.NO, self.edge_no


@dataclass
class Position:
    """
    A recommended or actual position in a market.

    POSITION SIZING is crucial:
    - Too small: Leaving money on the table
    - Too large: Risk of ruin even with edge
    - Kelly optimal: Maximizes long-term growth
    """
    market: Market
    side: BetSide
    contracts: int
    entry_price: float

    # Kelly calculation inputs
    model_prob: float
    edge: float
    kelly_fraction_used: float

    # Risk metrics
    max_loss: float = 0  # Worst case loss
    expected_profit: float = 0  # EV of position

    # Status
    is_filled: bool = False
    fill_price: float = None
    fill_time: datetime = None

    @property
    def cost(self) -> float:
        """Total capital required to enter position."""
        return self.contracts * self.entry_price

    @property
    def potential_profit(self) -> float:
        """Profit if bet wins."""
        return self.contracts * (1 - self.entry_price)


# =============================================================================
# KELLY CRITERION - The Mathematical Foundation
# =============================================================================

class KellyCalculator:
    """
    Implements Kelly Criterion for optimal position sizing.

    DERIVATION (First Principles):
    ==============================
    Goal: Maximize E[log(Wealth)] - expected logarithmic growth

    After one bet:
      Wealth_new = Wealth × (1 + f×b)  if win (prob p)
      Wealth_new = Wealth × (1 - f)    if lose (prob q = 1-p)

    Expected log growth:
      G(f) = p × log(1 + f×b) + q × log(1 - f)

    Take derivative, set to zero:
      dG/df = p×b/(1+f×b) - q/(1-f) = 0

    Solve for f:
      f* = (p×b - q) / b = (p×(b+1) - 1) / b

    For binary markets (win pays 1/price - 1):
      b = (1 - price) / price
      f* = (p - price) / (1 - price)

    Q: Why log(Wealth) instead of just Wealth?
    A: Log transforms multiplicative growth to additive, matching how
       compound returns work. Maximizing E[Wealth] can lead to ruin
       because it ignores the asymmetry of percentage gains/losses.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_kelly_fraction(
        self,
        prob_win: float,
        price: float,
        side: BetSide
    ) -> float:
        """
        Calculate optimal Kelly fraction for a binary bet.

        Args:
            prob_win: Your estimated probability of winning (0-1)
            price: Market price you'd pay (0-1)
            side: Whether betting YES or NO

        Returns:
            Optimal fraction of bankroll to bet (before applying fractional Kelly)

        EXAMPLE:
        --------
        Market: "Will it rain tomorrow?" YES at $0.40
        Your model says 60% chance of rain.

        If betting YES:
          - prob_win = 0.60
          - price = 0.40
          - odds = (1 - 0.40) / 0.40 = 1.5 (you win $1.50 per $1 bet)
          - kelly = (0.60 × 1.5 - 0.40) / 1.5 = (0.90 - 0.40) / 1.5 = 0.333
          - Bet 33.3% of bankroll (full Kelly)

        Q: What if prob_win < price? Try it and see what Kelly returns.
        """
        if price <= 0 or price >= 1:
            return 0.0

        # For YES bets: you pay 'price', win '1 - price' profit
        # For NO bets: same math, but prob_win is (1 - your_yes_prob)

        # Odds = profit per dollar risked
        odds = (1 - price) / price

        # Kelly formula
        prob_lose = 1 - prob_win
        kelly = (prob_win * odds - prob_lose) / odds

        # Alternative equivalent formula (easier to verify):
        # kelly = (prob_win - price) / (1 - price)
        kelly_alt = (prob_win - price) / (1 - price)

        # They should match (sanity check)
        assert abs(kelly - kelly_alt) < 0.0001, f"Kelly formulas don't match: {kelly} vs {kelly_alt}"

        return max(0, kelly)  # Never negative (don't bet if no edge)

    def calculate_position_size(
        self,
        prediction: Prediction,
        bankroll: float = None
    ) -> Optional[Position]:
        """
        Calculate optimal position size for a prediction.

        Applies:
        1. Kelly criterion for optimal sizing
        2. Fractional Kelly for variance reduction
        3. Maximum position limits for risk management
        4. Minimum edge threshold filter

        Returns None if bet doesn't meet criteria.
        """
        bankroll = bankroll or self.config.bankroll

        # Determine best side to bet
        side, edge = prediction.best_side

        # Check minimum edge threshold
        if edge < self.config.min_ev:
            logger.debug(f"Edge {edge:.2%} below threshold {self.config.min_ev:.2%}")
            return None

        # Get relevant price
        if side == BetSide.YES:
            price = prediction.market.yes_price
            prob_win = prediction.model_prob_yes
        else:
            price = prediction.market.no_price
            prob_win = 1 - prediction.model_prob_yes

        # Calculate full Kelly
        full_kelly = self.calculate_kelly_fraction(prob_win, price, side)

        if full_kelly <= 0:
            return None

        # Apply fractional Kelly
        kelly_adjusted = full_kelly * self.config.kelly_fraction

        # Apply maximum position limit
        kelly_adjusted = min(kelly_adjusted, self.config.max_position_pct)

        # Calculate dollar amount and contracts
        dollar_amount = bankroll * kelly_adjusted
        contracts = int(dollar_amount / price)

        if contracts <= 0:
            return None

        # Create position
        position = Position(
            market=prediction.market,
            side=side,
            contracts=contracts,
            entry_price=price,
            model_prob=prob_win,
            edge=edge,
            kelly_fraction_used=kelly_adjusted,
            max_loss=contracts * price,
            expected_profit=contracts * edge
        )

        return position

    def kelly_growth_rate(self, prob_win: float, price: float, fraction: float) -> float:
        """
        Calculate expected logarithmic growth rate for a given bet size.

        This is the objective function Kelly maximizes.

        G(f) = p × log(1 + f×b) + (1-p) × log(1 - f)

        WHERE:
        - f = fraction of bankroll bet
        - b = odds (profit per dollar if win)
        - p = probability of winning

        Q: What's the growth rate at f=0? At f=1? At f=Kelly?
        Try: Plot G(f) for different values to visualize.
        """
        if fraction <= 0 or fraction >= 1:
            return float('-inf') if fraction >= 1 else 0

        odds = (1 - price) / price

        # Expected log growth
        growth = (
            prob_win * np.log(1 + fraction * odds) +
            (1 - prob_win) * np.log(1 - fraction)
        )

        return growth


# =============================================================================
# PROBABILITY MODELS FOR ECONOMIC EVENTS
# =============================================================================

class EconomicModels:
    """
    Statistical models for economic indicators.

    FUNDAMENTAL INSIGHT:
    ====================
    Economic data releases (jobs, CPI, Fed decisions) are the PRIMARY
    drivers of prediction market movements. If you can model these
    better than the market, you have edge.

    KEY DISTRIBUTIONS:
    ------------------
    1. NORMAL (Gaussian): Jobs reports, GDP growth
       - Symmetric around mean
       - 68% within 1 SD, 95% within 2 SD, 99.7% within 3 SD

    2. LOGNORMAL: Price indices (CPI), because prices can't go negative
       - Skewed right (long tail of high values)
       - Use when modeling percentage changes

    3. BERNOULLI: Fed decisions (hike/cut/hold)
       - Binary or categorical outcomes
       - Model directly as probabilities
    """

    @staticmethod
    def normal_prob_above(threshold: float, mean: float, std: float) -> float:
        """
        Probability that a normally distributed variable exceeds threshold.

        P(X > threshold) = 1 - Φ((threshold - mean) / std)

        WHERE Φ is the standard normal CDF.

        EXAMPLE:
        --------
        Jobs report: consensus = 150K, historical SD = 80K
        What's P(jobs > 200K)?

        z = (200000 - 150000) / 80000 = 0.625
        P(X > 200K) = 1 - Φ(0.625) = 1 - 0.734 = 0.266 (26.6%)

        Q: How does the probability change if SD doubles?
        """
        if std <= 0:
            raise ValueError("Standard deviation must be positive")

        z_score = (threshold - mean) / std
        prob_below = norm.cdf(z_score)
        return 1 - prob_below

    @staticmethod
    def normal_prob_between(lower: float, upper: float, mean: float, std: float) -> float:
        """
        Probability that value falls in range [lower, upper].

        P(lower < X < upper) = Φ((upper-μ)/σ) - Φ((lower-μ)/σ)
        """
        z_lower = (lower - mean) / std
        z_upper = (upper - mean) / std
        return norm.cdf(z_upper) - norm.cdf(z_lower)

    @staticmethod
    def lognormal_prob_above(threshold: float, mean: float, std: float) -> float:
        """
        Probability for lognormally distributed variable (e.g., CPI).

        WHY LOGNORMAL for prices?
        - Prices can't be negative
        - Percentage changes are roughly normal
        - If Y = e^X where X ~ Normal, then Y ~ Lognormal

        PARAMETERS:
        - mean, std are of the UNDERLYING normal distribution (log space)
        - If you have mean/std of actual values, convert first:
          μ_log = log(mean²/√(var + mean²))
          σ_log = √(log(1 + var/mean²))
        """
        if threshold <= 0:
            return 1.0  # Lognormal is always positive

        # Parameters are for the underlying normal
        prob_below = lognorm.cdf(threshold, s=std, scale=np.exp(mean))
        return 1 - prob_below

    @staticmethod
    def estimate_consensus_uncertainty(
        forecasts: List[float],
        historical_errors: List[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate mean and SD from analyst forecasts and/or historical errors.

        TWO SOURCES OF UNCERTAINTY:
        1. Disagreement among forecasters (SD of forecasts)
        2. Historical forecast errors (even consensus is often wrong)

        We combine both for robust uncertainty estimation.

        Args:
            forecasts: List of analyst predictions
            historical_errors: Past (actual - forecast) values

        Returns:
            (mean, std) tuple

        Q: Which source of uncertainty is usually larger? Why?
        """
        mean = np.mean(forecasts)

        # Forecast dispersion
        forecast_std = np.std(forecasts, ddof=1) if len(forecasts) > 1 else 0

        # Historical error std
        if historical_errors and len(historical_errors) > 1:
            error_std = np.std(historical_errors, ddof=1)
        else:
            error_std = forecast_std * 1.5  # Assume errors > dispersion

        # Combined uncertainty (root sum of squares)
        # This assumes the two sources are independent
        combined_std = np.sqrt(forecast_std**2 + error_std**2)

        return mean, combined_std


class JobsReportModel:
    """
    Model for Non-Farm Payrolls (NFP) / Jobs Report.

    BACKGROUND:
    ===========
    - Released first Friday of each month by BLS
    - Most market-moving economic indicator
    - Consensus forecast typically available from Bloomberg, Reuters
    - Historical SD of forecast errors: ~80,000-100,000 jobs

    MARKET STRUCTURE:
    - Kalshi offers "Jobs above X" contracts for various thresholds
    - Thresholds typically at 50K intervals around consensus
    """

    def __init__(
        self,
        consensus_mean: float,
        consensus_std: float = 83000  # Historical error SD
    ):
        """
        Args:
            consensus_mean: Economist consensus forecast (e.g., 150000)
            consensus_std: Expected standard deviation of actual vs consensus
        """
        self.mean = consensus_mean
        self.std = consensus_std

    def prob_above(self, threshold: float) -> float:
        """Probability jobs report exceeds threshold."""
        return EconomicModels.normal_prob_above(threshold, self.mean, self.std)

    def prob_in_range(self, lower: float, upper: float) -> float:
        """Probability jobs falls in range."""
        return EconomicModels.normal_prob_between(lower, upper, self.mean, self.std)

    def generate_threshold_probs(
        self,
        thresholds: List[float] = None
    ) -> Dict[float, float]:
        """
        Generate probabilities for multiple thresholds.

        Useful for scanning Kalshi markets.
        """
        if thresholds is None:
            # Default: thresholds around consensus
            offsets = [-100000, -50000, 0, 50000, 100000, 150000, 200000]
            thresholds = [self.mean + offset for offset in offsets]

        return {t: self.prob_above(t) for t in thresholds}


class CPIModel:
    """
    Model for Consumer Price Index releases.

    CPI BASICS:
    ===========
    - Monthly inflation measure
    - Year-over-year (YoY) is most watched
    - Core CPI excludes food/energy (less volatile)
    - Markets price "CPI above X%" contracts

    WHY LOGNORMAL?
    - Inflation rates can't go below ~-2% (deflation floor)
    - Right-skewed: easier to get 5% inflation than -5%
    - But for small ranges near consensus, normal is fine
    """

    def __init__(
        self,
        consensus_yoy: float,  # e.g., 3.2 for 3.2%
        consensus_std: float = 0.15  # Historical error in percentage points
    ):
        self.mean = consensus_yoy
        self.std = consensus_std

    def prob_above(self, threshold: float) -> float:
        """Probability CPI YoY exceeds threshold."""
        # Use normal for typical ranges (close enough to consensus)
        return EconomicModels.normal_prob_above(threshold, self.mean, self.std)


class FedDecisionModel:
    """
    Model for Federal Reserve interest rate decisions.

    FED FUNDS FUTURES:
    ==================
    - CME FedWatch tool provides market-implied probabilities
    - You can use these as a baseline and adjust
    - Decisions: Hike (+25bp), Cut (-25bp), Hold (0)

    Your edge might come from:
    - Better interpretation of Fed speakers
    - Economic data released since futures priced
    - Different weighting of inflation vs employment
    """

    def __init__(
        self,
        prob_hike: float = 0.0,
        prob_cut: float = 0.0,
        prob_hold: float = 1.0
    ):
        """Probabilities must sum to 1."""
        total = prob_hike + prob_cut + prob_hold
        self.prob_hike = prob_hike / total
        self.prob_cut = prob_cut / total
        self.prob_hold = prob_hold / total

    def prob_no_change(self) -> float:
        """Probability Fed holds rates steady."""
        return self.prob_hold

    def prob_rate_move(self) -> float:
        """Probability of any rate change."""
        return 1 - self.prob_hold


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class BacktestTrade:
    """Record of a single trade in backtest."""
    entry_time: datetime
    market_ticker: str
    side: BetSide
    contracts: int
    entry_price: float
    model_prob: float
    edge: float

    # Outcome
    outcome: bool = None  # True if bet won
    exit_price: float = None
    pnl: float = None

    def resolve(self, outcome: bool):
        """Resolve the trade with actual outcome."""
        self.outcome = outcome
        if outcome:
            self.pnl = self.contracts * (1 - self.entry_price)
        else:
            self.pnl = -self.contracts * self.entry_price
        self.exit_price = 1.0 if outcome else 0.0


class BacktestEngine:
    """
    Backtesting engine for prediction market strategies.

    BACKTESTING FUNDAMENTALS:
    =========================

    PURPOSE: Test if your strategy WOULD HAVE worked historically.

    CRITICAL LIMITATIONS:
    1. LOOK-AHEAD BIAS: Using information you wouldn't have had
       - Example: Using actual jobs number to "predict" jobs
       - Solution: Only use data available at decision time

    2. SURVIVORSHIP BIAS: Only testing on markets that still exist
       - Failed/delisted markets might have been your worst bets

    3. OVERFITTING: Tuning parameters to match past data perfectly
       - A strategy that "predicts" coin flips 100% historically is useless
       - Solution: Out-of-sample testing, cross-validation

    4. EXECUTION ASSUMPTIONS: Assuming you get your price
       - Slippage, fees, and timing matter in reality

    Q: How would you test for overfitting in this context?
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.starting_bankroll = config.bankroll

    def add_trade(self, trade: BacktestTrade):
        """Record a trade."""
        self.trades.append(trade)

    def run_backtest(
        self,
        historical_predictions: List[Tuple[Prediction, bool]],
        kelly_calc: KellyCalculator
    ) -> Dict:
        """
        Run backtest on historical predictions with known outcomes.

        Args:
            historical_predictions: List of (Prediction, actual_outcome) tuples
            kelly_calc: Kelly calculator for position sizing

        Returns:
            Dictionary of performance metrics
        """
        bankroll = self.starting_bankroll
        self.equity_curve = [bankroll]
        self.trades = []

        for prediction, actual_outcome in historical_predictions:
            # Calculate position
            position = kelly_calc.calculate_position_size(prediction, bankroll)

            if position is None:
                continue  # No bet meets criteria

            # Record trade
            trade = BacktestTrade(
                entry_time=prediction.timestamp,
                market_ticker=prediction.market.ticker,
                side=position.side,
                contracts=position.contracts,
                entry_price=position.entry_price,
                model_prob=position.model_prob,
                edge=position.edge
            )

            # Determine if bet won
            if position.side == BetSide.YES:
                bet_won = actual_outcome
            else:
                bet_won = not actual_outcome

            trade.resolve(bet_won)
            self.trades.append(trade)

            # Update bankroll
            bankroll += trade.pnl
            self.equity_curve.append(bankroll)

            # Check for ruin
            if bankroll <= 0:
                logger.warning("Bankroll depleted - stopping backtest")
                break

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        METRICS EXPLAINED:
        ==================

        1. WIN RATE: % of bets won
           - NOT the same as profitability!
           - A 30% win rate can be profitable with good odds

        2. PROFIT FACTOR: Gross profit / Gross loss
           - >1 = profitable, >2 = good, >3 = excellent

        3. SHARPE RATIO: Risk-adjusted return
           - (Mean return - Risk free) / Std of returns
           - Annualize by × √(252) for daily or √(12) for monthly

        4. MAX DRAWDOWN: Largest peak-to-trough decline
           - Critical for survival
           - 50% drawdown requires 100% gain to recover!

        5. CALMAR RATIO: Annual return / Max drawdown
           - Higher = better risk-adjusted returns
        """
        if not self.trades:
            return {"error": "No trades to analyze"}

        # Basic counts
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.outcome)
        losses = total_trades - wins

        # Win rate
        win_rate = wins / total_trades if total_trades > 0 else 0

        # P&L
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0

        # Expected value
        expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Returns series for Sharpe
        returns = [t.pnl / self.starting_bankroll for t in self.trades]
        mean_return = np.mean(returns) if returns else 0
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0

        # Sharpe (assuming risk-free = 0 for simplicity)
        sharpe = mean_return / std_return if std_return > 0 else 0

        # Max drawdown
        max_dd, max_dd_duration = self._calculate_max_drawdown()

        # Final return
        final_equity = self.equity_curve[-1] if self.equity_curve else self.starting_bankroll
        total_return = (final_equity - self.starting_bankroll) / self.starting_bankroll

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return_pct": total_return * 100,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "expected_value": expected_value,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "max_drawdown_duration": max_dd_duration,
            "final_equity": final_equity
        }

    def _calculate_max_drawdown(self) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and its duration.

        DRAWDOWN = (Peak - Current) / Peak

        This measures the worst loss from any high point.
        Essential for understanding risk of ruin.
        """
        if len(self.equity_curve) < 2:
            return 0.0, 0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max

        max_dd = np.max(drawdowns)
        max_dd_idx = np.argmax(drawdowns)

        # Find duration (bars from peak to recovery)
        peak_idx = np.argmax(equity[:max_dd_idx + 1])

        # Find recovery point (if any)
        recovery_idx = max_dd_idx
        for i in range(max_dd_idx, len(equity)):
            if equity[i] >= equity[peak_idx]:
                recovery_idx = i
                break

        duration = recovery_idx - peak_idx

        return max_dd, duration


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for risk analysis.

    WHY MONTE CARLO?
    ================
    Backtests show ONE path through history. But that path was just
    one of infinitely many possible outcomes. Monte Carlo simulates
    MANY possible paths to understand:

    1. DISTRIBUTION of outcomes (not just average)
    2. PROBABILITY of ruin at different bet sizes
    3. CONFIDENCE INTERVALS for expected performance
    4. WORST-CASE scenarios (tail risk)

    METHOD:
    -------
    1. Define your edge (win rate, odds)
    2. Simulate thousands of bet sequences
    3. Analyze the distribution of final bankrolls

    Q: With 55% win rate at even odds, what's P(ruin) after 100 bets
       betting 10% each time vs 5% each time? Simulate to find out.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def simulate_strategy(
        self,
        prob_win: float,
        odds: float,  # Net profit if win (e.g., 1.0 = even money)
        bet_fraction: float,
        n_bets: int,
        n_simulations: int = None
    ) -> Dict:
        """
        Simulate a betting strategy many times.

        Args:
            prob_win: Probability of winning each bet
            odds: Profit per dollar if win
            bet_fraction: Fraction of bankroll to bet
            n_bets: Number of bets per simulation
            n_simulations: Number of simulations to run

        Returns:
            Distribution statistics of outcomes
        """
        n_simulations = n_simulations or self.config.simulation_runs

        # Run simulations
        final_bankrolls = []
        ruin_count = 0
        max_drawdowns = []

        for _ in range(n_simulations):
            bankroll = self.config.bankroll
            peak = bankroll
            max_dd = 0

            for _ in range(n_bets):
                bet_size = bankroll * bet_fraction

                # Simulate outcome
                if np.random.random() < prob_win:
                    bankroll += bet_size * odds  # Win
                else:
                    bankroll -= bet_size  # Lose

                # Track drawdown
                if bankroll > peak:
                    peak = bankroll
                else:
                    dd = (peak - bankroll) / peak
                    max_dd = max(max_dd, dd)

                # Check for ruin
                if bankroll <= 0:
                    ruin_count += 1
                    break

            final_bankrolls.append(max(0, bankroll))
            max_drawdowns.append(max_dd)

        # Calculate statistics
        final_bankrolls = np.array(final_bankrolls)

        return {
            "mean_final_bankroll": np.mean(final_bankrolls),
            "median_final_bankroll": np.median(final_bankrolls),
            "std_final_bankroll": np.std(final_bankrolls),
            "min_final_bankroll": np.min(final_bankrolls),
            "max_final_bankroll": np.max(final_bankrolls),
            "percentile_5": np.percentile(final_bankrolls, 5),
            "percentile_25": np.percentile(final_bankrolls, 25),
            "percentile_75": np.percentile(final_bankrolls, 75),
            "percentile_95": np.percentile(final_bankrolls, 95),
            "ruin_probability": ruin_count / n_simulations,
            "mean_max_drawdown": np.mean(max_drawdowns),
            "simulations": n_simulations,
            "bets_per_sim": n_bets
        }

    def compare_kelly_fractions(
        self,
        prob_win: float,
        odds: float,
        n_bets: int = 100,
        fractions: List[float] = None
    ) -> pd.DataFrame:
        """
        Compare different Kelly fractions.

        This helps you understand the variance/growth tradeoff.

        EXPECTED RESULTS:
        - Full Kelly: Highest mean, highest variance, highest ruin risk
        - Half Kelly: ~75% growth, ~50% variance
        - Quarter Kelly: ~50% growth, ~25% variance, low ruin risk
        """
        if fractions is None:
            # Calculate optimal Kelly first
            full_kelly = (prob_win * odds - (1 - prob_win)) / odds
            fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            fractions = [f * full_kelly for f in fractions]
            fractions = [f for f in fractions if 0 < f < 1]  # Valid range

        results = []
        for frac in fractions:
            sim = self.simulate_strategy(prob_win, odds, frac, n_bets)
            sim["kelly_multiple"] = frac / full_kelly if full_kelly > 0 else 0
            sim["bet_fraction"] = frac
            results.append(sim)

        return pd.DataFrame(results)


# =============================================================================
# CORRELATION AND PORTFOLIO ANALYSIS
# =============================================================================

class CorrelationAnalyzer:
    """
    Analyze correlations between prediction markets.

    WHY CORRELATION MATTERS:
    ========================

    If two bets are correlated, your actual risk is HIGHER than
    if you sized them independently.

    EXAMPLE:
    --------
    Bet 1: "Jobs report above 150K" - betting YES
    Bet 2: "Fed will cut rates" - betting YES

    These are NEGATIVELY correlated:
    - Strong jobs → less likely Fed cuts
    - If jobs bet loses, Fed bet more likely to win

    This is GOOD - natural hedge reduces portfolio variance.

    CONVERSELY:
    Bet 1: "Jobs above 150K" - YES
    Bet 2: "Jobs above 100K" - YES

    These are POSITIVELY correlated:
    - If first loses (jobs < 150K), second could still win
    - But if second loses (jobs < 100K), first definitely loses too

    Sizing both fully ignores this dependency and overstates bankroll.
    """

    @staticmethod
    def estimate_correlation(
        market1_category: str,
        market2_category: str,
        historical_data: pd.DataFrame = None
    ) -> float:
        """
        Estimate correlation between two markets.

        Without historical data, use economic intuition:
        - Same event, different thresholds: HIGH positive (0.7-0.9)
        - Related events (jobs & Fed): MODERATE negative (-0.3 to -0.5)
        - Unrelated events: LOW (~0)
        """
        # Heuristic correlations (replace with data if available)
        correlation_map = {
            ("JOBS", "JOBS"): 0.8,  # Same event type
            ("CPI", "CPI"): 0.8,
            ("FED", "FED"): 0.8,
            ("JOBS", "FED"): -0.4,  # Strong jobs → less cuts
            ("CPI", "FED"): 0.5,   # High CPI → more hikes
            ("JOBS", "CPI"): 0.2,   # Weak relationship
        }

        # Normalize to alphabetical order for lookup
        key = tuple(sorted([market1_category, market2_category]))

        return correlation_map.get(key, 0.0)

    @staticmethod
    def adjust_position_for_correlation(
        positions: List[Position],
        max_correlated_exposure: float
    ) -> List[Position]:
        """
        Adjust position sizes to account for correlations.

        If correlated positions exceed max exposure, scale down proportionally.

        This is a simplified approach. Production systems use
        proper portfolio optimization (mean-variance, etc.).
        """
        # Group by category
        category_exposure = {}
        for pos in positions:
            cat = pos.market.category
            exposure = pos.cost
            category_exposure[cat] = category_exposure.get(cat, 0) + exposure

        # Check each category against limit
        total_bankroll = sum(p.cost for p in positions) / positions[0].kelly_fraction_used if positions else 1

        adjusted_positions = []
        for pos in positions:
            cat = pos.market.category
            cat_exposure = category_exposure.get(cat, 0)
            max_allowed = total_bankroll * max_correlated_exposure

            if cat_exposure > max_allowed:
                # Scale down
                scale_factor = max_allowed / cat_exposure
                new_contracts = int(pos.contracts * scale_factor)
                if new_contracts > 0:
                    adjusted_pos = Position(
                        market=pos.market,
                        side=pos.side,
                        contracts=new_contracts,
                        entry_price=pos.entry_price,
                        model_prob=pos.model_prob,
                        edge=pos.edge,
                        kelly_fraction_used=pos.kelly_fraction_used * scale_factor,
                        max_loss=new_contracts * pos.entry_price,
                        expected_profit=new_contracts * pos.edge
                    )
                    adjusted_positions.append(adjusted_pos)
            else:
                adjusted_positions.append(pos)

        return adjusted_positions


# =============================================================================
# TRADE EXECUTION (Disabled by default)
# =============================================================================

class TradeExecutor:
    """
    Handles actual trade execution across platforms.

    SAFETY FIRST:
    =============
    This class is disabled by default. Live trading requires:
    1. Setting config.live_trading_enabled = True
    2. Understanding that REAL MONEY is at risk
    3. Starting with small sizes to verify execution

    EXECUTION CONSIDERATIONS:
    -------------------------
    1. SLIPPAGE: You may not get your exact price
       - Use limit orders when possible
       - Account for 0.5-1% slippage in edge calculations

    2. FEES: Reduce your effective edge
       - Kalshi: ~1% on profits
       - Factor into position sizing

    3. TIMING: Markets move, execute quickly
       - Price you analyzed may be gone
       - Have fallback logic for changed prices
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trade_log: List[Dict] = []

    def execute(self, position: Position) -> Dict:
        """
        Execute a trade (or simulate if live trading disabled).

        Returns execution result.
        """
        if not self.config.live_trading_enabled:
            return self._simulate_execution(position)

        # LIVE EXECUTION
        # Uncomment and implement when ready
        """
        if position.market.platform == Platform.KALSHI:
            return self._execute_kalshi(position)
        elif position.market.platform == Platform.POLYMARKET:
            return self._execute_polymarket(position)
        else:
            return {"error": f"Unsupported platform: {position.market.platform}"}
        """

        return {"error": "Live trading not implemented yet"}

    def _simulate_execution(self, position: Position) -> Dict:
        """Simulate execution for testing."""
        result = {
            "status": "SIMULATED",
            "position": asdict(position) if hasattr(position, '__dict__') else str(position),
            "market": position.market.ticker,
            "side": position.side.value,
            "contracts": position.contracts,
            "price": position.entry_price,
            "cost": position.cost,
            "timestamp": datetime.now().isoformat(),
            "message": "Live trading disabled. Set config.live_trading_enabled=True to execute."
        }

        self.trade_log.append(result)
        return result

    def _execute_kalshi(self, position: Position) -> Dict:
        """
        Execute trade on Kalshi.

        API call structure:
        client.create_order(
            ticker=ticker,
            side=side.lower(),  # "yes" or "no"
            count=contracts,
            price=int(price * 100),  # Price in cents
            type="limit"
        )
        """
        # Implementation would go here
        # from api-tester import KalshiAuthenticator
        # ... actual API calls
        pass

    def _execute_polymarket(self, position: Position) -> Dict:
        """Execute trade on Polymarket (requires wallet signature)."""
        # Implementation would go here
        pass


# =============================================================================
# MAIN TRADING SYSTEM
# =============================================================================

class PredictionMarketTrader:
    """
    Main trading system orchestrating all components.

    WORKFLOW:
    =========
    1. Fetch markets from all platforms
    2. Generate model predictions
    3. Calculate Kelly-optimal positions
    4. Apply correlation adjustments
    5. Execute (or simulate) trades
    6. Log results

    USAGE:
    ------
    trader = PredictionMarketTrader(config)

    # Add your predictions
    trader.add_prediction(market, model_prob)

    # Get recommendations
    recommendations = trader.get_recommendations()

    # Backtest
    results = trader.run_backtest(historical_data)

    # Execute (when ready)
    trader.execute_recommendations()
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.kelly = KellyCalculator(self.config)
        self.executor = TradeExecutor(self.config)
        self.backtester = BacktestEngine(self.config)
        self.monte_carlo = MonteCarloSimulator(self.config)

        self.markets: List[Market] = []
        self.predictions: List[Prediction] = []
        self.recommendations: List[Position] = []

    def add_market(self, market: Market):
        """Add a market to analyze."""
        self.markets.append(market)

    def add_prediction(self, market: Market, model_prob_yes: float, confidence: float = 1.0):
        """
        Add your model's prediction for a market.

        Args:
            market: The market being predicted
            model_prob_yes: Your probability estimate (0-1)
            confidence: How confident in this prediction (0-1)
        """
        pred = Prediction(
            market=market,
            model_prob_yes=model_prob_yes,
            confidence=confidence
        )
        self.predictions.append(pred)

    def get_recommendations(self) -> List[Position]:
        """
        Generate position recommendations from all predictions.

        Steps:
        1. Calculate Kelly size for each prediction
        2. Filter by edge threshold
        3. Adjust for correlations
        4. Return final recommendations
        """
        positions = []

        for prediction in self.predictions:
            # Skip low-confidence predictions
            if prediction.confidence < 0.5:
                continue

            position = self.kelly.calculate_position_size(prediction)
            if position:
                positions.append(position)

        # Adjust for correlations
        adjusted = CorrelationAnalyzer.adjust_position_for_correlation(
            positions,
            self.config.max_correlated_exposure
        )

        self.recommendations = adjusted
        return adjusted

    def print_recommendations(self):
        """Display recommendations in a readable format."""
        if not self.recommendations:
            print("No recommendations - run get_recommendations() first or no edges found.")
            return

        print("\n" + "=" * 70)
        print("TRADE RECOMMENDATIONS")
        print("=" * 70)
        print(f"Bankroll: ${self.config.bankroll:.2f}")
        print(f"Kelly Fraction: {self.config.kelly_fraction:.0%}")
        print(f"Min Edge: {self.config.min_ev:.1%}")
        print("-" * 70)

        total_cost = 0
        total_ev = 0

        for i, pos in enumerate(self.recommendations, 1):
            print(f"\n{i}. {pos.market.ticker}")
            print(f"   Platform: {pos.market.platform.value}")
            print(f"   Side: {pos.side.value.upper()}")
            print(f"   Contracts: {pos.contracts}")
            print(f"   Entry Price: ${pos.entry_price:.2f}")
            print(f"   Total Cost: ${pos.cost:.2f}")
            print(f"   Model Prob: {pos.model_prob:.1%}")
            print(f"   Edge: {pos.edge:.1%}")
            print(f"   Expected Profit: ${pos.expected_profit:.2f}")
            print(f"   Max Loss: ${pos.max_loss:.2f}")
            print(f"   Kelly Used: {pos.kelly_fraction_used:.1%}")

            total_cost += pos.cost
            total_ev += pos.expected_profit

        print("\n" + "-" * 70)
        print(f"TOTAL CAPITAL DEPLOYED: ${total_cost:.2f} ({total_cost/self.config.bankroll:.1%} of bankroll)")
        print(f"TOTAL EXPECTED PROFIT: ${total_ev:.2f}")
        print("=" * 70)

    def execute_recommendations(self) -> List[Dict]:
        """
        Execute all recommendations.

        Returns list of execution results.
        """
        results = []
        for position in self.recommendations:
            result = self.executor.execute(position)
            results.append(result)

            if result.get("status") == "SIMULATED":
                logger.info(f"SIMULATED: {position.market.ticker} {position.side.value} x{position.contracts}")
            else:
                logger.info(f"EXECUTED: {position.market.ticker} {position.side.value} x{position.contracts}")

        return results

    def run_monte_carlo_analysis(self) -> Dict:
        """
        Run Monte Carlo simulation on current strategy.

        Uses average edge from recommendations to simulate many outcomes.
        """
        if not self.recommendations:
            return {"error": "No recommendations to simulate"}

        # Calculate average parameters from recommendations
        avg_edge = np.mean([p.edge for p in self.recommendations])
        avg_price = np.mean([p.entry_price for p in self.recommendations])
        avg_prob = np.mean([p.model_prob for p in self.recommendations])

        # Odds for Monte Carlo
        odds = (1 - avg_price) / avg_price

        # Run simulation
        results = self.monte_carlo.simulate_strategy(
            prob_win=avg_prob,
            odds=odds,
            bet_fraction=self.config.kelly_fraction * ((avg_prob - avg_price) / (1 - avg_price)),
            n_bets=100  # Simulate 100 bets
        )

        return results


# =============================================================================
# DEMONSTRATION / EXAMPLE USAGE
# =============================================================================

def demonstrate_kelly_math():
    """
    Interactive demonstration of Kelly Criterion.

    Run this to build intuition about optimal bet sizing.
    """
    print("\n" + "=" * 70)
    print("KELLY CRITERION DEMONSTRATION")
    print("=" * 70)

    # Example 1: Fair coin, 2:1 odds
    print("\n--- Example 1: Favorable Coin Flip ---")
    print("You have a fair coin (50% heads)")
    print("But you get 2:1 odds (win $2 for every $1 bet)")
    print()

    prob_win = 0.5
    odds = 2.0  # Win $2 per $1

    kelly = (prob_win * odds - (1 - prob_win)) / odds
    print(f"Probability of winning: {prob_win:.0%}")
    print(f"Odds: {odds}:1")
    print(f"Full Kelly fraction: {kelly:.1%}")
    print()
    print(f"With $100 bankroll, optimal bet = ${100 * kelly:.2f}")

    # Example 2: Prediction market
    print("\n--- Example 2: Prediction Market ---")
    print("Market: 'Jobs report above 150K' trading at $0.40")
    print("Your model estimates 55% probability")
    print()

    market_price = 0.40
    your_prob = 0.55
    edge = your_prob - market_price
    odds = (1 - market_price) / market_price

    full_kelly = (your_prob - market_price) / (1 - market_price)

    print(f"Market price (YES): ${market_price:.2f}")
    print(f"Your probability: {your_prob:.0%}")
    print(f"Edge: {edge:.1%}")
    print(f"Odds: {odds:.2f}:1")
    print(f"Full Kelly: {full_kelly:.1%}")
    print(f"Half Kelly: {full_kelly * 0.5:.1%}")
    print(f"Quarter Kelly: {full_kelly * 0.25:.1%}")

    # Example 3: Why fractional Kelly?
    print("\n--- Example 3: Variance Comparison ---")
    config = TradingConfig(bankroll=100, simulation_runs=5000)
    mc = MonteCarloSimulator(config)

    print("Simulating 50 bets with 55% win rate, even odds...")
    print("Comparing Full Kelly vs Half Kelly vs Quarter Kelly")
    print()

    full_kelly_bet = 0.10  # 10% = full Kelly for 55% at even odds

    results_full = mc.simulate_strategy(0.55, 1.0, full_kelly_bet, 50)
    results_half = mc.simulate_strategy(0.55, 1.0, full_kelly_bet * 0.5, 50)
    results_quarter = mc.simulate_strategy(0.55, 1.0, full_kelly_bet * 0.25, 50)

    print(f"{'Metric':<25} {'Full Kelly':<15} {'Half Kelly':<15} {'Quarter Kelly':<15}")
    print("-" * 70)
    print(f"{'Mean Final Bankroll':<25} ${results_full['mean_final_bankroll']:<14.2f} ${results_half['mean_final_bankroll']:<14.2f} ${results_quarter['mean_final_bankroll']:<14.2f}")
    print(f"{'Median Final Bankroll':<25} ${results_full['median_final_bankroll']:<14.2f} ${results_half['median_final_bankroll']:<14.2f} ${results_quarter['median_final_bankroll']:<14.2f}")
    print(f"{'Std Dev':<25} ${results_full['std_final_bankroll']:<14.2f} ${results_half['std_final_bankroll']:<14.2f} ${results_quarter['std_final_bankroll']:<14.2f}")
    print(f"{'Ruin Probability':<25} {results_full['ruin_probability']:<14.1%} {results_half['ruin_probability']:<14.1%} {results_quarter['ruin_probability']:<14.1%}")
    print(f"{'5th Percentile':<25} ${results_full['percentile_5']:<14.2f} ${results_half['percentile_5']:<14.2f} ${results_quarter['percentile_5']:<14.2f}")
    print(f"{'95th Percentile':<25} ${results_full['percentile_95']:<14.2f} ${results_half['percentile_95']:<14.2f} ${results_quarter['percentile_95']:<14.2f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Half Kelly gives ~75% of the growth with ~50% of the variance.")
    print("For a $100 bankroll, Quarter Kelly is recommended until you verify your edge.")
    print("=" * 70)


def demonstrate_jobs_report_model():
    """
    Demonstrate probability modeling for Jobs Report.
    """
    print("\n" + "=" * 70)
    print("JOBS REPORT PROBABILITY MODEL")
    print("=" * 70)

    # Current consensus (update these before each report)
    consensus_mean = 150000  # 150K jobs expected
    historical_std = 85000   # Historical forecast error std dev

    model = JobsReportModel(consensus_mean, historical_std)

    print(f"\nConsensus forecast: {consensus_mean:,} jobs")
    print(f"Historical std dev: {historical_std:,} jobs")
    print()

    # Calculate probabilities for various thresholds
    thresholds = [50000, 100000, 125000, 150000, 175000, 200000, 250000]

    print(f"{'Threshold':<15} {'P(Above)':<15} {'P(Below)':<15}")
    print("-" * 45)

    for t in thresholds:
        prob_above = model.prob_above(t)
        print(f"{t:>12,}   {prob_above:>12.1%}   {1-prob_above:>12.1%}")

    # Range probabilities
    print("\nRange Probabilities:")
    ranges = [(100000, 150000), (150000, 200000), (100000, 200000)]
    for lower, upper in ranges:
        prob = model.prob_in_range(lower, upper)
        print(f"  P({lower:,} < Jobs < {upper:,}) = {prob:.1%}")


def run_example_backtest():
    """
    Run an example backtest with simulated historical data.

    In production, you'd use actual historical prediction/outcome pairs.
    """
    print("\n" + "=" * 70)
    print("BACKTEST EXAMPLE")
    print("=" * 70)

    config = TradingConfig(bankroll=100, kelly_fraction=0.25)
    kelly = KellyCalculator(config)
    backtester = BacktestEngine(config)

    # Simulate 20 historical predictions
    # In reality, these would come from your historical database
    np.random.seed(42)  # For reproducibility

    historical_data = []

    for i in range(20):
        # Create a mock market
        yes_price = np.random.uniform(0.30, 0.70)
        market = Market(
            ticker=f"TEST-{i}",
            title=f"Test Market {i}",
            platform=Platform.KALSHI,
            yes_price=yes_price,
            no_price=1 - yes_price + 0.02,  # 2% spread
            volume=50000
        )

        # Your model had some edge (on average)
        true_prob = yes_price + np.random.uniform(-0.05, 0.15)
        true_prob = max(0.1, min(0.9, true_prob))  # Clamp

        prediction = Prediction(
            market=market,
            model_prob_yes=true_prob
        )

        # Actual outcome (simulate based on true probability)
        actual_outcome = np.random.random() < true_prob

        historical_data.append((prediction, actual_outcome))

    # Run backtest
    results = backtester.run_backtest(historical_data, kelly)

    print(f"\nBacktest Results ({len(historical_data)} markets):")
    print("-" * 40)

    for key, value in results.items():
        if isinstance(value, float):
            if 'pct' in key or 'rate' in key or 'ratio' in key:
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: ${value:.2f}" if 'pnl' in key or 'equity' in key or 'win' in key or 'loss' in key else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nNote: This is simulated data. Real backtests require historical prices and outcomes.")


def main():
    """
    Main entry point demonstrating the trading system.
    """
    print("=" * 70)
    print("PREDICTION MARKET TRADING SYSTEM")
    print("=" * 70)
    print(f"Bankroll: ${config.bankroll}")
    print(f"Kelly Fraction: {config.kelly_fraction:.0%} (Quarter Kelly)")
    print(f"Min Edge: {config.min_ev:.1%}")
    print(f"Live Trading: {'ENABLED' if config.live_trading_enabled else 'DISABLED (simulation mode)'}")
    print()

    # Run demonstrations
    demonstrate_kelly_math()
    demonstrate_jobs_report_model()
    run_example_backtest()

    # Example: Create a real recommendation
    print("\n" + "=" * 70)
    print("EXAMPLE: Generating Trade Recommendation")
    print("=" * 70)

    trader = PredictionMarketTrader(config)

    # Example market (you'd fetch real data in production)
    example_market = Market(
        ticker="JOBS-FEB26-ABOVE150K",
        title="February Jobs Report Above 150,000",
        platform=Platform.KALSHI,
        yes_price=0.42,
        no_price=0.60,
        volume=75000,
        category="JOBS"
    )

    # Your model's prediction
    trader.add_prediction(
        market=example_market,
        model_prob_yes=0.52,  # You think 52% chance, market says 42%
        confidence=0.8
    )

    # Get recommendations
    recommendations = trader.get_recommendations()
    trader.print_recommendations()

    # Run Monte Carlo on the strategy
    print("\n--- Monte Carlo Risk Analysis ---")
    mc_results = trader.run_monte_carlo_analysis()
    if "error" not in mc_results:
        print(f"Mean outcome after 100 similar bets: ${mc_results['mean_final_bankroll']:.2f}")
        print(f"5th percentile (bad luck): ${mc_results['percentile_5']:.2f}")
        print(f"95th percentile (good luck): ${mc_results['percentile_95']:.2f}")
        print(f"Probability of ruin: {mc_results['ruin_probability']:.2%}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Update consensus values before each economic release")
    print("2. Fetch real market prices from Kalshi/Polymarket")
    print("3. Add predictions based on your research")
    print("4. Review recommendations carefully")
    print("5. When confident, set config.live_trading_enabled = True")
    print("=" * 70)


if __name__ == "__main__":
    main()
