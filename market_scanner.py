"""
Market Scanner & Arbitrage Detector
====================================
Integrates all prediction market platforms to:
1. Fetch live market prices
2. Detect arbitrage opportunities across platforms
3. Match similar events for cross-platform analysis
4. Store historical data for backtesting

ARBITRAGE FUNDAMENTALS (First Principles):
==========================================

WHAT IS ARBITRAGE?
------------------
Arbitrage is a RISK-FREE profit opportunity that exists when prices
across different markets are inconsistent.

BINARY MARKET ARBITRAGE:
------------------------
In a binary market (YES/NO), the probabilities MUST sum to 1.
If you can buy YES on Platform A and NO on Platform B for less
than $1 total, you're GUARANTEED profit regardless of outcome.

EXAMPLE:
--------
Event: "Bitcoin above $50K on March 1st"

Platform A (Kalshi):    YES = $0.45
Platform B (Polymarket): NO = $0.52  (equivalent to YES = $0.48)

If you could buy:
- YES on Kalshi at $0.45
- NO on Polymarket at $0.52
- Total cost: $0.97

Outcomes:
- If YES: Win $1 from Kalshi YES, lose $0.52 on Poly NO -> Net: $1 - $0.45 - $0.52 = $0.03
- If NO:  Lose $0.45 on Kalshi YES, win $1 from Poly NO -> Net: $1 - $0.52 - $0.45 = $0.03

GUARANTEED 3% profit! This is arbitrage.

WHY DOES ARBITRAGE EXIST?
-------------------------
1. Different liquidity pools (different traders on each platform)
2. Information asymmetry (news reaches platforms at different speeds)
3. Different fee structures affecting pricing
4. Market inefficiency during volatile periods

WHY IS IT RARE?
---------------
1. Sophisticated traders constantly scan for it
2. When found, they trade it away quickly
3. Fees often eat the small margins
4. Execution risk (prices move while you're trading)

MODEL CONFIDENCE (Mathematical Treatment):
==========================================

Your model's probability estimate has uncertainty. If you're 80% confident
in your model, you should shrink your edge estimate toward the market price.

BAYESIAN APPROACH:
------------------
adjusted_prob = confidence x model_prob + (1 - confidence) x market_prob

EXAMPLE:
--------
- Your model says 60% probability
- Market says 45%
- Your confidence in model: 70%

adjusted_prob = 0.70 x 0.60 + 0.30 x 0.45 = 0.42 + 0.135 = 0.555

Your effective edge is now 55.5% - 45% = 10.5%, not 15%.

This is CRUCIAL for position sizing:
- High confidence -> bet closer to full model probability
- Low confidence -> bet closer to market (smaller edge, smaller size)
"""

import os
import sys
import json
import time
import hashlib
import logging
import re
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path
from enum import Enum
from difflib import SequenceMatcher
from collections import deque
import sqlite3

import numpy as np
import pandas as pd


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    RATE LIMITING FUNDAMENTALS:
    ===========================
    APIs limit requests to prevent abuse and ensure fair access.
    Exceeding limits results in HTTP 429 (Too Many Requests) errors.

    KALSHI TIERS:
    -------------
    Basic:   20 reads/sec, 10 writes/sec
    Advanced: Higher limits (contact Kalshi)
    Premier:  Highest limits (contact Kalshi)

    TOKEN BUCKET ALGORITHM:
    -----------------------
    - Bucket holds up to 'capacity' tokens
    - Tokens regenerate at 'rate' per second
    - Each request consumes 1 token
    - If no tokens available, wait until one regenerates

    This smooths out bursts while allowing sustained throughput.
    """

    def __init__(self, rate: float, capacity: int = None):
        """
        Args:
            rate: Requests per second allowed
            capacity: Max burst size (defaults to rate)
        """
        self.rate = rate
        self.capacity = capacity or int(rate)
        self.tokens = self.capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire a token (permission to make a request).

        Blocks until a token is available or timeout.

        Args:
            timeout: Max seconds to wait

        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self.lock:
                # Regenerate tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            # Check timeout
            if time.time() - start_time > timeout:
                return False

            # Wait a bit before retrying
            time.sleep(0.05)

    def wait(self):
        """Block until a token is available (no timeout)."""
        self.acquire(timeout=float('inf'))

# Import from our existing modules
from app import (
    TradingConfig, Platform, BetSide, Market, Prediction, Position,
    KellyCalculator, PredictionMarketTrader, MonteCarloSimulator
)

# Import API clients
from api_tester import (
    KalshiAuthenticator, PolymarketClient, DKPredictionsClient
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# HISTORICAL DATA STORAGE
# =============================================================================

class HistoricalDataStore:
    """
    SQLite-based storage for historical market data.

    WHY STORE HISTORICAL DATA?
    ==========================
    1. Backtesting requires knowing past prices AND outcomes
    2. Patterns in price movements can inform edge
    3. Track your prediction accuracy over time
    4. Identify which market types you're good/bad at

    SCHEMA DESIGN:
    --------------
    markets: Core market info (ticker, title, platform, category)
    prices: Time-series of prices (bid, ask, volume at each timestamp)
    outcomes: Final resolution (YES or NO won)
    predictions: Your model's predictions for analysis
    """

    def __init__(self, db_path: str = "data/market_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Markets table - one row per unique market
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                title TEXT,
                platform TEXT,
                category TEXT,
                expiry TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Prices table - time series of market prices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                yes_price REAL,
                no_price REAL,
                yes_bid REAL,
                yes_ask REAL,
                volume REAL,
                open_interest INTEGER,
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)

        # Outcomes table - final resolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER UNIQUE,
                outcome TEXT,  -- 'YES' or 'NO'
                settled_at TIMESTAMP,
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)

        # Predictions table - your model's predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_prob_yes REAL,
                confidence REAL,
                model_name TEXT,
                notes TEXT,
                FOREIGN KEY (market_id) REFERENCES markets(id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_market ON prices(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_time ON prices(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markets_platform ON markets(platform)")

        conn.commit()
        conn.close()

    def store_market(self, market: Market) -> int:
        """Store or update a market, return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO markets (ticker, title, platform, category, expiry)
            VALUES (?, ?, ?, ?, ?)
        """, (market.ticker, market.title, market.platform.value,
              market.category, market.expiry))

        cursor.execute("SELECT id FROM markets WHERE ticker = ?", (market.ticker,))
        market_id = cursor.fetchone()[0]

        conn.commit()
        conn.close()
        return market_id

    def store_price(self, market: Market):
        """Store a price snapshot for a market."""
        market_id = self.store_market(market)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prices (market_id, yes_price, no_price, yes_bid, yes_ask, volume, open_interest)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (market_id, market.yes_price, market.no_price,
              market.yes_price, market.yes_ask, market.volume, market.open_interest))

        conn.commit()
        conn.close()

    def store_prediction(self, market: Market, model_prob_yes: float,
                         confidence: float, model_name: str = "manual"):
        """Store a prediction for analysis."""
        market_id = self.store_market(market)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (market_id, model_prob_yes, confidence, model_name)
            VALUES (?, ?, ?, ?)
        """, (market_id, model_prob_yes, confidence, model_name))

        conn.commit()
        conn.close()

    def store_outcome(self, ticker: str, outcome: str):
        """Store the final outcome of a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        result = cursor.fetchone()
        if result:
            market_id = result[0]
            cursor.execute("""
                INSERT OR REPLACE INTO outcomes (market_id, outcome, settled_at)
                VALUES (?, ?, ?)
            """, (market_id, outcome, datetime.now()))

        conn.commit()
        conn.close()

    def get_historical_prices(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get price history for a market."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT p.timestamp, p.yes_price, p.no_price, p.volume
            FROM prices p
            JOIN markets m ON p.market_id = m.id
            WHERE m.ticker = ?
            AND p.timestamp > datetime('now', ?)
            ORDER BY p.timestamp
        """

        df = pd.read_sql_query(query, conn, params=(ticker, f'-{days} days'))
        conn.close()
        return df

    def get_prediction_accuracy(self, model_name: str = None) -> Dict:
        """
        Calculate prediction accuracy for backtesting analysis.

        Returns metrics on how well your predictions performed.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                p.model_prob_yes,
                p.confidence,
                o.outcome,
                pr.yes_price as market_price
            FROM predictions p
            JOIN outcomes o ON p.market_id = o.market_id
            JOIN prices pr ON p.market_id = pr.market_id
            WHERE p.model_name = COALESCE(?, p.model_name)
        """

        cursor.execute(query, (model_name,))
        results = cursor.fetchall()
        conn.close()

        if not results:
            return {"error": "No resolved predictions found"}

        # Calculate metrics
        correct = 0
        total = 0
        brier_scores = []

        for model_prob, confidence, outcome, market_price in results:
            actual = 1 if outcome == 'YES' else 0

            # Did we predict correctly?
            predicted = 1 if model_prob > 0.5 else 0
            if predicted == actual:
                correct += 1
            total += 1

            # Brier score (lower is better, 0 = perfect)
            brier = (model_prob - actual) ** 2
            brier_scores.append(brier)

        return {
            "total_predictions": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "brier_score": np.mean(brier_scores) if brier_scores else None
        }


# =============================================================================
# LIVE MARKET DATA FETCHER
# =============================================================================

class LiveMarketFetcher:
    """
    Fetches live market data from all platforms.

    INTEGRATION ARCHITECTURE:
    =========================
    Each platform has different:
    - Authentication methods
    - API structures
    - Rate limits
    - Data formats

    This class normalizes everything into our Market dataclass.

    RATE LIMITS (Kalshi Basic Tier):
    --------------------------------
    - Read requests: 20/second
    - Write requests: 10/second

    We use conservative limits (80% of max) to avoid hitting limits.
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.data_store = HistoricalDataStore()

        # Rate limiters for Kalshi (Basic tier: 20 read/sec, 10 write/sec)
        # Using 80% of limit to be safe
        self.kalshi_read_limiter = RateLimiter(rate=16, capacity=16)  # 80% of 20
        self.kalshi_write_limiter = RateLimiter(rate=8, capacity=8)   # 80% of 10

        # Initialize platform clients
        self._init_kalshi()
        self._init_polymarket()
        self._init_dkpredictions()

    def _init_kalshi(self):
        """Initialize Kalshi API client."""
        try:
            self.kalshi = KalshiAuthenticator()
            self.kalshi_available = True
            logger.info("Kalshi API initialized")
        except Exception as e:
            logger.warning(f"Kalshi API not available: {e}")
            self.kalshi = None
            self.kalshi_available = False

    def _init_polymarket(self):
        """Initialize Polymarket client."""
        try:
            self.polymarket = PolymarketClient()
            self.polymarket_available = True
            logger.info("Polymarket API initialized")
        except Exception as e:
            logger.warning(f"Polymarket API not available: {e}")
            self.polymarket = None
            self.polymarket_available = False

    def _init_dkpredictions(self):
        """Initialize DKPredictions scraper."""
        try:
            self.dkpredictions = DKPredictionsClient(headless=True)
            self.dkpredictions_available = True
            logger.info("DKPredictions scraper initialized")
        except Exception as e:
            logger.warning(f"DKPredictions not available: {e}")
            self.dkpredictions = None
            self.dkpredictions_available = False

    def fetch_kalshi_markets(self, category: str = None, limit: int = 100) -> List[Market]:
        """
        Fetch markets from Kalshi.

        Categories: ECON (economics), POLITICS, SPORTS, ENTERTAINMENT, etc.

        Rate limited to respect Kalshi API limits (Basic tier: 20 reads/sec).
        """
        if not self.kalshi_available:
            logger.warning("Kalshi not available")
            return []

        markets = []

        try:
            # Wait for rate limiter
            if not self.kalshi_read_limiter.acquire(timeout=5):
                logger.warning("Kalshi rate limit timeout")
                return []

            # Build request path
            path = f"/trade-api/v2/markets?limit={limit}&status=open"  # Only open markets
            if category:
                path += f"&series_ticker={category}"

            headers = self.kalshi.get_headers("GET", path)
            url = f"{self.kalshi.base_url}{path}"

            import requests
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()
                for m in data.get("markets", []):
                    market = Market(
                        ticker=m.get("ticker", ""),
                        title=m.get("title", ""),
                        platform=Platform.KALSHI,
                        yes_price=m.get("yes_bid", 0) / 100,  # Convert cents to dollars
                        no_price=m.get("no_bid", 0) / 100,
                        yes_ask=m.get("yes_ask", 0) / 100,
                        volume=m.get("volume", 0),
                        open_interest=m.get("open_interest", 0),
                        category=m.get("series_ticker", ""),
                        expiry=datetime.fromisoformat(m["expiration_time"].replace("Z", "+00:00"))
                               if m.get("expiration_time") else None
                    )
                    markets.append(market)

                    # Store for historical tracking
                    self.data_store.store_price(market)

                logger.info(f"Fetched {len(markets)} markets from Kalshi")
            else:
                logger.error(f"Kalshi API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")

        return markets

    def fetch_polymarket_markets(self, limit: int = 100, active: bool = True) -> List[Market]:
        """
        Fetch markets from Polymarket.

        Polymarket uses the CLOB (Central Limit Order Book) API.
        """
        if not self.polymarket_available:
            logger.warning("Polymarket not available")
            return []

        markets = []

        try:
            result = self.polymarket.get_markets(limit=limit, active=active)

            if result.get("success") and result.get("data"):
                for m in result["data"]:
                    # Polymarket returns outcome prices as JSON strings
                    # Need to parse them: '["0.45", "0.55"]' -> [0.45, 0.55]
                    outcomes_raw = m.get("outcomes", '["Yes", "No"]')
                    prices_raw = m.get("outcomePrices", '["0.5", "0.5"]')

                    # Parse JSON strings if needed
                    if isinstance(outcomes_raw, str):
                        try:
                            outcomes = json.loads(outcomes_raw)
                        except:
                            outcomes = ["Yes", "No"]
                    else:
                        outcomes = outcomes_raw

                    if isinstance(prices_raw, str):
                        try:
                            prices = json.loads(prices_raw)
                        except:
                            prices = ["0.5", "0.5"]
                    else:
                        prices = prices_raw

                    # Convert to floats
                    try:
                        yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                        no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
                    except (ValueError, TypeError):
                        yes_price = 0.5
                        no_price = 0.5

                    # Skip markets with invalid prices (inactive/settled)
                    # Valid prices should be between 0.01 and 0.99
                    if yes_price <= 0.01 or yes_price >= 0.99:
                        continue
                    if no_price <= 0.01 or no_price >= 0.99:
                        continue

                    market = Market(
                        ticker=m.get("id", m.get("slug", "")),
                        title=m.get("question", m.get("title", "")),
                        platform=Platform.POLYMARKET,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(m.get("volume", 0)),
                        open_interest=int(float(m.get("liquidity", 0))),
                        category=m.get("category", ""),
                        expiry=datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
                               if m.get("endDate") else None
                    )
                    markets.append(market)
                    self.data_store.store_price(market)

                logger.info(f"Fetched {len(markets)} markets from Polymarket")

        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")

        return markets

    def fetch_dkpredictions_markets(self, sport: str = None) -> List[Market]:
        """
        Fetch markets from DKPredictions via web scraping.

        NOTE: This uses Selenium and is slower than API calls.
        """
        if not self.dkpredictions_available:
            logger.warning("DKPredictions not available")
            return []

        markets = []

        try:
            result = self.dkpredictions.scrape_predictions(sport=sport)

            if result.get("success") and result.get("predictions"):
                for p in result["predictions"]:
                    # Convert DK prediction format to our Market format
                    # DK uses player props (over/under lines), not binary markets
                    player = p.get("player_name", "Unknown")
                    stat = p.get("stat_type", "Unknown")
                    line = p.get("line", 0)

                    # Create pseudo-ticker
                    ticker = f"DK-{player.replace(' ', '-')}-{stat}-{line}"

                    # DK doesn't give us prices directly, so we estimate
                    # Most player props are priced around 50/50 with vig
                    over_mult = p.get("over_multiplier") or 1.0
                    under_mult = p.get("under_multiplier") or 1.0

                    # Convert multipliers to implied probability
                    # multiplier of 2.0 = 50% implied prob (break even)
                    yes_price = 1 / over_mult if over_mult > 0 else 0.5
                    no_price = 1 / under_mult if under_mult > 0 else 0.5

                    market = Market(
                        ticker=ticker,
                        title=f"{player} {stat} Over {line}",
                        platform=Platform.DKPREDICTIONS,
                        yes_price=min(0.95, max(0.05, yes_price)),  # Clamp to valid range
                        no_price=min(0.95, max(0.05, no_price)),
                        category=p.get("sport", "SPORTS")
                    )
                    markets.append(market)

                logger.info(f"Fetched {len(markets)} markets from DKPredictions")

        except Exception as e:
            logger.error(f"Error fetching DKPredictions markets: {e}")

        finally:
            # Clean up Selenium resources
            if self.dkpredictions:
                self.dkpredictions.close()
                self._init_dkpredictions()  # Reinit for next use

        return markets

    def fetch_all_markets(self) -> Dict[Platform, List[Market]]:
        """Fetch markets from all available platforms."""
        all_markets = {}

        print("Fetching markets from all platforms...")

        print("  Kalshi...")
        all_markets[Platform.KALSHI] = self.fetch_kalshi_markets()

        print("  Polymarket...")
        all_markets[Platform.POLYMARKET] = self.fetch_polymarket_markets()

        # DKPredictions is slow (Selenium), only fetch if needed
        # print("  DKPredictions...")
        # all_markets[Platform.DKPREDICTIONS] = self.fetch_dkpredictions_markets()
        all_markets[Platform.DKPREDICTIONS] = []  # Skip by default

        total = sum(len(m) for m in all_markets.values())
        print(f"Total markets fetched: {total}")

        return all_markets

    def close(self):
        """Clean up resources."""
        if self.dkpredictions:
            self.dkpredictions.close()


# =============================================================================
# ARBITRAGE DETECTOR
# =============================================================================

@dataclass
class ArbitrageOpportunity:
    """
    Represents a cross-platform arbitrage opportunity.

    ANATOMY OF AN ARBITRAGE:
    ========================
    - Same event on different platforms
    - Combined cost < $1.00
    - Guaranteed profit = $1.00 - combined cost
    """
    event_description: str
    platform1: Platform
    platform2: Platform
    market1: Market  # Buy YES here
    market2: Market  # Buy NO here
    yes_cost: float  # Price to buy YES on platform1
    no_cost: float   # Price to buy NO on platform2
    total_cost: float
    guaranteed_profit: float
    profit_pct: float
    confidence_match: float  # How confident we are these are the same event


class ArbitrageDetector:
    """
    Detects arbitrage opportunities across prediction market platforms.

    ARBITRAGE DETECTION ALGORITHM:
    ==============================

    1. SAME-PLATFORM ARBITRAGE (rare, usually fees eliminate it):
       - If YES_price + NO_price < 1.00 on same platform
       - Profit = 1.00 - (YES + NO)

    2. CROSS-PLATFORM ARBITRAGE:
       - Match similar events across platforms
       - Compare: Platform_A_YES + Platform_B_NO
       - If < 1.00, arbitrage exists

    EVENT MATCHING CHALLENGES:
    --------------------------
    Different platforms use different:
    - Naming conventions ("BTC > $50K" vs "Bitcoin above 50000")
    - Threshold values
    - Expiry times

    We use fuzzy string matching + keyword extraction.
    """

    def __init__(self, min_profit_pct: float = 0.01):
        """
        Args:
            min_profit_pct: Minimum profit percentage to report (default 1%)
        """
        self.min_profit_pct = min_profit_pct

    def detect_same_platform_arbitrage(self, markets: List[Market]) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage within a single platform.

        This is RARE because platforms typically ensure YES + NO >= 1.00
        But can occur during high volatility or technical issues.
        """
        opportunities = []

        for market in markets:
            # Skip invalid/settled markets (prices at 0 or 1 are settled)
            if market.yes_price <= 0.01 or market.yes_price >= 0.99:
                continue
            if market.no_price <= 0.01 or market.no_price >= 0.99:
                continue

            total = market.yes_price + market.no_price

            # Check if there's room for profit after accounting for typical fees
            # Also require total > 0.10 to filter out broken data
            if total > 0.10 and total < (1.0 - self.min_profit_pct):
                profit = 1.0 - total
                opp = ArbitrageOpportunity(
                    event_description=market.title,
                    platform1=market.platform,
                    platform2=market.platform,
                    market1=market,
                    market2=market,
                    yes_cost=market.yes_price,
                    no_cost=market.no_price,
                    total_cost=total,
                    guaranteed_profit=profit,
                    profit_pct=profit / total,
                    confidence_match=1.0  # Same market, 100% confidence
                )
                opportunities.append(opp)
                logger.info(f"Same-platform arb found: {market.ticker} profit={profit:.2%}")

        return opportunities

    def _normalize_event_text(self, text: str) -> str:
        """
        Normalize event text for comparison.

        Converts: "Will Bitcoin exceed $50,000 by March?"
        To: "bitcoin exceed 50000 march"
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation except hyphens
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Normalize numbers (remove commas, $, etc.)
        text = re.sub(r'[\$,]', '', text)

        # Remove common filler words
        stopwords = {'will', 'the', 'be', 'by', 'on', 'in', 'at', 'to', 'a', 'an', 'is', 'are'}
        words = [w for w in text.split() if w not in stopwords]

        return ' '.join(words)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from event text."""
        normalized = self._normalize_event_text(text)

        # Look for specific patterns
        keywords = set()

        # Numbers (thresholds, prices, dates)
        numbers = re.findall(r'\d+', normalized)
        keywords.update(numbers)

        # Key terms
        key_terms = ['bitcoin', 'btc', 'ethereum', 'eth', 'jobs', 'nfp', 'cpi',
                     'inflation', 'fed', 'rate', 'trump', 'biden', 'election',
                     'gdp', 'unemployment', 'above', 'below', 'over', 'under']

        for term in key_terms:
            if term in normalized:
                keywords.add(term)

        return keywords

    def _calculate_match_score(self, market1: Market, market2: Market) -> float:
        """
        Calculate how likely two markets refer to the same event.

        Returns value 0-1 where 1 = definitely same event.

        MATCHING HEURISTICS:
        --------------------
        1. String similarity of titles
        2. Keyword overlap
        3. Category match
        4. Expiry date proximity
        """
        # Normalize titles
        title1 = self._normalize_event_text(market1.title)
        title2 = self._normalize_event_text(market2.title)

        # String similarity
        string_sim = SequenceMatcher(None, title1, title2).ratio()

        # Keyword overlap
        kw1 = self._extract_keywords(market1.title)
        kw2 = self._extract_keywords(market2.title)

        if kw1 and kw2:
            keyword_sim = len(kw1 & kw2) / len(kw1 | kw2)
        else:
            keyword_sim = 0

        # Category match
        cat_match = 1.0 if market1.category.lower() == market2.category.lower() else 0.5

        # Expiry proximity (if available)
        expiry_sim = 1.0
        if market1.expiry and market2.expiry:
            delta = abs((market1.expiry - market2.expiry).total_seconds())
            # Within 24 hours = 1.0, decreases with distance
            expiry_sim = max(0, 1 - delta / (7 * 24 * 3600))  # 0 if > 1 week apart

        # Weighted combination
        score = (
            0.4 * string_sim +
            0.3 * keyword_sim +
            0.15 * cat_match +
            0.15 * expiry_sim
        )

        return score

    def detect_cross_platform_arbitrage(
        self,
        markets_by_platform: Dict[Platform, List[Market]],
        min_match_confidence: float = 0.6
    ) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across different platforms.

        Algorithm:
        1. For each pair of platforms
        2. For each market on platform A
        3. Find best matching market on platform B
        4. Check if YES_A + NO_B < 1.00

        Args:
            markets_by_platform: Dict mapping platform to list of markets
            min_match_confidence: Minimum similarity score to consider a match
        """
        opportunities = []
        platforms = list(markets_by_platform.keys())

        # Compare each pair of platforms
        for i, platform_a in enumerate(platforms):
            for platform_b in platforms[i+1:]:
                markets_a = markets_by_platform.get(platform_a, [])
                markets_b = markets_by_platform.get(platform_b, [])

                if not markets_a or not markets_b:
                    continue

                logger.info(f"Comparing {platform_a.value} ({len(markets_a)}) vs {platform_b.value} ({len(markets_b)})")

                # For each market on platform A, find matches on B
                for market_a in markets_a:
                    for market_b in markets_b:
                        match_score = self._calculate_match_score(market_a, market_b)

                        if match_score < min_match_confidence:
                            continue

                        # Check arbitrage both ways:
                        # 1. Buy YES on A, NO on B
                        cost_1 = market_a.yes_price + market_b.no_price
                        # 2. Buy NO on A, YES on B
                        cost_2 = market_a.no_price + market_b.yes_price

                        # Check option 1
                        if cost_1 < (1.0 - self.min_profit_pct):
                            profit = 1.0 - cost_1
                            opp = ArbitrageOpportunity(
                                event_description=f"{market_a.title} / {market_b.title}",
                                platform1=platform_a,
                                platform2=platform_b,
                                market1=market_a,
                                market2=market_b,
                                yes_cost=market_a.yes_price,
                                no_cost=market_b.no_price,
                                total_cost=cost_1,
                                guaranteed_profit=profit,
                                profit_pct=profit / cost_1,
                                confidence_match=match_score
                            )
                            opportunities.append(opp)

                        # Check option 2
                        if cost_2 < (1.0 - self.min_profit_pct):
                            profit = 1.0 - cost_2
                            opp = ArbitrageOpportunity(
                                event_description=f"{market_a.title} / {market_b.title}",
                                platform1=platform_b,
                                platform2=platform_a,
                                market1=market_b,
                                market2=market_a,
                                yes_cost=market_b.yes_price,
                                no_cost=market_a.no_price,
                                total_cost=cost_2,
                                guaranteed_profit=profit,
                                profit_pct=profit / cost_2,
                                confidence_match=match_score
                            )
                            opportunities.append(opp)

        # Sort by profit percentage (highest first)
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

        return opportunities

    def detect_all_arbitrage(
        self,
        markets_by_platform: Dict[Platform, List[Market]]
    ) -> List[ArbitrageOpportunity]:
        """Find all arbitrage opportunities."""
        all_opps = []

        # Same-platform arbitrage
        for platform, markets in markets_by_platform.items():
            opps = self.detect_same_platform_arbitrage(markets)
            all_opps.extend(opps)

        # Cross-platform arbitrage
        cross_opps = self.detect_cross_platform_arbitrage(markets_by_platform)
        all_opps.extend(cross_opps)

        return all_opps


# =============================================================================
# MODEL CONFIDENCE ADJUSTMENT
# =============================================================================

class ConfidenceAdjustedModel:
    """
    Adjusts model probabilities based on confidence level.

    MATHEMATICAL FOUNDATION:
    ========================

    Your model produces P_model, but you're not 100% certain your model is correct.
    The market price P_market represents the "wisdom of crowds."

    BAYESIAN SHRINKAGE:
    -------------------
    We blend your model with the market based on confidence:

    P_adjusted = confidence x P_model + (1 - confidence) x P_market

    WHY THIS MAKES SENSE:
    ---------------------
    - If confidence = 1.0: Use your model fully
    - If confidence = 0.0: Just follow the market (no edge)
    - If confidence = 0.5: Average of model and market

    EFFECT ON KELLY SIZING:
    -----------------------
    Lower confidence -> smaller adjusted edge -> smaller position size

    This AUTOMATICALLY implements prudent risk management:
    - Uncertain predictions get smaller bets
    - Confident predictions get larger bets

    CALIBRATION:
    ------------
    Your confidence should be CALIBRATED:
    - If you say 80% confidence, you should be right 80% of the time
    - Track your predictions to verify calibration
    - Most people are OVERCONFIDENT, so start conservative

    EXAMPLE:
    --------
    Model: 65% probability
    Market: 50% probability
    Confidence: 70%

    Adjusted = 0.70 x 0.65 + 0.30 x 0.50 = 0.455 + 0.15 = 0.605 (60.5%)

    Edge shrinks from 15% to 10.5%, position size decreases proportionally.
    """

    @staticmethod
    def adjust_probability(
        model_prob: float,
        market_prob: float,
        confidence: float
    ) -> float:
        """
        Adjust model probability based on confidence.

        Args:
            model_prob: Your model's probability estimate (0-1)
            market_prob: Market's implied probability (0-1)
            confidence: Your confidence in the model (0-1)

        Returns:
            Adjusted probability
        """
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        adjusted = confidence * model_prob + (1 - confidence) * market_prob
        return adjusted

    @staticmethod
    def calculate_confidence_adjusted_kelly(
        model_prob: float,
        market_price: float,
        confidence: float,
        kelly_fraction: float = 0.25
    ) -> Tuple[float, float, float]:
        """
        Calculate Kelly fraction with confidence adjustment.

        Returns:
            (adjusted_prob, adjusted_edge, kelly_bet_size)

        FORMULA DERIVATION:
        -------------------
        1. Adjust probability: P_adj = conf x P_model + (1-conf) x P_market
        2. Calculate edge: edge = P_adj - P_market
        3. Calculate Kelly: f = edge / (1 - P_market)
        4. Apply fractional Kelly: f_final = f x kelly_fraction
        """
        # Adjust probability
        adjusted_prob = ConfidenceAdjustedModel.adjust_probability(
            model_prob, market_price, confidence
        )

        # Calculate adjusted edge
        adjusted_edge = adjusted_prob - market_price

        # Kelly calculation
        if adjusted_edge <= 0:
            return adjusted_prob, adjusted_edge, 0.0

        full_kelly = adjusted_edge / (1 - market_price)
        adjusted_kelly = full_kelly * kelly_fraction

        return adjusted_prob, adjusted_edge, adjusted_kelly


# =============================================================================
# INTEGRATED SCANNER
# =============================================================================

class IntegratedScanner:
    """
    Main class integrating all components.

    WORKFLOW:
    =========
    1. Fetch live markets from all platforms
    2. Detect arbitrage opportunities
    3. For non-arbitrage: Apply your model + confidence
    4. Calculate Kelly-optimal positions
    5. Output recommendations

    USAGE:
    ------
    scanner = IntegratedScanner()
    scanner.scan_all()
    scanner.print_arbitrage_opportunities()
    scanner.print_recommendations()
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.fetcher = LiveMarketFetcher(self.config)
        self.arbitrage_detector = ArbitrageDetector(min_profit_pct=0.01)
        self.trader = PredictionMarketTrader(self.config)
        self.data_store = HistoricalDataStore()

        self.markets_by_platform: Dict[Platform, List[Market]] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []

    def scan_all(self, include_dk: bool = False):
        """
        Scan all platforms for markets and opportunities.

        Args:
            include_dk: Include DKPredictions (slow, requires Selenium)
        """
        print("\n" + "=" * 70)
        print("SCANNING ALL PREDICTION MARKETS")
        print("=" * 70)

        # Fetch markets
        self.markets_by_platform = self.fetcher.fetch_all_markets()

        # Optionally fetch DK
        if include_dk:
            print("  DKPredictions (this may take a while)...")
            self.markets_by_platform[Platform.DKPREDICTIONS] = \
                self.fetcher.fetch_dkpredictions_markets()

        # Detect arbitrage
        print("\nScanning for arbitrage opportunities...")
        self.arbitrage_opportunities = self.arbitrage_detector.detect_all_arbitrage(
            self.markets_by_platform
        )

        print(f"Found {len(self.arbitrage_opportunities)} potential arbitrage opportunities")

    def add_prediction_with_confidence(
        self,
        market: Market,
        model_prob_yes: float,
        confidence: float,
        store_prediction: bool = True
    ):
        """
        Add a prediction with confidence adjustment.

        This is the RECOMMENDED way to add predictions.
        """
        # Calculate adjusted probability
        adjusted_prob, adjusted_edge, kelly_size = \
            ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
                model_prob_yes,
                market.yes_price,
                confidence,
                self.config.kelly_fraction
            )

        print(f"\nPrediction for: {market.ticker}")
        print(f"  Model probability: {model_prob_yes:.1%}")
        print(f"  Market price: {market.yes_price:.1%}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Adjusted probability: {adjusted_prob:.1%}")
        print(f"  Adjusted edge: {adjusted_edge:.1%}")
        print(f"  Kelly bet size: {kelly_size:.1%}")

        # Add to trader with adjusted probability
        self.trader.add_prediction(market, adjusted_prob, confidence)

        # Store for tracking
        if store_prediction:
            self.data_store.store_prediction(
                market, model_prob_yes, confidence, "manual"
            )

    def print_arbitrage_opportunities(self):
        """Display arbitrage opportunities."""
        if not self.arbitrage_opportunities:
            print("\nNo arbitrage opportunities found.")
            return

        print("\n" + "=" * 70)
        print("ARBITRAGE OPPORTUNITIES")
        print("=" * 70)

        for i, opp in enumerate(self.arbitrage_opportunities[:10], 1):  # Top 10
            print(f"\n{i}. {opp.event_description[:60]}...")
            print(f"   Buy YES on {opp.platform1.value}: ${opp.yes_cost:.2f}")
            print(f"   Buy NO on {opp.platform2.value}: ${opp.no_cost:.2f}")
            print(f"   Total Cost: ${opp.total_cost:.2f}")
            print(f"   GUARANTEED PROFIT: ${opp.guaranteed_profit:.2f} ({opp.profit_pct:.1%})")
            print(f"   Match Confidence: {opp.confidence_match:.0%}")

        if len(self.arbitrage_opportunities) > 10:
            print(f"\n... and {len(self.arbitrage_opportunities) - 10} more opportunities")

    def print_market_summary(self):
        """Display summary of fetched markets."""
        print("\n" + "=" * 70)
        print("MARKET SUMMARY")
        print("=" * 70)

        for platform, markets in self.markets_by_platform.items():
            print(f"\n{platform.value.upper()}: {len(markets)} markets")

            if markets:
                # Show top 5 by volume
                sorted_markets = sorted(markets, key=lambda m: m.volume, reverse=True)[:5]
                for m in sorted_markets:
                    print(f"  - {m.ticker}: YES=${m.yes_price:.2f}, Vol=${m.volume:,.0f}")

    def get_recommendations(self) -> List[Position]:
        """Get position recommendations."""
        return self.trader.get_recommendations()

    def print_recommendations(self):
        """Display position recommendations."""
        self.trader.print_recommendations()

    def close(self):
        """Clean up resources."""
        self.fetcher.close()


# =============================================================================
# MATHEMATICAL DEEP DIVE
# =============================================================================

def explain_kelly_criterion():
    """
    In-depth explanation of Kelly Criterion mathematics.
    """
    print("""
    ======================================================================
                        KELLY CRITERION - DEEP DIVE
    ======================================================================

    THE PROBLEM:
    ============
    You have an edge in a repeated betting game. How much should you bet
    each time to maximize long-term wealth?

    INTUITION:
    ==========
    - Bet too little -> Leave money on the table, slow growth
    - Bet too much -> Risk of ruin, volatile path
    - Optimal bet -> Maximum long-term growth rate

    THE SETUP:
    ==========
    - You have wealth W
    - You bet fraction f of your wealth
    - Probability of winning: p
    - Odds: b (you win b dollars for every 1 dollar bet)

    AFTER ONE BET:
    ==============
    If you WIN:  W_new = W x (1 + f x b)    [wealth increases by f x b]
    If you LOSE: W_new = W x (1 - f)        [wealth decreases by f]

    EXPECTED GROWTH:
    ================
    We want to maximize E[log(W_new/W)] = expected LOG growth

    G(f) = p x log(1 + fxb) + (1-p) x log(1 - f)

    WHY LOGARITHM?
    ==============
    1. Log turns multiplication into addition:
       log(W1 x W2 x ... x Wn) = log(W1) + log(W2) + ... + log(Wn)

    2. This matches how compound growth works:
       After n bets, wealth = W0 x (growth1) x (growth2) x ... x (growth_n)

    3. Maximizing E[log(W)] is equivalent to maximizing the GEOMETRIC mean
       of outcomes, which is the relevant measure for compound growth.

    4. If you maximize E[W] instead (arithmetic mean), you can go bankrupt!
       Example: Bet everything each time with 60% win rate. Eventually you
       lose once and have $0.

    FINDING THE OPTIMAL f:
    ======================
    Take derivative of G(f) and set to zero:

    dG/df = pxb/(1+fxb) - (1-p)/(1-f) = 0

    Solve for f:
    pxbx(1-f) = (1-p)x(1+fxb)
    pxb - pxbxf = (1-p) + (1-p)xfxb
    pxb - (1-p) = pxbxf + (1-p)xfxb
    pxb - (1-p) = fxbx[p + (1-p)]
    pxb - (1-p) = fxb

    f* = (pxb - (1-p)) / b = (pxb - q) / b

    OR equivalently: f* = p - q/b = p - (1-p)/b

    FOR BINARY MARKETS:
    ===================
    In prediction markets, if you buy YES at price c:
    - You pay c dollars
    - If YES wins, you get $1 (profit = 1-c)
    - Odds b = (1-c)/c

    Substituting into Kelly:
    f* = (p - c) / (1 - c)

    This is the fraction of bankroll to bet on YES.

    EXAMPLE CALCULATION:
    ====================
    Market: YES at $0.40
    Your probability: 60%

    f* = (0.60 - 0.40) / (1 - 0.40) = 0.20 / 0.60 = 0.333 (33.3%)

    With $100 bankroll, bet $33.33 on YES.

    FRACTIONAL KELLY:
    =================
    Full Kelly has HIGH VARIANCE. The long-term path can have 50%+ drawdowns.

    Fractional Kelly trades growth for stability:
    f_actual = k x f*    where k < 1

    Properties of fractional Kelly:
    - Half Kelly (k=0.5): ~75% of growth rate, ~50% of variance
    - Quarter Kelly (k=0.25): ~50% of growth rate, ~25% of variance

    The math: Growth rate G(f) is approximately quadratic near f*
    G(kxf*) ~ kx(2-k) x G(f*)

    So half Kelly: 0.5 x 1.5 = 0.75 = 75% of optimal growth

    WHEN TO USE WHAT:
    =================
    Full Kelly: When you're CERTAIN of your edge and can handle volatility
    Half Kelly: Good balance for experienced traders
    Quarter Kelly: Recommended for beginners or uncertain edges

    CRITICAL INSIGHT:
    =================
    Kelly assumes you KNOW the true probability p. In reality, you only
    ESTIMATE p. Uncertainty in p should make you bet LESS than Kelly.

    This is why confidence adjustment is so important:
    - Use your model probability only to the extent you trust it
    - Blend with market probability for robustness
    """)


def explain_expected_value():
    """
    In-depth explanation of Expected Value mathematics.
    """
    print("""
    ======================================================================
                        EXPECTED VALUE - DEEP DIVE
    ======================================================================

    DEFINITION:
    ===========
    Expected Value (EV) is the probability-weighted average of all possible
    outcomes. It answers: "On average, what will I gain or lose?"

    FORMULA:
    ========
    EV = Sum (probability_i x outcome_i) for all outcomes i

    For a binary bet:
    EV = P(win) x profit_if_win + P(lose) x loss_if_lose

    BINARY MARKET EV:
    =================
    You buy YES at price c, your probability of YES is p.

    If YES occurs (probability p):   Profit = 1 - c
    If NO occurs (probability 1-p):  Loss = -c

    EV = p x (1-c) + (1-p) x (-c)
       = p - pc - c + pc
       = p - c

    BEAUTIFUL RESULT: EV = Your probability - Market price = YOUR EDGE

    EXAMPLE:
    ========
    Market: YES at $0.40
    Your probability: 55%

    EV = 0.55 - 0.40 = 0.15 = 15%

    For every $1 you risk on this bet, you expect to profit $0.15 on average.

    If you bet $10: EV = $10 x 0.15 = $1.50 expected profit

    EV IS ABOUT THE LONG RUN:
    =========================
    A single bet can win or lose. EV tells you what happens ON AVERAGE
    over many bets.

    Law of Large Numbers: As n -> infinity, actual average -> EV

    After 100 identical +EV bets:
    - Total EV = 100 x single_EV
    - Actual profit will be close to Total EV (with some variance)

    EV != WHAT YOU WILL DEFINITELY GET:
    ===================================
    EV is not a guarantee. It's a statistical expectation.

    Example: Lottery with 1% chance of $100 payout, costs $0.50
    EV = 0.01 x $100 - 0.99 x $0.50 = $1.00 - $0.495 = $0.505

    Positive EV! But you'll probably lose your $0.50 (99% of the time).
    The 1% jackpot makes it +EV on average.

    VARIANCE MATTERS:
    =================
    Two bets can have the same EV but very different risk profiles:

    Bet A: 50% chance to win $2, 50% chance to lose $1
           EV = 0.5 x $2 - 0.5 x $1 = $0.50

    Bet B: 1% chance to win $100, 99% chance to lose $0.495
           EV = 0.01 x $100 - 0.99 x $0.495 ~ $0.50

    Same EV, but Bet B is much more volatile!

    This is why position sizing (Kelly) considers BOTH edge AND odds.

    EDGE VS. EV:
    ============
    Edge = probability difference = p - c (dimensionless, 0-1)
    EV = edge x bet size (in dollars)

    You want POSITIVE edge, and you size bets to maximize long-term EV
    while managing variance.

    COMPOUNDING EV:
    ===============
    The magic of +EV betting is compounding:

    Start: $100
    Each bet: 5% edge, betting 10% of bankroll

    After 1 bet: EV = $100 x 0.10 x 0.05 = $0.50 -> $100.50
    After 2 bets: EV = $100.50 x 0.10 x 0.05 = $0.50 -> $101.00
    ...
    After 100 bets: Your EV has compounded!

    This is why LONG-TERM thinking is crucial in prediction markets.
    """)


def explain_variance_and_risk():
    """
    In-depth explanation of variance and risk management.
    """
    print("""
    ======================================================================
                  VARIANCE AND RISK MANAGEMENT - DEEP DIVE
    ======================================================================

    VARIANCE DEFINITION:
    ====================
    Variance measures how spread out your outcomes are from the average.

    Var(X) = E[(X - mu)^2] = E[X^2] - (E[X])^2

    Standard Deviation = sqrtVariance (same units as X)

    FOR A BINARY BET:
    =================
    Outcome: +W with prob p, -L with prob (1-p)

    Mean mu = pxW - (1-p)xL
    Variance = px(W-mu)^2 + (1-p)x(-L-mu)^2
             = px(1-p)x(W+L)^2

    For a $1 bet at price c with your probability p:
    W = 1-c (profit if win)
    L = c (loss if lose)

    Variance = px(1-p)x1^2 = px(1-p)

    Maximum variance at p = 0.5 (most uncertain)
    Variance = 0.5 x 0.5 = 0.25

    WHY VARIANCE MATTERS:
    =====================
    1. DRAWDOWNS: High variance means larger swings
       - You might be down 30% before recovering
       - Can you emotionally handle that?

    2. RISK OF RUIN: With limited bankroll, variance can bankrupt you
       - Even +EV strategies can bust if sized too aggressively

    3. TIME TO PROFIT: Higher variance = longer time to converge to EV
       - You need more bets for the Law of Large Numbers to kick in

    THE DRAWDOWN TRAP:
    ==================
    Drawdowns are ASYMMETRIC:

    If you lose 50%, you need 100% gain to recover!
    If you lose 20%, you need 25% gain to recover.

    Formula: Required recovery = Loss / (1 - Loss)

    Loss    Required Recovery
    10%     11.1%
    20%     25.0%
    30%     42.9%
    50%     100.0%
    75%     300.0%

    This is why avoiding large drawdowns is CRITICAL.

    VARIANCE VS. KELLY FRACTION:
    ============================
    Variance of bankroll after n bets scales with f^2:

    Var(W_n) proportional to f^2 x n

    Doubling your bet size QUADRUPLES variance!

    This is why fractional Kelly is so powerful:
    - Half Kelly: (0.5)^2 = 0.25 = 25% of full Kelly variance
    - Quarter Kelly: (0.25)^2 = 0.0625 = 6.25% of full Kelly variance

    SHARPE RATIO:
    =============
    Sharpe measures return per unit of risk:

    Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns

    Higher Sharpe = better risk-adjusted performance

    Benchmarks:
    - < 1.0: Poor (too much risk for the return)
    - 1.0 - 2.0: Good
    - 2.0 - 3.0: Excellent
    - > 3.0: Exceptional (possibly too good - check for errors!)

    PRACTICAL RISK MANAGEMENT:
    ==========================
    1. POSITION LIMITS: Never bet more than X% on one market
       - Typical: 5-20% max per position

    2. CORRELATION LIMITS: Cap exposure to correlated events
       - If jobs and Fed are correlated, don't overload on both

    3. DAILY LOSS LIMITS: Stop trading if down X% in a day
       - Prevents tilt and emotional trading

    4. BANKROLL PRESERVATION: Start conservative, increase as edge verified
       - Better to grow slowly than bust quickly

    5. DIVERSIFICATION: Spread across independent events
       - Reduces overall portfolio variance

    THE 50% RULE:
    =============
    A good heuristic: Size positions so your MAX LOSS in a worst-case
    correlated scenario is < 50% of bankroll.

    Why 50%? You can recover from 50% (need 100% gain).
    75% loss needs 300% gain - very hard to come back from.
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the integrated scanner."""
    print("=" * 70)
    print("PREDICTION MARKET SCANNER & ARBITRAGE DETECTOR")
    print("=" * 70)

    # Create scanner
    config = TradingConfig(
        bankroll=100.0,
        kelly_fraction=0.25,
        min_ev=0.03
    )
    scanner = IntegratedScanner(config)

    try:
        # Scan markets
        scanner.scan_all(include_dk=False)

        # Show results
        scanner.print_market_summary()
        scanner.print_arbitrage_opportunities()

        # Example prediction with confidence
        print("\n" + "=" * 70)
        print("EXAMPLE: Adding a prediction with confidence")
        print("=" * 70)

        # Get a sample market if available
        all_markets = []
        for markets in scanner.markets_by_platform.values():
            all_markets.extend(markets)

        if all_markets:
            sample_market = all_markets[0]
            print(f"\nSample market: {sample_market.title}")
            print(f"Current YES price: ${sample_market.yes_price:.2f}")

            # Add prediction with confidence
            scanner.add_prediction_with_confidence(
                market=sample_market,
                model_prob_yes=0.55,  # Your model's estimate
                confidence=0.7        # 70% confident in model
            )

            # Get recommendations
            scanner.print_recommendations()

    finally:
        scanner.close()

    # Educational content
    print("\n" + "=" * 70)
    print("MATHEMATICAL DEEP DIVES AVAILABLE:")
    print("=" * 70)
    print("Run these functions for detailed explanations:")
    print("  explain_kelly_criterion()")
    print("  explain_expected_value()")
    print("  explain_variance_and_risk()")


if __name__ == "__main__":
    main()
