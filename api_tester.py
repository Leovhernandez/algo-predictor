"""
API Connection Tester for Prediction Market Platforms
======================================================
Tests connectivity to:
1. Kalshi - RSA signature-based authentication
2. Polymarket - Public CLOB API (no auth required for reads)
3. DKPredictions - Web scraping (no public API)
"""

import os
import time
import base64
import hashlib
import requests
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Selenium imports for browser automation
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# BeautifulSoup for HTML parsing
from bs4 import BeautifulSoup


# =============================================================================
# KALSHI API - RSA Signature Authentication
# =============================================================================
#
# WHY RSA signing instead of plain API keys?
# - Prevents replay attacks: Each request has unique timestamp
# - Key never transmitted: Only signature sent, private key stays local
# - Non-repudiation: Server can verify you signed the request

class KalshiAuthenticator:
    """
    Handles Kalshi API authentication using RSA-SHA256 signatures.

    The signature proves you possess the private key without transmitting it.
    Message = timestamp_ms + http_method + path (e.g., "1706000000000GET/trade/v3/exchange/status")
    """

    def __init__(self, credentials_path: str = "API/Read Write Algo Predictor.txt"):
        # Load credentials from file (kept outside source control)
        self.key_id, self.private_key = self._load_credentials(credentials_path)
        self.base_url = "https://api.kalshi.com"

    def _load_credentials(self, path: str):
        """
        Parse the credentials file to extract Key ID and RSA private key.

        Q: What happens if you change the path separator on different OS?
        Try: print(Path(path).resolve()) to see normalized path
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {path}")

        content = file_path.read_text()

        # Extract Key ID (UUID between the Key ID markers)
        key_id = None
        private_key_pem = None

        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            # Key ID is on line after "-----Key ID-----"
            if "Key ID" in line and i + 1 < len(lines):
                key_id = lines[i + 1].strip()
            # Capture the full PEM block
            if "BEGIN RSA PRIVATE KEY" in line:
                pem_lines = []
                for j in range(i, len(lines)):
                    pem_lines.append(lines[j])
                    if "END RSA PRIVATE KEY" in lines[j]:
                        break
                private_key_pem = '\n'.join(pem_lines)

        if not key_id or not private_key_pem:
            raise ValueError("Could not parse Key ID or Private Key from credentials file")

        # Convert PEM string to cryptography key object
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend()
        )

        return key_id, private_key

    def _generate_signature(self, timestamp_ms: int, method: str, path: str) -> str:
        """
        Create RSA-SHA256 signature of the request.

        WHY this specific format?
        - timestamp: Prevents replay attacks (requests expire)
        - method: Ensures GET signature can't be reused for POST
        - path: Binds signature to specific endpoint

        Q: What vulnerability would exist if we omitted the method?
        """
        # Construct message exactly as Kalshi expects
        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode('utf-8')

        # Sign with RSA-SHA256 and PKCS1v15 padding
        signature = self.private_key.sign(
            message_bytes,
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Base64 encode for HTTP header transmission
        return base64.b64encode(signature).decode('utf-8')

    def get_headers(self, method: str, path: str) -> dict:
        """
        Generate authentication headers for a Kalshi API request.

        Returns dict with:
        - KALSHI-ACCESS-KEY: Your key ID (public identifier)
        - KALSHI-ACCESS-TIMESTAMP: Current time in milliseconds
        - KALSHI-ACCESS-SIGNATURE: RSA signature proving key ownership
        """
        timestamp_ms = int(time.time() * 1000)
        signature = self._generate_signature(timestamp_ms, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }

    def test_connection(self) -> dict:
        """
        Test API connectivity by fetching exchange status.

        This endpoint is lightweight and confirms:
        1. Network connectivity
        2. Credential validity
        3. Signature generation correctness
        """
        path = "/trade/v3/exchange/status"
        method = "GET"

        headers = self.get_headers(method, path)
        url = f"{self.base_url}{path}"

        response = requests.get(url, headers=headers, timeout=10)

        return {
            "platform": "Kalshi",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else response.text
        }


# =============================================================================
# POLYMARKET API - Public CLOB (Central Limit Order Book)
# =============================================================================
#
# Polymarket uses a public API for reading market data.
# No authentication required for read operations.
# Docs: https://docs.polymarket.com/

class PolymarketClient:
    """
    Client for Polymarket's public CLOB API.

    WHY "CLOB"?
    - Central Limit Order Book: All orders aggregated centrally
    - Allows viewing current bids/asks without authentication
    - Write operations (placing orders) require wallet signature
    """

    def __init__(self):
        # Gamma API for market metadata
        self.gamma_url = "https://gamma-api.polymarket.com"
        # CLOB API for order book data
        self.clob_url = "https://clob.polymarket.com"

    def get_markets(self, limit: int = 10, active: bool = True) -> dict:
        """
        Fetch available prediction markets.

        Q: Why might we want to filter by 'active' status?
        Try: Compare results with active=True vs active=False
        """
        params = {
            "limit": limit,
            "active": str(active).lower()
        }

        response = requests.get(
            f"{self.gamma_url}/markets",
            params=params,
            timeout=10
        )

        return {
            "platform": "Polymarket",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else response.text
        }

    def test_connection(self) -> dict:
        """Test API connectivity by fetching a small number of markets."""
        return self.get_markets(limit=1)


# =============================================================================
# DKPREDICTIONS - Selenium Web Scraping
# =============================================================================
#
# DraftKings Predictions lacks a public API.
# We use Selenium (headless Chrome) to render JavaScript and extract data.
#
# ARCHITECTURE:
# 1. Selenium drives a headless browser (no visible window)
# 2. Browser executes JavaScript, rendering dynamic content
# 3. BeautifulSoup parses the final HTML
# 4. We extract structured data from the DOM
#
# IMPORTANT CONSIDERATIONS:
# - Check robots.txt before scraping
# - Respect rate limits (add delays between requests)
# - Terms of Service may prohibit scraping - use for research only

@dataclass
class DKPrediction:
    """
    Data structure for a single DK prediction/pick.

    WHY use dataclass?
    - Immutable by default (safer)
    - Auto-generates __init__, __repr__, __eq__
    - Type hints provide documentation and IDE support

    Q: What's the difference between @dataclass and a regular class?
    Try: print(DKPrediction.__dict__) to see auto-generated methods
    """
    player_name: str
    stat_type: str  # e.g., "Points", "Rebounds", "Passing Yards"
    line: float  # The over/under threshold
    over_multiplier: Optional[float] = None
    under_multiplier: Optional[float] = None
    sport: Optional[str] = None
    game_info: Optional[str] = None


class DKPredictionsClient:
    """
    Selenium-based scraper for DraftKings Pick6/Predictions.

    WHY Selenium over plain requests?
    - DK content is JavaScript-rendered (React/Next.js)
    - Plain HTTP returns empty shells, not actual data
    - Selenium executes JS just like a real browser

    FLOW:
    1. Initialize headless Chrome via WebDriver
    2. Navigate to target URL
    3. Wait for dynamic content to load (explicit waits)
    4. Extract rendered HTML
    5. Parse with BeautifulSoup
    6. Return structured data
    """

    def __init__(self, headless: bool = True):
        """
        Initialize the scraper.

        Args:
            headless: If True, browser runs invisibly (faster, no GUI)
                      If False, browser window visible (useful for debugging)

        Q: Why would you ever run with headless=False?
        Try: Set headless=False and watch the browser navigate
        """
        self.base_url = "https://pick6.draftkings.com"
        self.headless = headless
        self.driver = None  # Lazy initialization

        # Simple request headers for non-Selenium requests
        self.headers = {
            "User-Agent": "AlgoPredictor/1.0 (Research Bot)"
        }

    def _create_driver(self) -> webdriver.Chrome:
        """
        Create and configure a Chrome WebDriver instance.

        WHY these specific options?
        - headless: No visible window (server-friendly)
        - disable-gpu: Prevents GPU errors on headless systems
        - no-sandbox: Required for some Linux environments
        - window-size: Some sites render differently at small sizes
        - disable-dev-shm-usage: Prevents /dev/shm overflow in Docker

        Returns configured Chrome WebDriver
        """
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")  # New headless mode (Chrome 109+)

        # Performance and compatibility options
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        # Reduce detection fingerprint
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Set a realistic user agent
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        # WebDriver Manager auto-downloads correct ChromeDriver version
        # WHY? Chrome updates frequently; manual driver management is painful
        service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)

        # Additional anti-detection: Remove webdriver property
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )

        return driver

    def _ensure_driver(self):
        """Lazy initialization of WebDriver (create only when needed)."""
        if self.driver is None:
            self.driver = self._create_driver()

    def close(self):
        """
        Clean up WebDriver resources.

        IMPORTANT: Always call this when done!
        Unclosed drivers leave zombie Chrome processes.

        Q: What happens if you forget to call close()?
        Try: Check Task Manager after running without close()
        """
        if self.driver:
            self.driver.quit()
            self.driver = None

    def __enter__(self):
        """Context manager entry - enables 'with' statement usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-cleanup on scope exit."""
        self.close()

    def check_robots_txt(self) -> dict:
        """
        Check robots.txt to understand scraping permissions.

        robots.txt is a standard file that tells bots which paths
        they're allowed/forbidden to access.

        WHY check this?
        - Ethical scraping practice
        - Violating can result in IP bans
        - Some jurisdictions consider ignoring robots.txt in ToS violations
        """
        try:
            response = requests.get(
                f"{self.base_url}/robots.txt",
                headers=self.headers,
                timeout=10
            )
            return {
                "platform": "DKPredictions",
                "check": "robots.txt",
                "status_code": response.status_code,
                "content": response.text if response.status_code == 200 else None
            }
        except requests.RequestException as e:
            return {
                "platform": "DKPredictions",
                "check": "robots.txt",
                "error": str(e)
            }

    def test_connection(self) -> dict:
        """
        Test basic HTTP connectivity (no Selenium).

        This lightweight check confirms the site is reachable
        before investing in full browser automation.
        """
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                timeout=10
            )
            is_html = "text/html" in response.headers.get("Content-Type", "")

            return {
                "platform": "DKPredictions",
                "status_code": response.status_code,
                "success": response.status_code == 200 and is_html,
                "note": "Use scrape_predictions() for full data extraction",
                "content_type": response.headers.get("Content-Type")
            }
        except requests.RequestException as e:
            return {
                "platform": "DKPredictions",
                "success": False,
                "error": str(e)
            }

    def test_selenium_connection(self) -> dict:
        """
        Test Selenium browser automation.

        This verifies:
        1. ChromeDriver is installed and working
        2. Headless Chrome can launch
        3. Site loads without blocking automation
        """
        try:
            self._ensure_driver()

            # Navigate to the site
            self.driver.get(self.base_url)

            # Wait for page to have a title (basic load confirmation)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            title = self.driver.title
            current_url = self.driver.current_url

            return {
                "platform": "DKPredictions",
                "method": "Selenium",
                "success": True,
                "page_title": title,
                "final_url": current_url,
                "note": "Selenium working - ready for full scraping"
            }

        except TimeoutException:
            return {
                "platform": "DKPredictions",
                "method": "Selenium",
                "success": False,
                "error": "Page load timeout - site may be blocking automation"
            }
        except WebDriverException as e:
            return {
                "platform": "DKPredictions",
                "method": "Selenium",
                "success": False,
                "error": f"WebDriver error: {str(e)}"
            }
        except Exception as e:
            return {
                "platform": "DKPredictions",
                "method": "Selenium",
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def scrape_predictions(self, sport: str = None, wait_time: int = 15) -> dict:
        """
        Scrape prediction/pick data from DraftKings.

        Args:
            sport: Optional sport filter (e.g., "nba", "nfl")
            wait_time: Max seconds to wait for content to load

        Returns:
            Dict with success status, predictions list, and metadata

        SCRAPING STRATEGY:
        1. Navigate to picks page
        2. Wait for dynamic content (JavaScript renders the data)
        3. Extract full HTML after JS execution
        4. Parse with BeautifulSoup to find prediction cards
        5. Extract structured data from each card
        """
        try:
            self._ensure_driver()

            # Build URL (sport filter if provided)
            url = self.base_url
            if sport:
                url = f"{self.base_url}/{sport.lower()}"

            print(f"  Navigating to: {url}")
            self.driver.get(url)

            # Wait for content to load
            # Strategy: Wait for elements that indicate data is present
            # This is fragile - selector may change if DK updates their site
            print(f"  Waiting up to {wait_time}s for content...")

            # Try multiple possible selectors (sites change frequently)
            possible_selectors = [
                "[data-testid='player-card']",
                "[data-testid='pick-card']",
                ".player-card",
                ".pick-card",
                "[class*='PlayerCard']",
                "[class*='PickCard']",
                "article",  # Fallback - many sites use article for cards
            ]

            content_found = False
            for selector in possible_selectors:
                try:
                    WebDriverWait(self.driver, 3).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    content_found = True
                    print(f"  Found content with selector: {selector}")
                    break
                except TimeoutException:
                    continue

            if not content_found:
                # Even if no specific selector found, continue - might find data in HTML
                print("  No specific card selector found, attempting generic parse...")
                time.sleep(wait_time)  # Give JS time to render

            # Get fully rendered HTML
            html = self.driver.page_source

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Extract predictions
            predictions = self._parse_predictions(soup)

            return {
                "platform": "DKPredictions",
                "success": True,
                "url_scraped": url,
                "predictions_count": len(predictions),
                "predictions": predictions,
                "note": "Data structure may vary based on site updates"
            }

        except TimeoutException:
            return {
                "platform": "DKPredictions",
                "success": False,
                "error": "Timeout waiting for page content"
            }
        except WebDriverException as e:
            return {
                "platform": "DKPredictions",
                "success": False,
                "error": f"Browser automation error: {str(e)}"
            }
        except Exception as e:
            return {
                "platform": "DKPredictions",
                "success": False,
                "error": f"Scraping error: {str(e)}"
            }

    def _parse_predictions(self, soup: BeautifulSoup) -> list:
        """
        Parse BeautifulSoup object to extract prediction data.

        WHY separate parsing method?
        - Separation of concerns: Navigation vs. Parsing
        - Easier to update when site structure changes
        - Testable independently (mock HTML input)

        NOTE: This is inherently fragile. Site redesigns will break it.
        The selectors here are educated guesses based on common patterns.
        You may need to inspect the actual site HTML and adjust.
        """
        predictions = []

        # Strategy 1: Look for JSON data in script tags
        # Many React sites embed state as JSON
        script_tags = soup.find_all("script", {"type": "application/json"})
        for script in script_tags:
            try:
                import json
                data = json.loads(script.string)
                # Look for prediction-like structures
                predictions.extend(self._extract_from_json(data))
            except (json.JSONDecodeError, TypeError):
                continue

        # Strategy 2: Look for __NEXT_DATA__ (Next.js apps)
        next_data = soup.find("script", {"id": "__NEXT_DATA__"})
        if next_data and next_data.string:
            try:
                import json
                data = json.loads(next_data.string)
                predictions.extend(self._extract_from_json(data))
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: Parse visible DOM elements
        # Look for common card/prediction patterns
        card_candidates = soup.find_all(
            lambda tag: tag.name in ["div", "article", "section"]
            and any(cls for cls in (tag.get("class") or [])
                   if any(kw in cls.lower() for kw in ["card", "pick", "player", "prop"]))
        )

        for card in card_candidates:
            prediction = self._parse_card_element(card)
            if prediction:
                predictions.append(prediction)

        # Deduplicate by player+stat combination
        seen = set()
        unique_predictions = []
        for p in predictions:
            key = f"{p.get('player_name', '')}_{p.get('stat_type', '')}_{p.get('line', '')}"
            if key not in seen:
                seen.add(key)
                unique_predictions.append(p)

        return unique_predictions

    def _extract_from_json(self, data, depth: int = 0) -> list:
        """
        Recursively search JSON for prediction-like structures.

        WHY recursive?
        - JSON can be deeply nested
        - Prediction data might be at any level
        - We look for keys that suggest player/line data

        Args:
            data: JSON object (dict or list)
            depth: Current recursion depth (prevents infinite loops)

        Returns:
            List of prediction dicts found
        """
        if depth > 10:  # Prevent infinite recursion
            return []

        predictions = []

        if isinstance(data, dict):
            # Check if this dict looks like a prediction
            has_player = any(k in data for k in ["player", "playerName", "name", "athlete"])
            has_line = any(k in data for k in ["line", "value", "threshold", "over", "under"])
            has_stat = any(k in data for k in ["stat", "statType", "category", "prop"])

            if has_player and (has_line or has_stat):
                predictions.append({
                    "player_name": data.get("player") or data.get("playerName") or data.get("name") or data.get("athlete", {}).get("name", "Unknown"),
                    "stat_type": data.get("stat") or data.get("statType") or data.get("category") or "Unknown",
                    "line": data.get("line") or data.get("value") or data.get("threshold") or 0,
                    "over_multiplier": data.get("overMultiplier") or data.get("overOdds"),
                    "under_multiplier": data.get("underMultiplier") or data.get("underOdds"),
                    "raw_data": data  # Keep original for debugging
                })

            # Recurse into all values
            for value in data.values():
                predictions.extend(self._extract_from_json(value, depth + 1))

        elif isinstance(data, list):
            for item in data:
                predictions.extend(self._extract_from_json(item, depth + 1))

        return predictions

    def _parse_card_element(self, card) -> Optional[dict]:
        """
        Extract prediction data from a DOM card element.

        This uses heuristics to find player names, stats, and lines
        from visible text content.

        Returns dict if valid prediction found, None otherwise
        """
        text = card.get_text(separator=" ", strip=True)

        if len(text) < 10:  # Too short to be meaningful
            return None

        # Look for patterns like "LeBron James Points 25.5"
        # or "Patrick Mahomes Passing Yards 275.5"

        # Common stat types to search for
        stat_keywords = [
            "points", "rebounds", "assists", "steals", "blocks",
            "passing yards", "rushing yards", "receiving yards",
            "touchdowns", "completions", "interceptions",
            "strikeouts", "hits", "runs", "home runs", "rbis",
            "goals", "saves", "shots"
        ]

        # Try to find a number that looks like a line
        numbers = re.findall(r'\d+\.?\d*', text)
        if not numbers:
            return None

        # The line is usually a decimal number
        potential_lines = [float(n) for n in numbers if '.' in n or float(n) > 1]
        if not potential_lines:
            return None

        # Find which stat keyword appears
        stat_found = None
        text_lower = text.lower()
        for stat in stat_keywords:
            if stat in text_lower:
                stat_found = stat.title()
                break

        if not stat_found:
            return None

        # Try to extract player name (text before the stat keyword)
        stat_pos = text_lower.find(stat_found.lower() if stat_found else "")
        player_name = text[:stat_pos].strip() if stat_pos > 0 else "Unknown"

        # Clean up player name (remove common prefixes)
        player_name = re.sub(r'^(over|under|more|less)\s+', '', player_name, flags=re.IGNORECASE)
        player_name = player_name.strip()

        if len(player_name) < 3:  # Name too short
            return None

        return {
            "player_name": player_name[:50],  # Limit length
            "stat_type": stat_found,
            "line": potential_lines[0],
            "source": "DOM_parse"
        }

    def get_available_sports(self) -> dict:
        """
        Discover which sports/categories are available on the site.

        Scrapes the navigation or sidebar to find sport options.
        """
        try:
            self._ensure_driver()
            self.driver.get(self.base_url)

            # Wait for nav to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "nav"))
            )

            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            # Look for navigation links
            nav_links = soup.find_all("a", href=True)
            sports = []

            sport_keywords = ["nba", "nfl", "mlb", "nhl", "pga", "soccer", "tennis", "mma", "ufc"]

            for link in nav_links:
                href = link.get("href", "").lower()
                text = link.get_text(strip=True)

                for sport in sport_keywords:
                    if sport in href or sport in text.lower():
                        sports.append({
                            "name": text or sport.upper(),
                            "path": link.get("href"),
                            "keyword": sport
                        })

            # Deduplicate
            seen = set()
            unique_sports = []
            for s in sports:
                if s["keyword"] not in seen:
                    seen.add(s["keyword"])
                    unique_sports.append(s)

            return {
                "platform": "DKPredictions",
                "success": True,
                "sports": unique_sports
            }

        except Exception as e:
            return {
                "platform": "DKPredictions",
                "success": False,
                "error": str(e)
            }


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests(include_selenium: bool = False):
    """
    Execute connectivity tests for all platforms.

    Args:
        include_selenium: If True, also runs Selenium browser tests (slower)

    Q: Why test each platform independently instead of failing fast?
    Try: Comment out one platform and observe behavior
    """
    results = {}

    # Kalshi (authenticated)
    print("Testing Kalshi API...")
    try:
        kalshi = KalshiAuthenticator()
        results["kalshi"] = kalshi.test_connection()
        print(f"  Status: {results['kalshi']['status_code']}")
        print(f"  Success: {results['kalshi']['success']}")
    except Exception as e:
        results["kalshi"] = {"success": False, "error": str(e)}
        print(f"  Error: {e}")

    print()

    # Polymarket (public)
    print("Testing Polymarket API...")
    try:
        poly = PolymarketClient()
        results["polymarket"] = poly.test_connection()
        print(f"  Status: {results['polymarket']['status_code']}")
        print(f"  Success: {results['polymarket']['success']}")
    except Exception as e:
        results["polymarket"] = {"success": False, "error": str(e)}
        print(f"  Error: {e}")

    print()

    # DKPredictions (HTTP check)
    print("Testing DKPredictions (HTTP)...")
    try:
        dk = DKPredictionsClient()
        results["dkpredictions_http"] = dk.test_connection()
        print(f"  Status: {results['dkpredictions_http'].get('status_code', 'N/A')}")
        print(f"  Success: {results['dkpredictions_http']['success']}")
    except Exception as e:
        results["dkpredictions_http"] = {"success": False, "error": str(e)}
        print(f"  Error: {e}")

    # DKPredictions (Selenium) - optional, slower
    if include_selenium:
        print()
        print("Testing DKPredictions (Selenium)...")
        try:
            # Use context manager to ensure cleanup
            with DKPredictionsClient(headless=True) as dk_selenium:
                results["dkpredictions_selenium"] = dk_selenium.test_selenium_connection()
                print(f"  Success: {results['dkpredictions_selenium']['success']}")
                if results['dkpredictions_selenium'].get('page_title'):
                    print(f"  Page Title: {results['dkpredictions_selenium']['page_title']}")
        except Exception as e:
            results["dkpredictions_selenium"] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for platform, result in results.items():
        status = "[OK]" if result.get("success") else "[FAIL]"
        print(f"{status} {platform}")

    return results


def demo_dk_scraping(sport: str = None):
    """
    Demonstrate full DKPredictions scraping.

    This shows the complete workflow:
    1. Initialize Selenium browser
    2. Navigate to predictions page
    3. Wait for JS to render
    4. Extract and parse data
    5. Clean up browser

    Args:
        sport: Optional sport to filter (e.g., "nba", "nfl")
    """
    print("=" * 50)
    print("DKPredictions Scraping Demo")
    print("=" * 50)

    # Using context manager ensures browser cleanup even on error
    with DKPredictionsClient(headless=True) as dk:
        # First, check robots.txt
        print("\n1. Checking robots.txt...")
        robots = dk.check_robots_txt()
        if robots.get("content"):
            print("   robots.txt found - review for allowed paths")
        else:
            print("   No robots.txt or error fetching")

        # Test Selenium connectivity
        print("\n2. Testing Selenium browser...")
        selenium_test = dk.test_selenium_connection()
        if not selenium_test["success"]:
            print(f"   Failed: {selenium_test.get('error')}")
            return selenium_test

        print(f"   Page loaded: {selenium_test.get('page_title')}")

        # Discover available sports
        print("\n3. Discovering available sports...")
        sports = dk.get_available_sports()
        if sports["success"] and sports.get("sports"):
            print(f"   Found {len(sports['sports'])} sports:")
            for s in sports["sports"][:5]:  # Show first 5
                print(f"     - {s['name']}")
        else:
            print("   Could not discover sports (site may have changed)")

        # Scrape predictions
        print(f"\n4. Scraping predictions{f' for {sport}' if sport else ''}...")
        predictions = dk.scrape_predictions(sport=sport)

        if predictions["success"]:
            print(f"   Found {predictions['predictions_count']} predictions")

            # Show sample predictions
            if predictions["predictions"]:
                print("\n   Sample predictions:")
                for pred in predictions["predictions"][:5]:
                    player = pred.get("player_name", "Unknown")
                    stat = pred.get("stat_type", "Unknown")
                    line = pred.get("line", "N/A")
                    print(f"     - {player}: {stat} {line}")
        else:
            print(f"   Scraping failed: {predictions.get('error')}")

        return predictions


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--selenium":
            # Run all tests including Selenium
            run_all_tests(include_selenium=True)
        elif sys.argv[1] == "--demo":
            # Run DK scraping demo
            sport = sys.argv[2] if len(sys.argv) > 2 else None
            demo_dk_scraping(sport=sport)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python api-tester.py           # Run basic connectivity tests")
            print("  python api-tester.py --selenium # Include Selenium browser tests")
            print("  python api-tester.py --demo     # Demo DK scraping")
            print("  python api-tester.py --demo nba # Demo DK scraping for NBA")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: run basic tests without Selenium
        run_all_tests(include_selenium=False)
