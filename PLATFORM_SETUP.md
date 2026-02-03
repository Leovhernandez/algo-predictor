# Platform Authentication & Setup Guide

## Quick Reference

| Platform | Read Data | Place Trades | Status |
|----------|-----------|--------------|--------|
| Kalshi | API Key | API Key | Ready |
| Polymarket | No auth | Crypto wallet | Setup needed |
| DraftKings | Web scraping | Not available | Read-only |

---

## 1. KALSHI (Ready)

### Authentication
Your API key is already configured in `API/Read Write Algo Predictor.txt`.

### Rate Limits (Basic Tier)
- **Read requests**: 20/second
- **Write requests**: 10/second

The system uses 80% of these limits (16 read/sec, 8 write/sec) to be safe.

### Upgrading Tiers
Contact Kalshi support to request Advanced or Premier tier for higher limits.

### Testing Connection
```bash
python api_tester.py
```

If you see DNS errors, check your network/firewall settings.

---

## 2. POLYMARKET (Setup Needed for Trading)

### Reading Data (No Auth Required)
The public API works without authentication. You can already fetch markets.

### Placing Trades (Wallet Required)
Polymarket is a **decentralized exchange** on the Polygon blockchain.
To trade, you need:

1. **Crypto Wallet** (MetaMask, Coinbase Wallet, etc.)
   - Install: https://metamask.io/
   - Create or import a wallet

2. **USDC on Polygon Network**
   - Polymarket uses USDC stablecoin
   - Bridge USDC from Ethereum to Polygon, or
   - Buy USDC directly on Polygon via exchange

3. **Connect Wallet to Polymarket**
   - Go to https://polymarket.com
   - Click "Connect Wallet"
   - Approve the connection

4. **For Programmatic Trading**
   - Polymarket uses the CLOB (Central Limit Order Book)
   - Trades require cryptographic signatures from your wallet
   - See: https://docs.polymarket.com/

### Implementation Note
The current system can READ Polymarket data. For trading, we would need to
integrate with web3.py or similar to sign transactions with your wallet's
private key. This is more complex than API key authentication.

---

## 3. DRAFTKINGS PREDICTIONS (Read-Only)

### No API Available
DraftKings does not provide a public API for their predictions market.

### Web Scraping
We use Selenium to scrape the website:
- Requires Chrome browser installed
- Slower than API calls
- May break if DK changes their website

### No Trading
There is no programmatic way to place trades on DraftKings.
Use their data for reference/comparison only.

### Running the Scraper
```bash
python api_tester.py --demo
```

---

## Rate Limiting Configuration

The system includes rate limiting to prevent API bans:

```python
# In market_scanner.py
self.kalshi_read_limiter = RateLimiter(rate=16, capacity=16)   # 80% of 20/sec
self.kalshi_write_limiter = RateLimiter(rate=8, capacity=8)    # 80% of 10/sec
```

To adjust for higher tiers, modify these values in `market_scanner.py`:

```python
# Advanced tier example (hypothetical limits)
self.kalshi_read_limiter = RateLimiter(rate=80, capacity=80)   # 100/sec
self.kalshi_write_limiter = RateLimiter(rate=40, capacity=40)  # 50/sec
```

---

## Running the Scanner

### From Your Machine

1. Open PowerShell or Command Prompt
2. Navigate to the project:
   ```
   cd C:\quant\algo-predictor
   ```
3. Activate virtual environment:
   ```
   venv\Scripts\activate
   ```
4. Run the scanner:
   ```
   python run.py
   ```
   Then select option 5 (Scan Live Markets)

### Or run directly:
```python
from market_scanner import IntegratedScanner, TradingConfig

config = TradingConfig(bankroll=100.0)
scanner = IntegratedScanner(config)
scanner.scan_all()
scanner.print_market_summary()
scanner.print_arbitrage_opportunities()
scanner.close()
```

---

## Troubleshooting

### "Failed to resolve api.kalshi.com"
- Check internet connection
- Check if firewall blocks the connection
- Try: `ping api.kalshi.com` in terminal
- VPN might interfere with DNS resolution

### Polymarket returns 0 markets
- Check internet connection
- API endpoint may have changed
- Try direct test: `curl https://gamma-api.polymarket.com/markets?limit=1`

### DraftKings scraper returns empty
- Chrome browser required
- Site structure may have changed
- Content may require login

---

## Next Steps

1. **Test Kalshi connection** from your home network
2. **Set up MetaMask** if you want to trade on Polymarket
3. **Run the interactive session** to practice with simulated trades
4. **Verify your edge** with paper trading before enabling live trading
