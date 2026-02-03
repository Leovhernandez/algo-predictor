#!/usr/bin/env python3
"""
Prediction Market Trading System - Runner Script
=================================================

This is the MAIN ENTRY POINT for the trading system.

HOW TO RUN:
-----------
1. Activate virtual environment:
   Windows: venv\\Scripts\\activate
   Mac/Linux: source venv/bin/activate

2. Run this script:
   python run.py

3. Choose from the menu options

WHAT EACH MODULE DOES:
----------------------
- app.py: Core trading logic, Kelly criterion, backtesting
- api_tester.py: API connections to Kalshi, Polymarket, DKPredictions
- market_scanner.py: Live market fetching, arbitrage detection
- run.py: This file - user interface and execution

FILE STRUCTURE:
---------------
algo-predictor/
├── app.py              # Core trading system
├── api_tester.py       # API clients for each platform
├── market_scanner.py   # Arbitrage & live market scanner
├── run.py              # This runner script (START HERE)
├── data/               # Historical data storage (auto-created)
│   └── market_history.db
├── API/                # API credentials (gitignored)
│   └── Read Write Algo Predictor.txt
└── venv/               # Python virtual environment
"""

## To-Do: Need to add other strategies in addition to arbitrage, edit this program for just Kalshi/Polymarket since those have public APIs to get info and execute trades.
## After adding other strategies that are tested and proven to work, program the bot to be fully-automated and run on a schedule (e.g., daily/weekly) to scan markets and place trades automatically.
## It must also be able to identify which strategy to use based on information it receives each run so that trades are executed optimally and consistently.


import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("""
    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗
    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝
    ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║
    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║
    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║
    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝
              PREDICTION MARKET TRADING SYSTEM
    """)
    print("=" * 70)


def print_menu():
    """Print main menu."""
    print("""
    MAIN MENU
    ---------
    1. Run Kelly Criterion Demo (Learn the math)
    2. Run Jobs Report Model Demo
    3. Run Backtest Example
    4. Test API Connections
    5. Scan Live Markets & Detect Arbitrage
    6. Interactive Trading Session
    7. Mathematical Deep Dives
    8. Exit

    Enter choice (1-8): """, end="")


def run_kelly_demo():
    """Run Kelly criterion demonstration."""
    from app import demonstrate_kelly_math
    demonstrate_kelly_math()
    input("\nPress Enter to continue...")


def run_jobs_demo():
    """Run jobs report model demonstration."""
    from app import demonstrate_jobs_report_model
    demonstrate_jobs_report_model()
    input("\nPress Enter to continue...")


def run_backtest():
    """Run backtest example."""
    from app import run_example_backtest
    run_example_backtest()
    input("\nPress Enter to continue...")


def test_api_connections():
    """Test API connections."""
    print("\n" + "=" * 70)
    print("TESTING API CONNECTIONS")
    print("=" * 70)

    import api_tester
    api_tester.run_all_tests(include_selenium=False)
    input("\nPress Enter to continue...")


def scan_markets():
    """Scan live markets and detect arbitrage."""
    print("\n" + "=" * 70)
    print("SCANNING LIVE MARKETS")
    print("=" * 70)

    try:
        from market_scanner import IntegratedScanner, TradingConfig

        config = TradingConfig(bankroll=100.0, kelly_fraction=0.25)
        scanner = IntegratedScanner(config)

        print("\nInclude DKPredictions? (Requires Selenium, slower)")
        include_dk = input("Enter y/n [n]: ").lower().strip() == 'y'

        scanner.scan_all(include_dk=include_dk)
        scanner.print_market_summary()
        scanner.print_arbitrage_opportunities()

        scanner.close()

    except Exception as e:
        print(f"\nError: {e}")
        print("This may be due to network issues or API unavailability.")

    input("\nPress Enter to continue...")


def interactive_session():
    """Run interactive trading session."""
    print("\n" + "=" * 70)
    print("INTERACTIVE TRADING SESSION")
    print("=" * 70)

    from app import (
        TradingConfig, Platform, Market, PredictionMarketTrader,
        MonteCarloSimulator
    )

    # Get bankroll
    print("\nCurrent default bankroll: $100.00")
    try:
        bankroll_input = input("Enter your bankroll (or press Enter for $100): ").strip()
        bankroll = float(bankroll_input) if bankroll_input else 100.0
    except ValueError:
        bankroll = 100.0
        print("Invalid input, using $100.00")

    # Get Kelly fraction
    print("""
    Kelly Fraction Options:
    - Full Kelly (1.0): Aggressive, high variance
    - Half Kelly (0.5): Balanced
    - Quarter Kelly (0.25): Conservative (RECOMMENDED for beginners)
    """)
    try:
        kelly_input = input("Enter Kelly fraction (or press Enter for 0.25): ").strip()
        kelly_fraction = float(kelly_input) if kelly_input else 0.25
    except ValueError:
        kelly_fraction = 0.25
        print("Invalid input, using 0.25")

    config = TradingConfig(
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_ev=0.03,
        live_trading_enabled=False
    )

    trader = PredictionMarketTrader(config)

    print(f"\nConfiguration:")
    print(f"  Bankroll: ${config.bankroll:.2f}")
    print(f"  Kelly Fraction: {config.kelly_fraction:.0%}")
    print(f"  Min Edge: {config.min_ev:.0%}")
    print(f"  Live Trading: {'ENABLED' if config.live_trading_enabled else 'DISABLED (simulation mode)'}")

    # Interactive prediction entry
    while True:
        print("\n" + "-" * 50)
        print("ADD A PREDICTION")
        print("-" * 50)
        print("Enter market details (or 'done' to finish):")

        ticker = input("  Market ticker (e.g., JOBS-FEB26-ABOVE150K): ").strip()
        if ticker.lower() == 'done':
            break

        title = input("  Market title: ").strip() or ticker

        try:
            yes_price = float(input("  YES price (e.g., 0.42): ").strip())
            no_price = float(input("  NO price (e.g., 0.60): ").strip() or str(1 - yes_price + 0.02))
            model_prob = float(input("  Your model probability for YES (e.g., 0.55): ").strip())
            confidence = float(input("  Your confidence in model (0-1, e.g., 0.7): ").strip() or "1.0")
        except ValueError:
            print("Invalid input, skipping this prediction.")
            continue

        # Create market and add prediction
        market = Market(
            ticker=ticker,
            title=title,
            platform=Platform.KALSHI,
            yes_price=yes_price,
            no_price=no_price,
            volume=50000,
            category="MANUAL"
        )

        # Apply confidence adjustment
        from market_scanner import ConfidenceAdjustedModel
        adjusted_prob, adjusted_edge, kelly_size = \
            ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
                model_prob, yes_price, confidence, config.kelly_fraction
            )

        print(f"\n  Raw model probability: {model_prob:.1%}")
        print(f"  Market price: {yes_price:.1%}")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Adjusted probability: {adjusted_prob:.1%}")
        print(f"  Adjusted edge: {adjusted_edge:.1%}")

        trader.add_prediction(market, adjusted_prob, confidence)
        print("  Prediction added!")

    # Show recommendations
    if trader.predictions:
        recommendations = trader.get_recommendations()
        trader.print_recommendations()

        # Monte Carlo analysis
        print("\n" + "-" * 50)
        print("MONTE CARLO RISK ANALYSIS")
        print("-" * 50)

        mc_results = trader.run_monte_carlo_analysis()
        if "error" not in mc_results:
            print(f"Simulating 100 similar bets with 10,000 iterations...")
            print(f"  Mean final bankroll: ${mc_results['mean_final_bankroll']:.2f}")
            print(f"  5th percentile (bad luck): ${mc_results['percentile_5']:.2f}")
            print(f"  95th percentile (good luck): ${mc_results['percentile_95']:.2f}")
            print(f"  Probability of ruin: {mc_results['ruin_probability']:.2%}")
        else:
            print(f"  {mc_results['error']}")

        # Execute?
        print("\n" + "-" * 50)
        print("EXECUTION")
        print("-" * 50)
        print("Live trading is DISABLED for safety.")
        print("To execute, you would need to:")
        print("  1. Set config.live_trading_enabled = True")
        print("  2. Ensure API credentials are configured")
        print("  3. Review recommendations carefully")
        print("\nSimulating execution...")
        results = trader.execute_recommendations()
        for r in results:
            print(f"  {r.get('status', 'ERROR')}: {r.get('market', 'Unknown')}")

    else:
        print("\nNo predictions added.")

    input("\nPress Enter to continue...")


def math_deep_dives():
    """Show mathematical deep dive options."""
    while True:
        print("\n" + "=" * 70)
        print("MATHEMATICAL DEEP DIVES")
        print("=" * 70)
        print("""
    Choose a topic:
    1. Kelly Criterion (Optimal bet sizing)
    2. Expected Value (Understanding edge)
    3. Variance and Risk Management
    4. Back to main menu

    Enter choice (1-4): """, end="")

        choice = input().strip()

        if choice == '1':
            from market_scanner import explain_kelly_criterion
            explain_kelly_criterion()
            input("\nPress Enter to continue...")
        elif choice == '2':
            from market_scanner import explain_expected_value
            explain_expected_value()
            input("\nPress Enter to continue...")
        elif choice == '3':
            from market_scanner import explain_variance_and_risk
            explain_variance_and_risk()
            input("\nPress Enter to continue...")
        elif choice == '4':
            break
        else:
            print("Invalid choice.")


def main():
    """Main entry point."""
    print_header()

    while True:
        print_menu()
        choice = input().strip()

        if choice == '1':
            run_kelly_demo()
        elif choice == '2':
            run_jobs_demo()
        elif choice == '3':
            run_backtest()
        elif choice == '4':
            test_api_connections()
        elif choice == '5':
            scan_markets()
        elif choice == '6':
            interactive_session()
        elif choice == '7':
            math_deep_dives()
        elif choice == '8':
            print("\nGoodbye! Happy trading!")
            break
        else:
            print("Invalid choice. Please enter 1-8.")


# =============================================================================
# QUICK START EXAMPLES (for importing in Python REPL)
# =============================================================================

def quick_example():
    """
    Quick example you can run in Python REPL.

    >>> from run import quick_example
    >>> quick_example()
    """
    from app import (
        TradingConfig, Platform, Market, Prediction,
        PredictionMarketTrader, KellyCalculator
    )

    print("=" * 50)
    print("QUICK EXAMPLE")
    print("=" * 50)

    # 1. Configure
    config = TradingConfig(
        bankroll=100.0,
        kelly_fraction=0.25,  # Quarter Kelly
        min_ev=0.03
    )
    print(f"\n1. Config: ${config.bankroll} bankroll, {config.kelly_fraction:.0%} Kelly")

    # 2. Create trader
    trader = PredictionMarketTrader(config)
    print("2. Trader created")

    # 3. Create a market
    market = Market(
        ticker="JOBS-FEB26-ABOVE150K",
        title="February Jobs Report Above 150,000",
        platform=Platform.KALSHI,
        yes_price=0.42,
        no_price=0.60,
        volume=75000,
        category="JOBS"
    )
    print(f"3. Market: {market.ticker} at ${market.yes_price:.2f}")

    # 4. Add prediction with confidence
    model_prob = 0.55
    confidence = 0.75

    # Adjust for confidence
    from market_scanner import ConfidenceAdjustedModel
    adj_prob, adj_edge, kelly = ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
        model_prob, market.yes_price, confidence, config.kelly_fraction
    )
    print(f"4. Prediction: {model_prob:.0%} prob, {confidence:.0%} confidence")
    print(f"   -> Adjusted: {adj_prob:.1%} prob, {adj_edge:.1%} edge")

    trader.add_prediction(market, adj_prob, confidence)

    # 5. Get recommendations
    recommendations = trader.get_recommendations()
    print(f"5. Recommendations: {len(recommendations)} positions")

    if recommendations:
        pos = recommendations[0]
        print(f"   -> {pos.side.value.upper()} {pos.contracts} contracts @ ${pos.entry_price:.2f}")
        print(f"   -> Cost: ${pos.cost:.2f}, Expected profit: ${pos.expected_profit:.2f}")

    # 6. Simulate execution
    results = trader.execute_recommendations()
    print(f"6. Execution: {results[0].get('status', 'ERROR') if results else 'No trades'}")

    print("\n" + "=" * 50)
    print("To run interactively: python run.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
