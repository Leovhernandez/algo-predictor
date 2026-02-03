"""
Interactive Trading Session Demonstration
==========================================
This script demonstrates the complete workflow of the trading system.
"""

from app import TradingConfig, Platform, Market, PredictionMarketTrader
from market_scanner import ConfidenceAdjustedModel

def run_demo():
    print("=" * 70)
    print("INTERACTIVE TRADING SESSION")
    print("=" * 70)

    # Configuration
    bankroll = 100.0
    kelly_fraction = 0.25

    config = TradingConfig(
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_ev=0.03,
        live_trading_enabled=False
    )

    trader = PredictionMarketTrader(config)

    print(f"\nConfiguration:")
    print(f"  Bankroll: ${config.bankroll:.2f}")
    print(f"  Kelly Fraction: {config.kelly_fraction:.0%} (Quarter Kelly)")
    print(f"  Min Edge: {config.min_ev:.0%}")
    print(f"  Live Trading: DISABLED (simulation mode)")

    # ============================================================
    # PREDICTION 1: Jobs Report
    # ============================================================
    print("\n" + "-" * 50)
    print("ADDING PREDICTION 1: Jobs Report")
    print("-" * 50)

    market1 = Market(
        ticker="JOBS-FEB26-ABOVE150K",
        title="February Jobs Report Above 150,000",
        platform=Platform.KALSHI,
        yes_price=0.42,
        no_price=0.60,
        volume=75000,
        category="JOBS"
    )

    print(f"  Market: {market1.title}")
    print(f"  Ticker: {market1.ticker}")
    print(f"  YES price: ${market1.yes_price:.2f}")
    print(f"  NO price: ${market1.no_price:.2f}")

    model_prob_1 = 0.55
    confidence_1 = 0.75

    print(f"\n  Your model probability: {model_prob_1:.0%}")
    print(f"  Your confidence: {confidence_1:.0%}")

    # Apply confidence adjustment
    adj_prob_1, adj_edge_1, kelly_1 = ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
        model_prob_1, market1.yes_price, confidence_1, config.kelly_fraction
    )

    print(f"\n  CONFIDENCE ADJUSTMENT:")
    print(f"    Formula: adj = conf x model + (1-conf) x market")
    print(f"    adj = {confidence_1:.2f} x {model_prob_1:.2f} + {1-confidence_1:.2f} x {market1.yes_price:.2f}")
    print(f"    adj = {confidence_1 * model_prob_1:.3f} + {(1-confidence_1) * market1.yes_price:.3f}")
    print(f"    Adjusted probability: {adj_prob_1:.1%}")
    print(f"    Adjusted edge: {adj_edge_1:.1%}")
    print(f"    Kelly bet size: {kelly_1:.1%} of bankroll")

    trader.add_prediction(market1, adj_prob_1, confidence_1)
    print("\n  [OK] Prediction added!")

    # ============================================================
    # PREDICTION 2: Bitcoin Price
    # ============================================================
    print("\n" + "-" * 50)
    print("ADDING PREDICTION 2: Bitcoin Price")
    print("-" * 50)

    market2 = Market(
        ticker="BTC-MAR26-ABOVE50K",
        title="Bitcoin Above $50,000 by March 1st",
        platform=Platform.POLYMARKET,
        yes_price=0.38,
        no_price=0.65,
        volume=120000,
        category="CRYPTO"
    )

    print(f"  Market: {market2.title}")
    print(f"  Ticker: {market2.ticker}")
    print(f"  YES price: ${market2.yes_price:.2f}")
    print(f"  NO price: ${market2.no_price:.2f}")

    model_prob_2 = 0.48
    confidence_2 = 0.60

    print(f"\n  Your model probability: {model_prob_2:.0%}")
    print(f"  Your confidence: {confidence_2:.0%}")

    # Apply confidence adjustment
    adj_prob_2, adj_edge_2, kelly_2 = ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
        model_prob_2, market2.yes_price, confidence_2, config.kelly_fraction
    )

    print(f"\n  CONFIDENCE ADJUSTMENT:")
    print(f"    Formula: adj = conf x model + (1-conf) x market")
    print(f"    adj = {confidence_2:.2f} x {model_prob_2:.2f} + {1-confidence_2:.2f} x {market2.yes_price:.2f}")
    print(f"    adj = {confidence_2 * model_prob_2:.3f} + {(1-confidence_2) * market2.yes_price:.3f}")
    print(f"    Adjusted probability: {adj_prob_2:.1%}")
    print(f"    Adjusted edge: {adj_edge_2:.1%}")
    print(f"    Kelly bet size: {kelly_2:.1%} of bankroll")

    trader.add_prediction(market2, adj_prob_2, confidence_2)
    print("\n  [OK] Prediction added!")

    # ============================================================
    # PREDICTION 3: Fed Decision (NO edge - should be filtered)
    # ============================================================
    print("\n" + "-" * 50)
    print("ADDING PREDICTION 3: Fed Decision (Testing edge filter)")
    print("-" * 50)

    market3 = Market(
        ticker="FED-JAN26-NOCHANGE",
        title="Fed Holds Rates in January",
        platform=Platform.KALSHI,
        yes_price=0.92,
        no_price=0.10,
        volume=50000,
        category="FED"
    )

    print(f"  Market: {market3.title}")
    print(f"  YES price: ${market3.yes_price:.2f}")

    model_prob_3 = 0.93  # Only 1% edge
    confidence_3 = 0.80

    adj_prob_3, adj_edge_3, kelly_3 = ConfidenceAdjustedModel.calculate_confidence_adjusted_kelly(
        model_prob_3, market3.yes_price, confidence_3, config.kelly_fraction
    )

    print(f"  Your model probability: {model_prob_3:.0%}")
    print(f"  Adjusted edge: {adj_edge_3:.1%}")
    print(f"  Min edge threshold: {config.min_ev:.1%}")

    if adj_edge_3 < config.min_ev:
        print(f"\n  [FILTERED] Edge too small! Not adding prediction.")
    else:
        trader.add_prediction(market3, adj_prob_3, confidence_3)
        print("\n  [OK] Prediction added!")

    # ============================================================
    # GET RECOMMENDATIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 70)

    recommendations = trader.get_recommendations()
    trader.print_recommendations()

    # ============================================================
    # MONTE CARLO ANALYSIS
    # ============================================================
    print("\n" + "-" * 50)
    print("MONTE CARLO RISK ANALYSIS")
    print("-" * 50)

    print("\nSimulating 100 similar bets across 10,000 scenarios...")

    mc_results = trader.run_monte_carlo_analysis()
    if "error" not in mc_results:
        print(f"\n  Starting bankroll: ${config.bankroll:.2f}")
        print(f"\n  SIMULATION RESULTS (after 100 bets):")
        print(f"    Mean outcome: ${mc_results['mean_final_bankroll']:.2f}")
        print(f"    Median outcome: ${mc_results['median_final_bankroll']:.2f}")
        print(f"    Std deviation: ${mc_results['std_final_bankroll']:.2f}")
        print(f"\n  PERCENTILES:")
        print(f"    5th (bad luck):  ${mc_results['percentile_5']:.2f}")
        print(f"    25th:            ${mc_results['percentile_25']:.2f}")
        print(f"    75th:            ${mc_results['percentile_75']:.2f}")
        print(f"    95th (good luck): ${mc_results['percentile_95']:.2f}")
        print(f"\n  RISK:")
        print(f"    Probability of ruin: {mc_results['ruin_probability']:.2%}")
        print(f"    Max drawdown (avg): {mc_results['mean_max_drawdown']:.1%}")
    else:
        print(f"  Error: {mc_results['error']}")

    # ============================================================
    # EXECUTION (SIMULATED)
    # ============================================================
    print("\n" + "-" * 50)
    print("EXECUTION (SIMULATED)")
    print("-" * 50)

    print("\nLive trading is DISABLED for safety.")
    print("Simulating execution of recommendations...\n")

    results = trader.execute_recommendations()
    for r in results:
        status = r.get('status', 'ERROR')
        market = r.get('market', 'Unknown')
        side = r.get('side', '?').upper()
        contracts = r.get('contracts', 0)
        price = r.get('price', 0)
        cost = r.get('cost', 0)

        print(f"  [{status}] {market}")
        print(f"           Side: {side}, Contracts: {contracts}, Price: ${price:.2f}, Cost: ${cost:.2f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)

    print(f"\n  Predictions entered: {len(trader.predictions)}")
    print(f"  Recommendations generated: {len(recommendations)}")
    print(f"\n  POSITIONS TO TAKE:")

    total_cost = 0
    total_ev = 0
    for i, pos in enumerate(recommendations, 1):
        print(f"\n    {i}. {pos.market.ticker}")
        print(f"       {pos.side.value.upper()} {pos.contracts} contracts @ ${pos.entry_price:.2f}")
        print(f"       Cost: ${pos.cost:.2f}, Expected profit: ${pos.expected_profit:.2f}")
        total_cost += pos.cost
        total_ev += pos.expected_profit

    print(f"\n  TOTALS:")
    print(f"    Capital to deploy: ${total_cost:.2f} ({total_cost/config.bankroll:.1%} of bankroll)")
    print(f"    Expected profit: ${total_ev:.2f}")
    print(f"    Remaining cash: ${config.bankroll - total_cost:.2f}")

    print(f"\n  NEXT STEPS:")
    print(f"    1. Review recommendations carefully")
    print(f"    2. Check live prices have not moved significantly")
    print(f"    3. When ready: set config.live_trading_enabled = True")
    print(f"    4. Execute with trader.execute_recommendations()")

    print("\n" + "=" * 70)
    print("END OF INTERACTIVE SESSION")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
