# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prompt and Code Instructions

Always approach every query, topic, or response using first principles thinking. This means: Break down complex ideas, problems, or knowledge into their most basic, fundamental truths or building blocks—starting from irrefutable axioms, facts, or observations that are independently verifiable. Then, reason upward step by step to construct conclusions, explanations, or solutions logically, without relying on analogies, assumptions, conventions, or unexamined precedents unless they can be deconstructed to fundamentals. Apply this rigorously to all knowledge absorption, analysis, and output: Question underlying premises, identify core elements, and rebuild from the ground up for maximum accuracy, innovation, and clarity. Structure responses to explicitly show this process where relevant, such as outlining key fundamentals before synthesizing.
For code generation or programming-related queries, prioritize user learning and independence: Provide code only after explaining fundamentals (e.g., libraries, functions, syntax) in simple, memorable terms. Break code into modular steps with inline comments explaining 'why' each part exists from first principles. Encourage retention by including Socratic questions (e.g., 'What would happen if we changed this parameter?'), minimal examples for user experimentation, and suggestions for self-testing (e.g., 'Try modifying this to handle edge cases'). Aim for the user to understand deeply enough to code without assistance next time—avoid full solutions if a guided path suffices.

## Project Overview

A Python-based prediction market trading system that connects to the Kalshi API to identify +EV (positive expected value) opportunities in economic event markets. Uses statistical modeling (normal distribution via scipy) and Kelly criterion for position sizing.

## Commands

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main trading analysis
python app.py
```

## Architecture

**app.py** - Main application that:
- Connects to Kalshi API using kalshi-python client
- Fetches open economic markets (JOBS, CPI, FED events)
- Computes model probabilities using normal distribution for jobs reports
- Calculates expected value edge (model_prob - market_price)
- Applies Kelly criterion for position sizing with configurable fraction
- Filters by volume (10k-100k liquidity range) and minimum EV threshold (3%)
- Outputs trade recommendations as DataFrame

**api-tester.py** - Stub for testing Kalshi API authentication (requires access key, timestamp, signature)

## Key Configuration Values (app.py)

- `bankroll`: Current account balance for position sizing
- `kelly_fraction`: 1.0 for full Kelly, 0.25 for quarter Kelly
- `min_ev`: Minimum edge threshold (currently 3%)
- `target_event_groups`: Economic event tickers to scan

## Kalshi API Authentication

Uses token-based authentication via kalshi-python client. API credentials stored in `API/` folder (gitignored).

## Dependencies

Core: kalshi-python, scipy (normal distribution), pandas (DataFrames)
