# KASE Vision v6

KASE Vision — educational analytics platform for Kazakhstan Stock Exchange (KASE).

## Features
- KASE public market monitor with automatic refresh
- Historical price and monthly-return analysis
- Portfolio analytics based on 60,000 feasible portfolios
- Minimum-risk, maximum-Sharpe and equal-weight comparisons
- Interactive portfolio frontier
- Portfolio variant explorer

## Method
The analytics engine uses synchronized price histories, monthly returns, annualized mean returns and covariance, non-negative portfolio weights summing to 1, and Sharpe ratio with a zero risk-free rate for the educational model.

The demo dataset contains 24 months and uses a fixed random seed of 42.

## Market data note
The public monitor reads published KASE investor pages and refreshes automatically. It is **not a licensed exchange real-time feed**. True exchange real-time data requires the appropriate KASE market-data access (for example FIX/FAST).

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

> This project is an educational/research simulator and does not place broker orders or provide personalized investment advice.
