# Portfolio Risk Analytics with Monte Carlo Simulation

This project implements a portfolio risk analytics tool using Python. It fetches historical stock data, runs Monte Carlo simulations to project future returns, and calculates various risk metrics such as Value at Risk (VaR) and Conditional VaR (CVaR).

## Features

- Fetches 5 years of historical stock data for any portfolio.
- Runs 10,000 Monte Carlo simulations in under a second.
- Calculates VaR and CVaR at 90%, 95%, and 99% confidence levels.
- Computes portfolio Greeks (Delta, Gamma, Vega, Theta, Rho).
- Generates a comprehensive 9-panel visualization dashboard.
- Produces detailed risk analysis reports.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the analysis:**
    ```bash
    python portfolio_risk_analytics/main.py
    ```

    This will generate a `portfolio_risk_analysis.png` file with the results.

## Sample Output

![Sample Output](portfolio_risk_analysis.png)
