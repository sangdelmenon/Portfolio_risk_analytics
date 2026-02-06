import numpy as np
from scipy.stats import norm

class RiskMetricsCalculator:
    """
    Calculates various financial risk metrics for a portfolio.
    """
    def __init__(self, simulation_results: np.ndarray, initial_investment: float):
        """
        Initializes the risk metrics calculator.

        Args:
            simulation_results: Numpy array of final portfolio values from Monte Carlo.
            initial_investment: The initial investment amount.
        """
        self.simulation_results = simulation_results
        self.initial_investment = initial_investment
        self.losses = initial_investment - simulation_results

    def calculate_var(self, confidence_level: float = 0.95) -> dict:
        """
        Calculates Value at Risk (VaR) from simulation results.

        Args:
            confidence_level: The confidence level for VaR (e.g., 0.95 for 95%).

        Returns:
            A dictionary with VaR value and percentage.
        """
        var_value = np.percentile(self.losses, confidence_level * 100)
        var_percent = var_value / self.initial_investment
        return {"VaR": var_value, "VaR %": var_percent}

    def calculate_cvar(self, confidence_level: float = 0.95) -> dict:
        """
        Calculates Conditional VaR (CVaR) or Expected Shortfall.

        Args:
            confidence_level: The confidence level for CVaR.

        Returns:
            A dictionary with CVaR value and percentage.
        """
        var_value = self.calculate_var(confidence_level)['VaR']
        tail_losses = self.losses[self.losses >= var_value]
        cvar_value = np.mean(tail_losses)
        cvar_percent = cvar_value / self.initial_investment
        return {"CVaR": cvar_value, "CVaR %": cvar_percent}

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculates the Sharpe Ratio of the portfolio.

        Args:
            risk_free_rate: The annual risk-free rate.

        Returns:
            The Sharpe Ratio.
        """
        expected_return = (np.mean(self.simulation_results) / self.initial_investment) - 1
        portfolio_volatility = np.std(self.simulation_results / self.initial_investment - 1) * np.sqrt(252) # Assuming daily returns are used
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility
        return sharpe_ratio

    def calculate_portfolio_greeks(self, S, K, T, r, sigma) -> dict:
        """
        Calculates the Greeks for a single option.
        This is a placeholder for a portfolio of options.

        Args:
            S: Stock price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            A dictionary with the Greeks.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }

if __name__ == '__main__':
    # Example Usage (uses synthetic data)
    initial_investment = 100000
    # Simulate some results: 10% average return, 15% volatility
    simulated_returns = np.random.normal(1.1, 0.15, 10000)
    final_values = initial_investment * simulated_returns

    risk_calculator = RiskMetricsCalculator(final_values, initial_investment)

    # VaR and CVaR
    for cl in [0.90, 0.95, 0.99]:
        var = risk_calculator.calculate_var(cl)
        cvar = risk_calculator.calculate_cvar(cl)
        print(f"\nConfidence Level: {cl*100}%")
        print(f"  VaR: ${var['VaR']:,.2f} ({var['VaR %']:.2%})")
        print(f"  CVaR: ${cvar['CVaR']:,.2f} ({cvar['CVaR %']:.2%})")

    # Sharpe Ratio
    sharpe = risk_calculator.calculate_sharpe_ratio()
    print(f"\nSharpe Ratio: {sharpe:.4f}")
    
    # Greeks
    greeks = risk_calculator.calculate_portfolio_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
    print("\nPortfolio Greeks (example for one option):")
    for k, v in greeks.items():
        print(f"  {k}: {v:.4f}")
