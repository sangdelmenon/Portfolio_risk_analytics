
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_fetcher import PortfolioDataFetcher
from monte_carlo import MonteCarloSimulator
from risk_metrics import RiskMetricsCalculator

class PortfolioRiskAnalytics:
    """
    Orchestrates the entire portfolio risk analysis workflow.
    """
    def __init__(self, tickers: list, weights: dict = None, initial_investment: float = 100000):
        self.tickers = tickers
        self.initial_investment = initial_investment
        if weights is None:
            self.weights = np.array([1/len(tickers)] * len(tickers))
        else:
            if abs(sum(weights.values()) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = np.array([weights[t] for t in self.tickers])
        
        self.data_fetcher = PortfolioDataFetcher(self.tickers)
        self.returns_data = None
        self.simulation_results = None
        self.risk_metrics = {}

    def fetch_data(self, start_date: str = None, end_date: str = None):
        print("="*70)
        print("STEP 1: FETCHING HISTORICAL DATA")
        print("="*70)
        self.data_fetcher.fetch_data(start_date, end_date)
        self.returns_data = self.data_fetcher.calculate_returns()
        print(f"✓ Returns calculated: {len(self.returns_data)} days")

    def run_monte_carlo(self, n_simulations: int = 10000, time_horizon: int = 252):
        if self.returns_data is None:
            self.fetch_data()
        
        print("\n" + "="*70)
        print("STEP 2: MONTE CARLO SIMULATION")
        print("="*70)
        
        self.simulator = MonteCarloSimulator(self.returns_data, self.weights)
        self.simulation_results = self.simulator.run_simulation(n_simulations, time_horizon, self.initial_investment)

    def calculate_risk_metrics(self):
        if self.simulation_results is None:
            self.run_monte_carlo()
            
        print("\n" + "="*70)
        print("STEP 3: RISK METRICS CALCULATION")
        print("="*70)
        
        risk_calculator = RiskMetricsCalculator(self.simulation_results, self.initial_investment)
        
        print("\nValue at Risk (VaR) and Conditional VaR (CVaR):")
        print(f"{'Confidence Level':>18} {'VaR ($)':>12} {'VaR (%)':>10} {'CVaR ($)':>12} {'CVaR (%)':>10}")
        for cl in [0.90, 0.95, 0.99]:
            var = risk_calculator.calculate_var(cl)
            cvar = risk_calculator.calculate_cvar(cl)
            self.risk_metrics[f'VaR_{cl}'] = var
            self.risk_metrics[f'CVaR_{cl}'] = cvar
            print(f"{cl*100:17.0f}% {var['VaR']:12,.2f} {var['VaR %']:9.2%} {cvar['CVaR']:12,.2f} {cvar['CVaR %']:9.2%}")
            
        self.risk_metrics['Sharpe Ratio'] = risk_calculator.calculate_sharpe_ratio()
        print(f"\nSharpe Ratio: {self.risk_metrics['Sharpe Ratio']:.4f}")

    def generate_visualizations(self):
        print("\n" + "="*70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Portfolio Risk Analysis Dashboard', fontsize=24, y=0.95)
        
        # 1. Price History
        ax = axes[0, 0]
        norm_prices = self.data_fetcher.raw_data.div(self.data_fetcher.raw_data.iloc[0]).mul(100)
        norm_prices.plot(ax=ax, legend=True)
        ax.set_title('Normalized Price History (Base 100)')
        ax.set_ylabel('Normalized Price')

        # 2. Daily Returns
        ax = axes[0, 1]
        self.returns_data.plot(kind='hist', bins=50, alpha=0.5, ax=ax)
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Daily Return')

        # 3. Correlation Matrix
        ax = axes[0, 2]
        sns.heatmap(self.returns_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Asset Correlation Matrix')

        # 4. Monte Carlo Simulation
        ax = axes[1, 0]
        sns.histplot(self.simulation_results, bins=100, ax=ax, kde=True)
        ax.axvline(self.initial_investment, color='green', linestyle='--', label='Initial Investment')
        ax.axvline(np.mean(self.simulation_results), color='red', linestyle='--', label='Expected Value')
        ax.set_title(f'Monte Carlo Simulation ({len(self.simulation_results)} paths)')
        ax.set_xlabel('Final Portfolio Value')
        ax.legend()
        
        # 5. VaR Visualization
        ax = axes[1, 1]
        losses = self.initial_investment - self.simulation_results
        sns.histplot(losses, bins=100, ax=ax, kde=True, color='red')
        var_95 = self.risk_metrics['VaR_0.95']['VaR']
        cvar_95 = self.risk_metrics['CVaR_0.95']['CVaR']
        ax.axvline(var_95, color='orange', linestyle='--', label=f'VaR (95%): ${var_95:,.2f}')
        ax.axvline(cvar_95, color='darkred', linestyle='--', label=f'CVaR (95%): ${cvar_95:,.2f}')
        ax.set_title('Distribution of Losses & VaR')
        ax.set_xlabel('Loss Amount')
        ax.legend()

        # 6. Risk-Return Profile
        ax = axes[1, 2]
        summary_stats = self.data_fetcher.get_summary_statistics()
        ax.scatter(summary_stats['Std Dev'] * np.sqrt(252), summary_stats['Mean'] * 252)
        for i, txt in enumerate(self.tickers):
            ax.annotate(txt, (summary_stats.loc[txt]['Std Dev'] * np.sqrt(252), summary_stats.loc[txt]['Mean'] * 252))
        ax.set_title('Risk-Return Profile')
        ax.set_xlabel('Annual Volatility (Risk)')
        ax.set_ylabel('Annual Return')

        # 7. Cumulative Returns
        ax = axes[2, 0]
        cumulative_returns = (1 + self.returns_data).cumprod()
        cumulative_returns.plot(ax=ax, legend=False)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Growth of $1')

        # 8. Portfolio Allocation
        ax = axes[2, 1]
        ax.pie(self.weights, labels=self.tickers, autopct='%1.1f%%', startangle=90)
        ax.set_title('Portfolio Allocation')
        ax.axis('equal')

        # 9. Summary Table
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = (
            f"Initial Investment: ${self.initial_investment:,.2f}\n"
            f"Expected Final Value: ${np.mean(self.simulation_results):,.2f}\n"
            f"Expected Return: {(np.mean(self.simulation_results)/self.initial_investment-1):.2%}\n"
            f"VaR (95%): ${self.risk_metrics['VaR_0.95']['VaR']:,.2f}\n"
            f"CVaR (95%): ${self.risk_metrics['CVaR_0.95']['CVaR']:,.2f}\n"
            f"Sharpe Ratio: {self.risk_metrics['Sharpe Ratio']:.4f}\n"
            f"Simulations: {len(self.simulation_results)}\n"
            f"Time Horizon: 1 Year (252 days)"
        )
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('portfolio_risk_analysis.png', dpi=300)
        print("✓ Visualizations saved as 'portfolio_risk_analysis.png'")

    def run_complete_analysis(self):
        self.fetch_data()
        self.run_monte_carlo()
        self.calculate_risk_metrics()
        self.generate_visualizations()
        
        print("\n" + "="*70)
        print("PORTFOLIO RISK ANALYTICS REPORT")
        print("="*70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Investment: ${self.initial_investment:,.2f}")
        print(f"Analysis Period: {self.data_fetcher.raw_data.index.min().strftime('%Y-%m-%d')} to {self.data_fetcher.raw_data.index.max().strftime('%Y-%m-%d')}")
        print(f"Portfolio Composition: {', '.join(self.tickers)}")
        print("="*70)
        print("\nANALYSIS COMPLETE!")


if __name__ == '__main__':
    portfolio = PortfolioRiskAnalytics(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        weights={'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.20, 'AMZN': 0.15, 'NVDA': 0.15},
    )
    portfolio.run_complete_analysis()
