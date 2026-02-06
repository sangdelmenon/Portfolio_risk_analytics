import numpy as np
import pandas as pd
import time

class MonteCarloSimulator:
    """
    Runs Monte Carlo simulations to project portfolio returns.
    """
    def __init__(self, returns_data: pd.DataFrame, weights: np.ndarray):
        """
        Initializes the simulator.

        Args:
            returns_data: DataFrame of historical daily returns for each asset.
            weights: Numpy array of portfolio weights.
        """
        self.returns_data = returns_data
        self.weights = weights
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.simulation_results = None

    def run_simulation(self, n_simulations: int, time_horizon: int, initial_investment: float) -> np.ndarray:
        """
        Runs the Monte Carlo simulation.

        Args:
            n_simulations: The number of simulations to run.
            time_horizon: The number of trading days to project.
            initial_investment: The initial investment amount.

        Returns:
            A numpy array of the final portfolio values for each simulation.
        """
        print(f"Running {n_simulations} Monte Carlo simulations...")
        start_time = time.time()

        # Generate random returns
        random_returns = np.random.multivariate_normal(
            mean=self.mean_returns,
            cov=self.cov_matrix,
            size=(n_simulations, time_horizon)
        )

        # Calculate daily portfolio returns
        portfolio_returns = np.dot(random_returns, self.weights)

        # Calculate cumulative returns
        compounded_returns = np.cumprod(1 + portfolio_returns, axis=1)
        final_portfolio_values = initial_investment * compounded_returns[:, -1]
        
        self.simulation_results = final_portfolio_values
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        print(f"✓ Simulation completed in {latency:.2f}ms")
        print(f"  Average latency per simulation: {latency/n_simulations:.4f}ms")

        return final_portfolio_values

    def get_percentile_outcomes(self) -> dict:
        """
        Calculates percentile outcomes from the simulation results.

        Returns:
            A dictionary with percentile outcomes.
        """
        if self.simulation_results is None:
            raise ValueError("Simulation has not been run yet.")
            
        return {
            "5th percentile": np.percentile(self.simulation_results, 5),
            "25th percentile": np.percentile(self.simulation_results, 25),
            "50th percentile": np.percentile(self.simulation_results, 50),
            "75th percentile": np.percentile(self.simulation_results, 75),
            "95th percentile": np.percentile(self.simulation_results, 95),
        }

    def calculate_probability_of_loss(self, initial_investment: float) -> dict:
        """
        Calculates the probability of loss and expected loss.

        Returns:
            A dictionary with probability of loss and expected loss.
        """
        if self.simulation_results is None:
            raise ValueError("Simulation has not been run yet.")
            
        loss_scenarios = self.simulation_results[self.simulation_results < initial_investment]
        prob_loss = len(loss_scenarios) / len(self.simulation_results)
        expected_loss = np.mean(initial_investment - loss_scenarios) if prob_loss > 0 else 0
        
        return {
            "Probability of Loss": prob_loss,
            "Expected Loss (given loss)": expected_loss,
        }

    def calculate_expected_return(self, initial_investment: float) -> dict:
        """
        Calculates the expected return and probability of gain.

        Returns:
            A dictionary with expected return and probability of gain.
        """
        if self.simulation_results is None:
            raise ValueError("Simulation has not been run yet.")

        expected_final_value = np.mean(self.simulation_results)
        expected_return_pct = (expected_final_value / initial_investment) - 1
        prob_gain = 1 - self.calculate_probability_of_loss(initial_investment)['Probability of Loss']
        
        return {
            "Expected Return": expected_return_pct,
            "Probability of Gain": prob_gain,
        }

if __name__ == '__main__':
    # Example Usage (uses synthetic data)
    tickers = ['AAPL', 'MSFT']
    returns_data = pd.DataFrame(np.random.randn(252, 2) / 100, columns=tickers)
    weights = np.array([0.5, 0.5])
    initial_investment = 100000

    simulator = MonteCarloSimulator(returns_data, weights)
    results = simulator.run_simulation(10000, 252, initial_investment)

    print("\nPercentile Outcomes:")
    percentiles = simulator.get_percentile_outcomes()
    for p, v in percentiles.items():
        print(f"  {p}: ${v:,.2f}")

    print("\nProbability Analysis:")
    prob_analysis = simulator.calculate_probability_of_loss(initial_investment)
    print(f"  Probability of Loss: {prob_analysis['Probability of Loss']:.2%}")
    print(f"  Expected Loss (given loss): ${prob_analysis['Expected Loss (given loss)']:,.2f}")
    
    print("\nExpected Return Analysis:")
    exp_return = simulator.calculate_expected_return(initial_investment)
    print(f"  Expected Return: {exp_return['Expected Return']:.2%}")
    print(f"  Probability of Gain: {exp_return['Probability of Gain']:.2%}")
