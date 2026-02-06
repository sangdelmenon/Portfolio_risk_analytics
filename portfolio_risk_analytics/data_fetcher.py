
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil

def clear_yfinance_cache():
    """Clears the yfinance cache."""
    try:
        # Manually construct the cache path for macOS
        cache_dir = os.path.expanduser('~/Library/Caches/yfinance')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("✓ yfinance cache cleared successfully.")
    except Exception as e:
        print(f"Could not clear yfinance cache: {e}")

class PortfolioDataFetcher:
    """
    Fetches and preprocesses historical stock data for a given portfolio.
    """
    def __init__(self, tickers: list[str]):
        """
        Initializes the data fetcher.

        Args:
            tickers: A list of stock tickers.
        """
        self.tickers = tickers
        self.raw_data = None
        self.returns_data = None

    def fetch_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Downloads historical stock prices from Yahoo Finance.

        Args:
            start_date: The start date for the data in 'YYYY-MM-DD' format.
                        Defaults to 5 years ago.
            end_date: The end date for the data in 'YYYY-MM-DD' format.
                      Defaults to today.

        Returns:
            A pandas DataFrame with the adjusted close prices of the stocks.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

        print(f"Fetching data for {len(self.tickers)} tickers from {start_date} to {end_date}...")
        self.raw_data = yf.download(self.tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
        print(f"✓ Data fetched successfully: {len(self.raw_data)} trading days")
        return self.raw_data

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculates the daily percentage returns of the stocks.

        Returns:
            A pandas DataFrame with the daily returns.
        """
        if self.raw_data is None:
            raise ValueError("Data has not been fetched yet. Call fetch_data() first.")
        self.returns_data = self.raw_data.pct_change().dropna()
        return self.returns_data

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculates summary statistics for the daily returns.

        Returns:
            A pandas DataFrame with the summary statistics.
        """
        if self.returns_data is None:
            self.calculate_returns()
        
        stats = pd.DataFrame(index=self.tickers)
        stats['Mean'] = self.returns_data.mean()
        stats['Std Dev'] = self.returns_data.std()
        stats['Sharpe Ratio'] = (stats['Mean'] / stats['Std Dev']) * np.sqrt(252)
        return stats

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculates the correlation matrix of the daily returns.

        Returns:
            A pandas DataFrame with the correlation matrix.
        """
        if self.returns_data is None:
            self.calculate_returns()
        return self.returns_data.corr()


if __name__ == '__main__':
    # Example Usage
    clear_yfinance_cache()
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    fetcher = PortfolioDataFetcher(tickers)
    
    # Fetch data
    price_data = fetcher.fetch_data()
    print("\nPrice Data (first 5 rows):")
    print(price_data.head())

    # Calculate and display returns
    returns = fetcher.calculate_returns()
    print("\nReturns Data (first 5 rows):")
    print(returns.head())

    # Get summary statistics
    summary_stats = fetcher.get_summary_statistics()
    print("\nSummary Statistics:")
    print(summary_stats)

    # Get correlation matrix
    correlation_matrix = fetcher.get_correlation_matrix()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
