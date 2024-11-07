import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TeslaRiskAnalysis:
    def __init__(self, file_path="../Data/TSLA_historical_data.csv"):
        self.file_path = file_path
        self.data = self.load_stock_data()
    
    def load_stock_data(self):
        """
        Load Tesla stock data from the provided CSV file.
        """
        stock_data = pd.read_csv(self.file_path, parse_dates=['Date'])
        stock_data.set_index('Date', inplace=True)
        return stock_data

    def calculate_daily_returns(self):
        """
        Calculate the daily returns for Tesla stock.
        """
        self.data['Daily Return'] = self.data['Adj Close'].pct_change()
        return self.data['Daily Return'].dropna()

    def calculate_var(self, confidence_interval=0.95):
        """
        Calculate Value at Risk (VaR) at the given confidence level for Tesla.
        """
        daily_returns = self.calculate_daily_returns()
        
        # Calculate VaR using the historical quantile method
        var = daily_returns.quantile(1 - confidence_interval)
        return var

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        """
        Calculate the Sharpe Ratio for Tesla stock.
        """
        daily_returns = self.calculate_daily_returns()
        
        # Calculate average return and standard deviation of daily returns
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # Sharpe ratio formula, annualizing the risk-free rate
        sharpe_ratio = (avg_daily_return - risk_free_rate / 252) / std_daily_return
        return sharpe_ratio

    def plot_distribution_of_returns(self):
        """
        Plot the distribution of Tesla's daily returns.
        """
        daily_returns = self.calculate_daily_returns()
        
        plt.figure(figsize=(10, 6))
        plt.hist(daily_returns, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of Tesla Daily Returns")
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def summarize_risk_analysis(self):
        """
        Summarize the key insights of Tesla's stock risk analysis: VaR, Sharpe Ratio, and distribution.
        """
        print(f"Risk Analysis for TSLA")
        
        # Value at Risk (VaR) at 95% confidence level
        var_95 = self.calculate_var(confidence_interval=0.95)
        print(f"Value at Risk (VaR) at 95% confidence level: {var_95 * 100:.2f}%")
        
        # Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio()
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot distribution of returns
        self.plot_distribution_of_returns()

        # Summary of daily returns
        daily_returns = self.calculate_daily_returns()
        print("\nSummary of Daily Returns:")
        print(daily_returns.describe())
