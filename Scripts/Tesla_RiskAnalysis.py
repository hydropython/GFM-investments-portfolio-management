import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

class TeslaRiskAnalysis:
    def __init__(self, file_path="../Data/TSLA_historical_data.csv"):
        self.file_path = file_path
        self.data = self.load_stock_data()
    
    def load_stock_data(self):
        """
        Load Tesla stock data from the provided CSV file.
        """
        logging.info(f"Loading Tesla stock data from: {self.file_path}")
        try:
            # Try loading the data with 'Date' as the column name for date
            stock_data = pd.read_csv(self.file_path)
            print(f"Columns in the CSV file: {stock_data.columns}")
            
            if 'Date' in stock_data.columns:
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                logging.info(f"Stock data loaded successfully. Data shape: {stock_data.shape}")
            else:
                logging.error("The 'Date' column is missing in the dataset. Please check the column name.")
                raise ValueError("The 'Date' column is missing in the dataset.")
                
            return stock_data
        
        except FileNotFoundError:
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except ValueError as e:
            logging.error(str(e))
            raise

    def calculate_daily_returns(self):
        """
        Calculate the daily returns for Tesla stock.
        """
        logging.info("Calculating daily returns...")
        self.data['Daily Return'] = self.data['Adj Close'].pct_change()
        return self.data['Daily Return'].dropna()

    def calculate_var(self, confidence_interval=0.95):
        """
        Calculate Value at Risk (VaR) at the given confidence level for Tesla.
        """
        logging.info(f"Calculating VaR at {confidence_interval*100}% confidence level...")
        daily_returns = self.calculate_daily_returns()
        
        # Calculate VaR using the historical quantile method
        var = daily_returns.quantile(1 - confidence_interval)
        logging.info(f"Calculated VaR: {var:.4f}")
        return var

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        """
        Calculate the Sharpe Ratio for Tesla stock.
        """
        logging.info("Calculating Sharpe Ratio...")
        daily_returns = self.calculate_daily_returns()
        
        # Calculate average return and standard deviation of daily returns
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # Sharpe ratio formula, annualizing the risk-free rate
        sharpe_ratio = (avg_daily_return - risk_free_rate / 252) / std_daily_return
        logging.info(f"Calculated Sharpe Ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio

    def plot_distribution_of_returns(self):
        """
        Plot the distribution of Tesla's daily returns.
        """
        logging.info("Plotting distribution of returns...")
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
        logging.info("Summarizing risk analysis...")
        
        # Value at Risk (VaR) at 95% confidence level
        var_95 = self.calculate_var(confidence_interval=0.95)
        logging.info(f"Value at Risk (VaR) at 95% confidence level: {var_95 * 100:.2f}%")
        
        # Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio()
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot distribution of returns
        self.plot_distribution_of_returns()

        # Summary of daily returns
        daily_returns = self.calculate_daily_returns()
        logging.info("\nSummary of Daily Returns:")
        logging.info(daily_returns.describe())
