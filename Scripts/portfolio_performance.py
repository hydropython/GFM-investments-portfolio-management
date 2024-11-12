import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, tsla_path, bnd_path, spy_path, image_folder='../Images/'):
        """
        Initializes the PortfolioOptimizer with forecasted price data for each asset.

        :param tsla_path: Path to the Tesla forecast CSV file.
        :param bnd_path: Path to the BND forecast CSV file.
        :param spy_path: Path to the SPY forecast CSV file.
        :param image_folder: Folder path to save images.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.logger.info("Initializing PortfolioOptimizer.")

        # Load forecasted data and check columns
        try:
            self.tsla = self.load_and_check_data(tsla_path, 'TSLA')
            self.bnd = self.load_and_check_data(bnd_path, 'BND')
            self.spy = self.load_and_check_data(spy_path, 'SPY')
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

        # Combine the forecasted data into one DataFrame
        self.forecast_df = pd.DataFrame({
            'TSLA': self.tsla,
            'BND': self.bnd,
            'SPY': self.spy
        }).dropna()

        # Calculate daily returns
        self.daily_returns = self.forecast_df.pct_change().dropna()
        self.weights = np.array([1/3, 1/3, 1/3])  # Initial equal weights for the assets
        self.image_folder = image_folder

        self.logger.info("PortfolioOptimizer initialization complete.")

    def load_and_check_data(self, file_path, asset_name):
        """
        Loads data from a CSV file, renaming 'Forecast' to 'Close' if necessary.

        :param file_path: Path to the CSV file.
        :param asset_name: Name of the asset (for error messaging).
        :return: Series with the specified column data.
        """
        self.logger.info(f"Loading data from {file_path} for {asset_name}.")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Rename 'Forecast' to 'Close' if 'Close' is missing and 'Forecast' exists
        if 'Close' not in data.columns and 'Forecast' in data.columns:
            data = data.rename(columns={'Forecast': 'Close'})

        # Check if the 'Close' column is available
        if 'Close' not in data.columns:
            raise KeyError(f"'{file_path}' for {asset_name} is missing the 'Close' column. Available columns: {data.columns.tolist()}")
        
        self.logger.info(f"Data for {asset_name} loaded successfully.")
        return data['Close']
    
    def optimize_portfolio(self):
        """
        Optimize the portfolio weights to maximize the Sharpe ratio.
        """
        self.logger.info("Starting portfolio optimization.")
        
        # Define the negative Sharpe ratio function
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.daily_returns.mean()) * 252  # Annualized return
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.daily_returns.cov() * 252, weights)))  # Annualized volatility
            return -portfolio_return / portfolio_volatility  # Negative Sharpe ratio
        
        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Bounds: Each weight is between 0 and 1
        bounds = tuple((0, 1) for _ in range(len(self.weights)))
        
        # Perform the optimization
        optimized = minimize(negative_sharpe, self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.optimized_weights = optimized.x
        
        # Portfolio stats
        self.portfolio_return = np.dot(self.optimized_weights, self.daily_returns.mean()) * 252
        self.portfolio_volatility = np.sqrt(np.dot(self.optimized_weights.T, np.dot(self.daily_returns.cov() * 252, self.optimized_weights)))
        self.sharpe_ratio = self.portfolio_return / self.portfolio_volatility

        self.logger.info("Portfolio optimization complete.")
        self.logger.info(f"Optimized weights: {self.optimized_weights}")
        self.logger.info(f"Expected annual return: {self.portfolio_return}")
        self.logger.info(f"Annual volatility: {self.portfolio_volatility}")
        self.logger.info(f"Sharpe ratio: {self.sharpe_ratio}")

    def visualize_portfolio(self):
        """
        Visualizes the cumulative returns of the optimized portfolio.
        """
        self.logger.info("Visualizing optimized portfolio.")
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.daily_returns.dot(self.optimized_weights)).cumprod()
        
        plt.figure(figsize=(14, 8))
        plt.plot(cumulative_returns, label='Optimized Portfolio', color='green')
        plt.title('Cumulative Returns of Optimized Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        
        image_path = f"{self.image_folder}/optimized_portfolio_cumulative_returns.png"
        plt.savefig(image_path)
        self.logger.info(f"Portfolio cumulative returns plot saved to {image_path}.")
        plt.show()

    def portfolio_summary(self):
        """
        Prints and returns a summary of the portfolio performance.
        """
        summary = {
            'Optimized Weights': self.optimized_weights,
            'Expected Annual Return': self.portfolio_return,
            'Annual Volatility (Risk)': self.portfolio_volatility,
            'Sharpe Ratio': self.sharpe_ratio
        }
        
        self.logger.info("Portfolio Summary:")
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")
        
        return summary
    
    def create_portfolio_subplot(self):
        """
        Creates a journal-style subplot for Expected Annual Return, Optimized Weights, and Annual Volatility.
        """
        self.logger.info("Creating portfolio subplot.")
        
        expected_return = self.portfolio_return
        annual_volatility = self.portfolio_volatility
        optimized_weights = self.optimized_weights
        asset_labels = ['TSLA', 'BND', 'SPY']

        # Create subplots with a 1x3 layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Expected Annual Return (Bar Chart)
        axes[0].bar(['Expected Annual Return'], [expected_return], color='darkgreen')
        axes[0].set_title('Expected Annual Return', fontsize=14)
        axes[0].set_ylabel('Return', fontsize=12)
        axes[0].set_ylim(0, max(expected_return * 1.5, 0.02))

        # Plot Optimized Weights (Pie Chart)
        axes[1].pie(optimized_weights, labels=asset_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1].set_title('Optimized Weights', fontsize=14)

        # Plot Annual Volatility (Bar Chart)
        axes[2].bar(['Annual Volatility'], [annual_volatility], color='darkred')
        axes[2].set_title('Annual Volatility', fontsize=14)
        axes[2].set_ylabel('Volatility', fontsize=12)
        axes[2].set_ylim(0, max(annual_volatility * 1.5, 0.002))

        # Adjust layout for better spacing
        plt.tight_layout()

        # Display the plot
        plt.show()

        self.logger.info("Portfolio subplot created and displayed.")