import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

class FinancialDataEDA:
    def __init__(self, data=None):
        if data is None:
            self.data = self.load_data_from_csv()
        else:
            self.data = data
        self.rename_columns()
        self.handle_missing_values()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            filename='../Logs/financial_data_eda.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        logging.info("Logging setup complete.")

    def load_data_from_csv(self):
        """Load data from CSV files in the ../Data folder."""
        files = ['BND_historical_data.csv', 'SPY_historical_data.csv', 'TSLA_historical_data.csv']
        data = {}
        
        for file in files:
            file_path = os.path.join('../Data', file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ticker = file.split('_')[0]  # Extract the ticker from the file name
                data[ticker] = df
                logging.info(f"Loaded data for {ticker} from {file}")
            else:
                logging.warning(f"File {file} not found in the ../Data directory.")
        
        return data

    def rename_columns(self):
        """Renames columns in the dataset to a standard format based on the number of columns."""
        for ticker, df in self.data.items():
            logging.info(f"Columns before renaming for {ticker}: {df.columns}")
            
            if len(df.columns) == 6:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low']
            elif len(df.columns) == 7:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open']
            elif len(df.columns) == 8:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            else:
                logging.warning(f"Unexpected column count for {ticker}: {len(df.columns)}")
                continue
            
            # Clean 'Date' or 'Ticker' rows if present
            df = df[~df['Date'].isin(['Date', 'Ticker'])]
            self.data[ticker] = df  # Update the dictionary with cleaned DataFrame
            logging.info(f"Columns renamed for {ticker}: {df.columns}")

    def handle_missing_values(self):
        """
        Handle missing values in the stock data by forward filling, interpolating,
        and then dropping rows with any remaining missing values. Ensures that all
        necessary columns are numeric for interpolation.
        """
        for ticker, df in self.data.items():
            logging.info(f"Handling missing values for {ticker}")
            
            # Ensure that 'Close' and other relevant columns are numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

            # Forward fill missing values
            df.fillna(method='ffill', inplace=True)
            
            # Interpolate for numeric columns
            df.interpolate(method='linear', inplace=True)
            
            # Drop rows with any remaining missing values
            df.dropna(inplace=True)
            
            # Log the number of missing values after handling
            missing_values = df.isnull().sum()
            logging.info(f"Missing values in {ticker}: {missing_values[missing_values > 0]}")

            # Update the DataFrame with the handled missing values
            self.data[ticker] = df
            logging.info(f"Missing values handled for {ticker}.")

    def decompose_time_series(self, ticker, column='Close', period=252):
        """
        Decompose the time series into trend, seasonal, and residual components.

        Parameters:
        - ticker (str): The ticker symbol of the data to decompose.
        - column (str): The column to decompose. Default is 'Close'.
        - period (int): The period for decomposition. Default is 252 (approx. trading days in a year).
        """
        if ticker not in self.data:
            logging.error(f"The ticker '{ticker}' is not available in the dataset.")
            raise ValueError(f"The ticker '{ticker}' is not available in the dataset.")
        
        df = self.data[ticker]
        
        if column not in df.columns:
            logging.error(f"The specified column '{column}' does not exist for {ticker}.")
            raise ValueError(f"The specified column '{column}' does not exist for {ticker}.")

        # Ensure Date is set as index and sorted for time series decomposition
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        # Decompose the time series
        decomposition = seasonal_decompose(df[column], model='multiplicative', period=period)
        
        # Extract components
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()

        # Numerical output
        logging.info(f"\n{ticker} - Trend Component:\n{trend.describe()}")
        logging.info(f"{ticker} - Seasonal Component:\n{seasonal.describe()}")
        logging.info(f"{ticker} - Residual Component:\n{residual.describe()}")

        # Plot decomposition
        plt.figure(figsize=(14, 10))

        # Original Series
        plt.subplot(411)
        plt.plot(df[column], label='Original', color='blue')
        plt.title(f'{ticker} - Original Series', fontsize=14)
        plt.legend(loc='upper left')

        # Trend Component
        plt.subplot(412)
        plt.plot(trend, label='Trend', color='orange')
        plt.title('Trend', fontsize=14)
        plt.legend(loc='upper left')

        # Seasonal Component
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality', color='green')
        plt.title('Seasonality', fontsize=14)
        plt.legend(loc='upper left')

        # Residual Component
        plt.subplot(414)
        plt.plot(residual, label='Residuals', color='red')
        plt.title('Residuals', fontsize=14)
        plt.legend(loc='upper left')

        # Adjust layout and save plot
        plt.tight_layout()

        # Ensure the Images directory exists
        images_dir = '../Images'
        os.makedirs(images_dir, exist_ok=True)

        # Save the plot
        plot_filename = os.path.join(images_dir, f'{ticker}_time_series_decomposition.png')
        plt.savefig(plot_filename, dpi=300)
        logging.info(f"Plot saved as {plot_filename}")
        plt.show()

    def check_basic_statistics(self):
        """Check basic statistics of the data to understand its distribution."""
        for ticker, df in self.data.items():
            logging.info(f"Basic statistics for {ticker}:")
            logging.info(df.describe())  # Show basic statistics like mean, std, min, max, etc.
            
    def analyze_volatility(self, window=30):
        """Analyze volatility by calculating and plotting rolling mean and standard deviation in subplots."""
        fig, axes = plt.subplots(len(self.data), 2, figsize=(12, 6 * len(self.data)))
        fig.suptitle("Volatility Analysis", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Ensure 'Close' is numeric
            df['Rolling Mean'] = df['Close'].rolling(window=window).mean()  # Calculate rolling mean
            df['Rolling Std'] = df['Close'].rolling(window=window).std()  # Calculate rolling standard deviation

            # Plot the rolling mean and std dev for each stock
            ax1, ax2 = axes[i] if len(self.data) > 1 else axes
            ax1.plot(df['Rolling Mean'], label=f'{ticker} {window}-Day Mean', color='mediumseagreen')
            ax1.set_title(f'{ticker} {window}-Day Rolling Mean')
            ax1.legend()

            ax2.plot(df['Rolling Std'], label=f'{ticker} {window}-Day Std Dev', color='darkorange')
            ax2.set_title(f'{ticker} {window}-Day Std Dev')
            ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/volatility_analysis_plot.png')
        logging.info(f"Volatility analysis plot saved as 'volatility_analysis_plot.png' in ../Images/")
        plt.show()


    def plot_closing_price(self):
        """Plots the closing price over time for each stock."""
        fig, axes = plt.subplots(len(self.data), 1, figsize=(10, 6))
        if len(self.data) == 1:
            axes = [axes]

        for i, (ticker, df) in enumerate(self.data.items()):
            ax = axes[i]
            ax.plot(df.index, df['Close'], label=f'{ticker} Close', color='mediumseagreen', linewidth=2)  # Emerald Green color
            ax.set_title(f'{ticker} - Closing Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()

        plt.tight_layout()
        plt.savefig('../Images/closing_price_plot.png')
        logging.info(f"Closing price plot saved as 'closing_price_plot.png' in ../Images/")
        plt.show()

    def plot_daily_pct_change(self):
        """Plots daily percentage change for each stock."""
        fig, axes = plt.subplots(nrows=len(self.data), figsize=(10, 6 * len(self.data)))
        fig.suptitle("Daily Percentage Change", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Ensure 'Close' is numeric
            df['Daily Return'] = df['Close'].pct_change()  # Calculate daily percentage change

            # Print the daily percentage change values for each ticker
            logging.info(f"Daily percentage change for {ticker}:")
            logging.info(df[['Date', 'Daily Return']].dropna())  # Display Date and Daily Return columns

            ax = axes[i] if len(self.data) > 1 else axes
            ax.plot(df['Daily Return'], label=f'{ticker} Daily % Change', color='mediumseagreen', linewidth=2)  # Emerald Green color
            ax.set_title(f'{ticker} - Daily Percentage Change')
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily % Change')
            ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust title positioning
        plt.savefig('../Images/daily_pct_change_plot.png')
        logging.info(f"Daily percentage change plot saved as 'daily_pct_change_plot.png' in ../Images/")
        plt.show()
    def detect_outliers(self, threshold=2.5):
        """Detect outliers based on a threshold for standard deviations in returns, and plot in subplots."""
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Start logging
        logging.info(f"Starting outlier detection with a threshold of {threshold} standard deviations.")

        fig, axes = plt.subplots(len(self.data), 1, figsize=(14, 4 * len(self.data)), sharex=True)
        fig.suptitle(f"Outliers Detection (Threshold = {threshold} Std Devs)", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            logging.info(f"Processing data for {ticker}...")
            
            df['Daily Return'] = df['Close'].pct_change()  # Calculate daily returns
            mean = df['Daily Return'].mean()  # Mean of daily returns
            std_dev = df['Daily Return'].std()  # Standard deviation of daily returns

            # Identify outliers where the daily return deviates more than the threshold times the standard deviation
            df['Outlier'] = (df['Daily Return'] - mean).abs() > threshold * std_dev
            outliers = df[df['Outlier']]  # Filter out the outliers

            if not outliers.empty:
                logging.info(f"Found {len(outliers)} outliers for {ticker}.")
                # Output numerical details of the outliers (e.g., 'Close' and 'Daily Return')
                logging.info(f"Outliers for {ticker}:")
                logging.info(outliers[['Close', 'Daily Return']].to_string(index=False))  # Display outliers
            else:
                logging.info(f"No outliers detected for {ticker}.")

            # Plot Daily Return with Outliers
            ax = axes[i] if len(self.data) > 1 else axes  # Handle multiple subplots if necessary
            ax.plot(df.index, df['Daily Return'], label='Daily Return', color='mediumseagreen')  # Daily return line
            ax.scatter(outliers.index, outliers['Daily Return'], color='gold', label='Outliers', zorder=5)  # Outliers in gold
            ax.set_title(f"{ticker} - Daily Returns with Outliers")
            ax.set_ylabel("Daily % Change")
            ax.legend()
            ax.grid(True)

        plt.xlabel("Date")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for better spacing
        plot_filepath = '../Images/outliers_detection_plot.png'
        plt.savefig(plot_filepath)  # Save the plot
        logging.info(f"Outliers detection plot saved as 'outliers_detection_plot.png' in ../Images/")
        plt.show()

        # Log the completion
        logging.info(f"Outlier detection complete.")