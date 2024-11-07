import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose

class FinancialDataEDA:
    def __init__(self, data):
        self.data = data
        self.rename_columns()
        self.handle_missing_values()

    def rename_columns(self):
        """Renames columns in the dataset to a standard format based on the number of columns."""
        for ticker, df in self.data.items():
            print(f"Columns before renaming for {ticker}:", df.columns)
            
            if len(df.columns) == 6:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low']
            elif len(df.columns) == 7:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open']
            elif len(df.columns) == 8:
                df.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            else:
                print(f"Unexpected column count for {ticker}: {len(df.columns)}")
                continue
            
            # Clean 'Date' or 'Ticker' rows if present
            df = df[~df['Date'].isin(['Date', 'Ticker'])]
            self.data[ticker] = df  # Update the dictionary with cleaned DataFrame

    def handle_missing_values(self):
        """Handle missing values by filling, interpolating, or removing them."""
        for ticker, df in self.data.items():
            print(f"Handling missing values for {ticker}...")
            # Check if there are missing values in the dataframe
            missing_values = df.isnull().sum()
            print(f"Missing values in {ticker}:\n{missing_values[missing_values > 0]}")

            # Handling missing values: fill forward, interpolate, or drop
            df.fillna(method='ffill', inplace=True)  # Forward fill
            df.interpolate(method='linear', inplace=True)  # Interpolate
            df.dropna(inplace=True)  # Drop rows with any remaining missing values
            
            self.data[ticker] = df  # Update the DataFrame after handling missing values
    def decompose_time_series(self, ticker, column='Close', period=252):
        """
        Decompose the time series into trend, seasonal, and residual components.

        Parameters:
        - ticker (str): The ticker symbol of the data to decompose.
        - column (str): The column to decompose. Default is 'Close'.
        - period (int): The period for decomposition. Default is 252 (approx. trading days in a year).

        Displays:
        - A four-panel plot with the original series, trend, seasonal, and residual components.
        """
        if ticker not in self.data:
            raise ValueError(f"The ticker '{ticker}' is not available in the dataset.")
        
        df = self.data[ticker]
        
        if column not in df.columns:
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
        print(f"\n{ticker} - Trend Component:\n", trend.describe())
        print(f"{ticker} - Seasonal Component:\n", seasonal.describe())
        print(f"{ticker} - Residual Component:\n", residual.describe())

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
        print(f"Plot saved as {plot_filename}")
        plt.show()

    def check_basic_statistics(self):
        """Check basic statistics of the data to understand its distribution."""
        for ticker, df in self.data.items():
            print(f"Basic statistics for {ticker}:")
            print(df.describe())  # Show basic statistics like mean, std, min, max, etc.

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
        print(f"Closing price plot saved as 'closing_price_plot.png' in ../Images/")
        plt.show()

    def plot_daily_pct_change(self):
        fig, axes = plt.subplots(nrows=len(self.data), figsize=(10, 6 * len(self.data)))
        fig.suptitle("Daily Percentage Change", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Ensure 'Close' is numeric
            df['Daily Return'] = df['Close'].pct_change()  # Calculate daily percentage change

            # Print the daily percentage change values for each ticker
            print(f"Daily percentage change for {ticker}:")
            print(df[['Date', 'Daily Return']].dropna())  # Display Date and Daily Return columns
            print("\n")

            ax = axes[i] if len(self.data) > 1 else axes
            ax.plot(df['Daily Return'], label=f'{ticker} Daily % Change', color='mediumseagreen', linewidth=2)  # Emerald Green color
            ax.set_title(f'{ticker} Daily Return')
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily % Change')
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../Images/daily_pct_change_plot.png')
        print(f"Daily percentage change plot saved as 'daily_pct_change_plot.png' in ../Images/")
        plt.show()

    def analyze_volatility(self, window=30):
        """Analyze volatility by calculating and plotting rolling mean and standard deviation in subplots."""
        fig, axes = plt.subplots(len(self.data), 2, figsize=(14, 4 * len(self.data)), sharex=True)
        fig.suptitle(f"{window}-Day Rolling Mean and Standard Deviation", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            # Calculate daily return if not already calculated
            if 'Daily Return' not in df.columns:
                df['Daily Return'] = df['Close'].pct_change()

            # Rolling Mean and Std Dev
            df['Rolling Mean'] = df['Daily Return'].rolling(window=window).mean()
            df['Rolling Std'] = df['Daily Return'].rolling(window=window).std()

            # Rolling Mean plot
            ax_mean = axes[i, 0] if len(self.data) > 1 else axes[0]
            ax_mean.plot(df['Rolling Mean'], label='Rolling Mean', color='mediumseagreen', linewidth=1.5)
            ax_mean.set_title(f"{ticker} - Rolling Mean")
            ax_mean.set_ylabel("Mean")
            ax_mean.legend()
            ax_mean.grid(True)
            
            # Rolling Std Dev plot
            ax_std = axes[i, 1] if len(self.data) > 1 else axes[1]
            ax_std.plot(df['Rolling Std'], label='Rolling Std Dev', color='mediumseagreen', linewidth=1.5)
            ax_std.set_title(f"{ticker} - Rolling Std Dev")
            ax_std.set_ylabel("Volatility")
            ax_std.legend()
            ax_std.grid(True)

            # Print last few rows of numerical output for rolling mean and std dev
            print(f"\n{ticker} - {window}-Day Rolling Mean and Standard Deviation:")
            print(df[['Rolling Mean', 'Rolling Std']].dropna().tail())

        plt.xlabel("Date")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'../Images/{window}_day_rolling_volatility.png')
        print(f"{window}-day rolling volatility plot saved as '{window}_day_rolling_volatility.png' in ../Images/")
        plt.show()
    def detect_outliers(self, threshold=2.5):
        """Detect outliers based on a threshold for standard deviations in returns, and plot in subplots."""
        fig, axes = plt.subplots(len(self.data), 1, figsize=(14, 4 * len(self.data)), sharex=True)
        fig.suptitle(f"Outliers Detection (Threshold = {threshold} Std Devs)", fontsize=16, fontweight='bold')

        for i, (ticker, df) in enumerate(self.data.items()):
            df['Daily Return'] = df['Close'].pct_change()
            mean = df['Daily Return'].mean()
            std_dev = df['Daily Return'].std()

            # Identify outliers
            df['Outlier'] = (df['Daily Return'] - mean).abs() > threshold * std_dev
            outliers = df[df['Outlier']]

            # Plot Daily Return with Outliers
            ax = axes[i] if len(self.data) > 1 else axes
            ax.plot(df.index, df['Daily Return'], label='Daily Return', color='mediumseagreen')  # Emerald Green color
            ax.scatter(outliers.index, outliers['Daily Return'], color='gold', label='Outliers', zorder=5)  # Gold for outliers
            ax.set_title(f"{ticker} - Daily Returns with Outliers")
            ax.set_ylabel("Daily % Change")
            ax.legend()
            ax.grid(True)

            # Print outliers for each ticker
            print(f"Outliers for {ticker}:")
            print(outliers[['Close', 'Daily Return']])

        plt.xlabel("Date")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('../Images/outliers_detection_plot.png')
        print(f"Outliers detection plot saved as 'outliers_detection_plot.png' in ../Images/")
        plt.show()