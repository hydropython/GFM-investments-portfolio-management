import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

class FinancialDataFetcher:
    def __init__(self, tickers, start_date, end_date, save_path, image_save_path):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.save_path = save_path
        self.image_save_path = image_save_path
        self.data = {}

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FinancialDataFetcher initialized with tickers: {self.tickers}, start_date: {self.start_date}, end_date: {self.end_date}, save_path: {self.save_path}, image_save_path: {self.image_save_path}")

    def fetch_data(self):
        """Fetch historical data for each ticker and store in a dictionary."""
        for ticker in self.tickers:
            self.logger.info(f"Fetching data for {ticker} from {self.start_date} to {self.end_date}")
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date)
                self.data[ticker] = data
                self.logger.info(f"Successfully fetched data for {ticker}. First few rows:\n{data.head()}")
                
                # Save each dataframe to a CSV file
                file_path = f"{self.save_path}/{ticker}_historical_data.csv"
                data.to_csv(file_path)
                self.logger.info(f"Data for {ticker} saved to {file_path}.")
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")

    def calculate_volatility(self, window=30):
        """Calculate rolling volatility for each asset."""
        self.logger.info(f"Calculating rolling volatility with a window of {window} days for each asset.")
        volatility = {}
        for ticker, df in self.data.items():
            self.logger.info(f"Calculating volatility for {ticker}.")
            try:
                df['Daily Return'] = df['Adj Close'].pct_change()
                df['Volatility'] = df['Daily Return'].rolling(window=window).std() * (252 ** 0.5)  # Annualized volatility
                volatility[ticker] = df['Volatility']
                self.logger.info(f"Calculated volatility for {ticker}. Latest value: {df['Volatility'].iloc[-1]:.4f}")
            except Exception as e:
                self.logger.error(f"Error calculating volatility for {ticker}: {str(e)}")

        # Log the volatility values for each ticker
        self.logger.info("\nAnnualized Volatility for each asset:")
        for ticker, vol in volatility.items():
            self.logger.info(f"{ticker}: {vol.iloc[-1]:.4f}")  # Log the latest volatility value

        return volatility

    def plot_volatility_comparison(self, volatility):
        """Plot the volatility of each asset."""
        self.logger.info(f"Plotting volatility comparison for {', '.join(volatility.keys())}.")
        try:
            plt.figure(figsize=(12, 6))

            for ticker, vol in volatility.items():
                plt.plot(vol.index, vol, label=f'{ticker} Volatility')

            plt.title("Volatility Comparison: TSLA, BND, SPY", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Annualized Volatility", fontsize=12)
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()

            # Save the figure to the Images folder
            plot_path = f'{self.image_save_path}/volatility_comparison.png'
            plt.savefig(plot_path)
            self.logger.info(f"Volatility comparison plot saved to {plot_path}.")
            plt.show()
        except Exception as e:
            self.logger.error(f"Error plotting volatility comparison: {str(e)}")

    def plot_data(self):
        """Plot Open, Close, Low, and High values in a single plot."""
        self.logger.info(f"Plotting financial data for tickers: {', '.join(self.tickers)}.")
        try:
            sns.set(style="whitegrid", palette="dark")
            # Define luxury colors
            colors = {'TSLA': '#228B22', 'BND': '#D4AF37', 'SPY': '#000000'}
            
            # Create a subplot with 2 rows and 2 columns for each ticker's data (Open, High, Low, Close + Volume)
            fig, axes = plt.subplots(len(self.tickers), 2, figsize=(18, 10), sharex=False)

            # Loop over each ticker and plot its data
            for i, ticker in enumerate(self.tickers):
                df = self.data[ticker]
                
                # Plot Open, High, Low, Close in a single plot
                ax1 = axes[i, 0]
                ax1.plot(df['Open'], label='Open', color='blue', linewidth=2)
                ax1.plot(df['High'], label='High', color='green', linewidth=2)
                ax1.plot(df['Low'], label='Low', color='red', linewidth=2)
                ax1.plot(df['Close'], label='Close', color='purple', linewidth=2)
                ax1.set_title(f'{ticker} - Open, High, Low, Close', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Price (USD)', fontsize=12)
                ax1.legend(loc='best')
                ax1.grid(True)

                # Plot Volume
                ax2 = axes[i, 1]
                ax2.plot(df['Volume'], label='Volume', color=colors.get(ticker, 'blue'), linewidth=2)
                ax2.set_title(f'{ticker} - Volume Traded', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.legend(loc='best')
                ax2.grid(True)

            # Set the x-axis labels to be more readable
            for ax in axes.flat:
                ax.set_xlabel('Date', fontsize=12)
                ax.tick_params(axis='x', rotation=45)

            # Add overall title
            fig.suptitle('Financial Data Overview for TSLA, BND, and SPY', fontsize=18, fontweight='bold')

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust title to fit

            # Save the figure to the specified path
            image_file_path = f"{self.image_save_path}/financial_data_overview.png"
            plt.savefig(image_file_path, bbox_inches='tight')
            self.logger.info(f"Figure saved to {image_file_path}.")

            # Show the plot
            plt.show()
        except Exception as e:
            self.logger.error(f"Error plotting financial data: {str(e)}")

    def get_data(self, ticker):
        """Return the historical data for a specific ticker."""
        self.logger.info(f"Fetching data for {ticker}.")
        data = self.data.get(ticker, f"No data available for {ticker}.")
        if isinstance(data, pd.DataFrame):
            self.logger.info(f"Data for {ticker}: {data.head()}")
        else:
            self.logger.warning(f"No data found for {ticker}")
        return data

    def display_data_summary(self):
        """Display summary of fetched data."""
        self.logger.info("Displaying summary of fetched data.")
        for ticker, df in self.data.items():
            self.logger.info(f"\nSummary of data for {ticker}:")
            self.logger.info(f"\n{df.head()}")