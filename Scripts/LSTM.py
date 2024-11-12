import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber

class LSTMTimeSeriesForecaster:
    def __init__(self, csv_path, date_column="Date", target_column="Close", look_back=60, downsample_ratio=1):
        """
        Initializes the class with dataset and model parameters.

        :param csv_path: Path to the CSV dataset.
        :param date_column: The column representing the date.
        :param target_column: The column to forecast.
        :param look_back: Number of time steps to look back.
        :param downsample_ratio: Ratio for downsampling data (default is 1, meaning no downsampling).
        """
        # Load the dataset with error handling
        try:
            self.data = pd.read_csv(csv_path, parse_dates=[date_column])
            self.data.set_index(date_column, inplace=True)
            print(f"Successfully loaded data from {csv_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file at path '{csv_path}' not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the data: {e}")
        
        self.target_column = target_column
        self.look_back = look_back
        
        # Downsample data if specified
        if downsample_ratio > 1:
            self.data = self.data[::downsample_ratio]
            print(f"Data downsampled by a factor of {downsample_ratio}")
        
        # Normalize the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_scaled = self.scaler.fit_transform(self.data[[self.target_column]].values)
        print("Data normalization complete.")
        
        # Split into training and test sets
        self.train_size = int(len(self.data_scaled) * 0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        print(f"Data split into training ({len(self.X_train)}) and testing ({len(self.X_test)}) sets.")
        
        # Build the LSTM model
        self.model = self._build_model()
        print("LSTM model built successfully.")
        
        # Initialize attributes for forecasting and evaluation
        self.forecast = None
        self.forecast_unscaled = None
        self.confidence_interval_upper = None
        self.confidence_interval_lower = None
        self.rmse = None

    def _prepare_data(self):
        """
        Prepares the data for LSTM by creating sequences of 'look_back' length.
        """
        X, y = [], []
        for i in range(self.look_back, len(self.data_scaled)):
            X.append(self.data_scaled[i - self.look_back:i, 0])  # Use the previous 'look_back' values
            y.append(self.data_scaled[i, 0])  # The current value is the target
        X, y = np.array(X), np.array(y)

        # Split data into train and test sets
        X_train, X_test = X[:self.train_size], X[self.train_size:]
        y_train, y_test = y[:self.train_size], y[self.train_size:]

        # Reshape data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test

    # Build the LSTM model
    def _build_model(self):
        """
        Builds and compiles the updated LSTM model with additional layers and dropout.
        """
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))  # 1st LSTM layer with return_sequences=True
        model.add(Dropout(0.2))  # Dropout for regularization
        model.add(LSTM(50, return_sequences=False))  # 2nd LSTM layer
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss=Huber())  # Use Huber loss directly from Keras
        model.summary()  # Display the model architecture
        return model

    def train_model(self, epochs=100, batch_size=32, patience=10, model_checkpoint_path="best_model.keras"):
        """
        Train the LSTM model with EarlyStopping and ModelCheckpoint.
        """
        # Callbacks for early stopping and model checkpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test),
                       callbacks=[early_stopping, model_checkpoint])

    def create_lag_features(self, df, lags=5):
        """
        Creates lag features for the dataset to account for previous time steps.
        
        :param df: The input dataframe (stock price data).
        :param lags: Number of lag features to create.
        :return: Dataframe with lag features.
        """
        lagged_df = df.copy()
        for lag in range(1, lags+1):
            lagged_df[f'lag_{lag}'] = lagged_df['Close'].shift(lag)
        lagged_df.dropna(inplace=True)
        return lagged_df

    def add_volatility_features(self, df, window=20):
        """
        Adds volatility features like rolling standard deviation to the dataset.
        
        :param df: The input dataframe (stock price data).
        :param window: The window size for rolling volatility calculation.
        :return: Dataframe with volatility features.
        """
        df['volatility'] = df['Close'].rolling(window=window).std()
        return df

    def evaluate_model(self):
        """
        Evaluates the model using RMSE and MSE and plots the forecast against actual values.
        """
        # Make predictions
        y_pred = self.model.predict(self.X_test)
    
        # Inverse transform predictions and actual values to original scale
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        y_test_inv = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
    
        # Calculate RMSE and MSE
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        self.rmse = rmse
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
    
        # Plot the results
        plt.figure(figsize=(14, 8))
        plt.plot(self.data.index[-len(self.y_test):], y_test_inv, label='Actual Data', color='blue', linestyle='--', linewidth=2)
        plt.plot(self.data.index[-len(self.y_test):], y_pred_inv, label='Predicted Data (LSTM)', color='orange', linewidth=2)
        
        # Title and labels for the plot
        plt.title(f'LSTM Model Forecast vs Actual\nMSE: {mse:.4f} | RMSE: {rmse:.4f}', fontsize=18, fontweight='bold', family='serif')
        plt.xlabel('Date', fontsize=14, fontweight='bold', family='serif')
        plt.ylabel(self.target_column, fontsize=14, fontweight='bold', family='serif')
        
        # Add gridlines and customize the legend
        plt.legend(loc='upper left', fontsize=12, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)
    
        # Save the plot
        plt.tight_layout()  # Ensures the layout is tight
        plt.savefig('forecast_vs_actual_LSTM.png', dpi=300, bbox_inches='tight')  # Save plot at high resolution
        plt.show()

    def generate_forecast(self, steps=30, output_csv_path='forecast_output.csv'):
        """
        Generate future forecasts using the trained LSTM model and save the forecasted values to a CSV file.

        :param steps: Number of future steps to forecast.
        :param output_csv_path: Path to save the forecasted data as a CSV file.
        :return: Forecasted values and the confidence intervals.
        """
        # Ensure the model is trained
        if not self.model:
            raise ValueError("Model is not trained yet. Please train the model first.")
        
        # Use the last test input for forecasting
        forecast_input = self.X_test[-1:]  # Last sequence from the test set
        
        forecast = []
        for _ in range(steps):
            # Predict the next value
            prediction = self.model.predict(forecast_input)
            forecast.append(prediction[0, 0])
            
            # Update the input sequence for the next prediction
            forecast_input = np.append(forecast_input[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        
        # Store the forecasted values
        self.forecast = np.array(forecast).reshape(-1, 1)
        
        # Inverse scale the forecasted values
        self.forecast_unscaled = self.scaler.inverse_transform(self.forecast)
        
        # Calculate confidence intervals (± 2 standard deviations)
        std_dev = np.std(self.forecast_unscaled)
        self.confidence_interval_upper = self.forecast_unscaled + 2 * std_dev
        self.confidence_interval_lower = self.forecast_unscaled - 2 * std_dev

        # Plot the forecast along with confidence intervals
        plt.figure(figsize=(14, 8))
        
        # Plot the actual data (scaled back to original scale)
        plt.plot(self.data.index[-len(self.y_test):], self.scaler.inverse_transform(self.y_test.reshape(-1, 1)), label='Actual Data', color='blue')
        
        # Plot the forecasted data (scaled back to original scale)
        forecast_dates = pd.date_range(self.data.index[-1], periods=steps+1, freq='D')[1:]
        plt.plot(forecast_dates, self.forecast_unscaled, label='Forecasted Data', color='orange')
        
        # Plot confidence intervals
        plt.fill_between(forecast_dates, 
                        self.confidence_interval_lower.flatten(), 
                        self.confidence_interval_upper.flatten(), 
                        color='orange', alpha=0.2, label='Confidence Interval')
        
        # Final plot customization
        plt.title(f'Future Forecasting with LSTM', fontsize=18)
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.show()

        # Save the forecast and confidence intervals to a CSV file
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': self.forecast_unscaled.flatten(),
            'Confidence Interval Lower': self.confidence_interval_lower.flatten(),
            'Confidence Interval Upper': self.confidence_interval_upper.flatten()
        })
        
        # Save to CSV
        forecast_df.to_csv(output_csv_path, index=False)
        print(f"Forecast saved to {output_csv_path}")

    def plot_forecast(self, forecast_steps=180, confidence_interval=1.96):
        """
        Plots the forecast alongside historical data with confidence intervals.
        
        :param forecast_steps: Number of future steps to forecast.
        :param confidence_interval: Confidence interval for the forecast range.
        """
        # Generate forecast if not already done
        if self.forecast is None:
            self.generate_forecast(forecast_steps)
    
        # Generate upper and lower confidence intervals
        forecast_std = np.std(self.forecast)  # Assuming constant variance for simplicity
        upper_bound = self.forecast_unscaled + confidence_interval * forecast_std
        lower_bound = self.forecast_unscaled - confidence_interval * forecast_std
    
        # Plot forecast and confidence intervals
        plt.figure(figsize=(14, 8))
        plt.plot(self.data.index, self.data[self.target_column], label='Actual Data-TSLA', color='blue', linestyle='--', linewidth=2)
        plt.plot(pd.date_range(start=self.data.index[-1], periods=forecast_steps+1, freq='B')[1:], self.forecast_unscaled, label='Forecasted Data_TSLA', color='orange', linewidth=2)
        plt.fill_between(pd.date_range(start=self.data.index[-1], periods=forecast_steps+1, freq='B')[1:], lower_bound.flatten(), upper_bound.flatten(), color='lightgray', alpha=0.5, label='Confidence Interval_TSLA')
        
        # Title and labels for the plot with updated font size and style
        plt.title(f'LSTM Forecast with Confidence Interval_TSLA\nRMSE: {self.rmse:.4f}', fontsize=18, fontweight='bold', family='serif')
        plt.xlabel('Date', fontsize=14, fontweight='bold', family='serif')
        plt.ylabel(self.target_column, fontsize=14, fontweight='bold', family='serif')
        
        # Add gridlines and customize the legend
        plt.legend(loc='upper left', fontsize=12, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)
    
        # Save the plot with high resolution for a journal or presentation
        plt.tight_layout()
        plt.savefig('../Images/forecast_with_confidence_interval_TSLA.png', dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
    def interpret_results(self):
        """
        Interpret the forecasted results and provide insights on potential investment decisions.
        
        This method will:
        - Display forecast summary statistics.
        - Discuss portfolio adjustments based on forecasted trends.
        - Calculate and display risk-adjusted metrics like expected returns and Sharpe Ratio.
        - Provide recommendations for rebalancing the portfolio.
        """

        # Check if forecast data is available
        if not hasattr(self, 'forecast_unscaled') or self.forecast_unscaled is None:
            raise ValueError("No forecast data found. Please run generate_forecast() first.")
        
        # Calculate basic statistics for forecasted data
        mean_forecasted_value = self.forecast_unscaled.mean()
        std_dev_forecasted_value = self.forecast_unscaled.std()
        upper_bound = self.confidence_interval_upper.mean()
        lower_bound = self.confidence_interval_lower.mean()
        
        print("Forecast Interpretation:")
        print(f"Mean Forecasted Value: {mean_forecasted_value:.2f}")
        print(f"Standard Deviation of Forecast: {std_dev_forecasted_value:.2f}")
        print(f"Confidence Interval (Mean ± 2 Std): [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Portfolio Adjustments based on forecasted trends
        # Example weights and portfolio risk metrics (assuming forecasted prices for TSLA, BND, SPY are available in a DataFrame `df`):
        assets = ['TSLA', 'BND', 'SPY']
        df_forecasted = pd.DataFrame({
            'TSLA': self.forecast_unscaled.flatten(),  # Example forecasted data for TSLA
            'BND': np.random.normal(mean_forecasted_value, std_dev_forecasted_value, len(self.forecast_unscaled)),  # Placeholder for BND
            'SPY': np.random.normal(mean_forecasted_value, std_dev_forecasted_value, len(self.forecast_unscaled))   # Placeholder for SPY
        })
        
        # Daily returns calculation
        daily_returns = df_forecasted.pct_change().dropna()
        
        # Annualized return and covariance matrix
        annualized_returns = daily_returns.mean() * 252
        cov_matrix = daily_returns.cov() * 252

        # Define initial portfolio weights
        initial_weights = np.array([0.4, 0.4, 0.2])  # Starting weights for TSLA, BND, SPY respectively

        # Define optimization functions and optimize for Sharpe Ratio as shown earlier
        
        # Example results after optimization
        optimized_weights = initial_weights  # Replace with actual optimized weights from optimization step
        portfolio_return = np.dot(optimized_weights, annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        print("\nOptimized Portfolio Results:")
        for asset, weight in zip(assets, optimized_weights):
            print(f"{asset} Weight: {weight:.2%}")
        print(f"\nExpected Portfolio Return: {portfolio_return:.2%}")
        print(f"Expected Portfolio Volatility: {portfolio_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Recommendations
        print("\nRecommendations:")
        if sharpe_ratio < 1:
            print("Consider increasing BND allocation for stability due to high volatility in forecasted TSLA prices.")
        elif sharpe_ratio >= 1 and sharpe_ratio < 2:
            print("Portfolio shows moderate risk-adjusted returns. Maintain diversification and monitor TSLA volatility.")
        else:
            print("High Sharpe Ratio indicates a favorable risk-return trade-off. Consider increasing TSLA allocation slightly.")

        # Plotting cumulative returns for visual insight (optional)
        cumulative_returns = (1 + daily_returns.dot(optimized_weights)).cumprod()
        cumulative_returns.plot(figsize=(10, 6), title="Portfolio Cumulative Returns Based on Forecast", xlabel="Date", ylabel="Cumulative Return")