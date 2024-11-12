import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
        # Note: There was a mistake in the previous code. It should be y[:train_size], y[train_size:]

        y_train, y_test = y[:self.train_size], y[self.train_size:]

        # Reshape data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test

    def _build_model(self):
        """
        Builds and compiles the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()  # Display the model architecture
        return model

    def train_model(self, epochs=30, batch_size=32, patience=3, model_checkpoint_path="best_model.keras"):
        """
        Trains the LSTM model on the training data with early stopping and model checkpoints.
    
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param patience: Number of epochs with no improvement to wait before stopping.
        :param model_checkpoint_path: Path to save the best model.
        """
        # Early stopping and model checkpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss')
    
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size, 
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        print("Model training complete.")
        return history

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
        plt.plot(self.data.index[-len(self.y_test):], y_test_inv, label='Actual Data_TSLA', color='blue', linestyle='--', linewidth=2)
        plt.plot(self.data.index[-len(self.y_test):], y_pred_inv, label='Predicted Data (LSTM)_TSLA', color='orange', linewidth=2)
        
        # Title and labels for the plot with updated font size and style
        plt.title(f'LSTM Model Forecast vs Actual\nMSE: {mse:.4f} | RMSE: {rmse:.4f}', fontsize=18, fontweight='bold', family='serif')
        plt.xlabel('Date', fontsize=14, fontweight='bold', family='serif')
        plt.ylabel(self.target_column, fontsize=14, fontweight='bold', family='serif')
        
        # Add gridlines and customize the legend
        plt.legend(loc='upper left', fontsize=12, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)
    
        # Save the plot with high resolution for a journal or presentation
        plt.tight_layout()  # Ensures the layout is tight and doesn't cut off labels
        plt.savefig('../Images/forecast_vs_actual_LSTM_TSLA.png', dpi=300, bbox_inches='tight')  # Save plot at 300 DPI for clarity
    
        # Show the plot
        plt.show()

    def generate_forecast(self, forecast_steps=180):
        """
        Generate forecasts for the specified number of future steps using the trained model.
        
        :param forecast_steps: Number of steps to forecast into the future.
        :return: Forecasted values in original scale.
        """
        forecast = []
        last_sequence = self.X_test[-1].copy()  # Start from the last sequence in test set

        for _ in range(forecast_steps):
            # Predict next value
            pred = self.model.predict(last_sequence[np.newaxis, :, :])[0, 0]
            forecast.append(pred)
            
            # Update the last_sequence by appending the prediction and removing the first value
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1, 0] = pred

        # Inverse transform to original scale
        forecast_inv = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        self.forecast = forecast
        self.forecast_unscaled = forecast_inv
        
        # Save the forecasted data to a CSV file
        forecast_df = pd.DataFrame(forecast_inv, columns=[self.target_column], index=pd.date_range(start=self.data.index[-1], periods=forecast_steps+1, freq='B')[1:])
        forecast_df.to_csv('../Images/forecast_TSLA.csv', index=True)
        
        print(f"Generated {forecast_steps} forecast steps and saved to 'forecast.csv'.")
        return forecast_inv

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
        Provides an interpretation of the forecast including trend analysis, volatility, and risk.
        """
        forecast = self.generate_forecast()
        ci_upper = forecast * 1.05  # Example for 5% confidence interval
        ci_lower = forecast * 0.95
        
        # 1. Trend Analysis
        print("\n--- Trend Analysis ---")
        trend_direction = "upward" if forecast[-1] > forecast[0] else "downward"
        print(f"Forecasted trend: {trend_direction}")
        
        # 2. Volatility and Risk
        print("\n--- Volatility and Risk_TSLA ---")
        volatility = np.std(forecast)
        print(f"Estimated forecast volatility+TSLA: {volatility:.4f}")
        print("Periods with higher forecast volatility are observed if CI width increases.")
        
        # 3. Market Opportunities and Risks
        print("\n--- Market Opportunities and Risks ---")
        if trend_direction == "upward":
            print("Potential Opportunity: Price expected to increase, suggesting potential buying opportunities.")
        else:
            print("Potential Risk: Price expected to decrease, suggesting potential selling opportunities or hedging.")