import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Load the dataset with error handling and logging
        try:
            self.data = pd.read_csv(csv_path, parse_dates=[date_column])
            self.data.set_index(date_column, inplace=True)
            logger.info(f"Successfully loaded data from {csv_path}")
        except FileNotFoundError:
            logger.error(f"CSV file at path '{csv_path}' not found.")
            raise FileNotFoundError(f"CSV file at path '{csv_path}' not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading the data: {e}")
            raise Exception(f"An error occurred while loading the data: {e}")
        
        self.target_column = target_column
        self.look_back = look_back
        
        # Downsample data if specified
        if downsample_ratio > 1:
            self.data = self.data[::downsample_ratio]
            logger.info(f"Data downsampled by a factor of {downsample_ratio}")
        
        # Normalize the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_scaled = self.scaler.fit_transform(self.data[[self.target_column]].values)
        logger.info("Data normalization complete.")
        
        # Split into training and test sets
        self.train_size = int(len(self.data_scaled) * 0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        logger.info(f"Data split into training ({len(self.X_train)}) and testing ({len(self.X_test)}) sets.")
        
        # Build the LSTM model
        self.model = self._build_model()
        logger.info("LSTM model built successfully.")
        
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

        logger.info(f"Data prepared with {len(X_train)} training samples and {len(X_test)} testing samples.")
        return X_train, X_test, y_train, y_test

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
        logger.info("LSTM model built successfully.")
        return model

    def train_model(self, epochs=100, batch_size=32, patience=10, model_checkpoint_path="best_model.keras"):
        """
        Train the LSTM model with EarlyStopping and ModelCheckpoint.
        """
        logger.info("Training the model...")
        # Callbacks for early stopping and model checkpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test),
                       callbacks=[early_stopping, model_checkpoint])
        logger.info("Model training complete.")

    def evaluate_model(self):
        """
        Evaluates the model using RMSE and MSE and plots the forecast against actual values.
        """
        # Make predictions
        logger.info("Evaluating the model...")
        y_pred = self.model.predict(self.X_test)
    
        # Inverse transform predictions and actual values to original scale
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        y_test_inv = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
    
        # Calculate RMSE and MSE
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        self.rmse = rmse
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
    
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
            logger.error("Model is not trained yet. Please train the model first.")
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
        
        # Create the forecasted dates
        forecast_dates = pd.date_range(self.data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Create a DataFrame to store the forecast
        forecast_df = pd.DataFrame(data=self.forecast_unscaled, index=forecast_dates, columns=[self.target_column])
        forecast_df.to_csv(output_csv_path)  # Save the forecast to CSV
        logger.info(f"Forecast saved to {output_csv_path}")
        
        return forecast_df

    def get_rmse(self):
        """
        Return the RMSE of the model after evaluation.
        """
        return self.rmse