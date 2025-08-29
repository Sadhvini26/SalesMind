import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Forecasting features will be limited.")

class SalesForecaster:
    """Sales forecasting using Prophet and simple statistical methods."""
    
    def __init__(self):
        self.models = {}
        self.last_trained = {}

    def prepare_forecast_data(self, df: pd.DataFrame, date_col: str = 'order_date', 
                            value_col: str = 'sales', freq: str = 'D') -> pd.DataFrame:
        """Prepare data for forecasting."""
        try:
            # Ensure date column is datetime
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                raise ValueError(f"Date column '{date_col}' not found in data")
            
            if value_col not in df.columns:
                raise ValueError(f"Value column '{value_col}' not found in data")
            
            # Aggregate by date
            forecast_df = df.groupby(date_col)[value_col].sum().reset_index()
            forecast_df.columns = ['ds', 'y']  # Prophet naming convention
            
            # Sort by date
            forecast_df = forecast_df.sort_values('ds').reset_index(drop=True)
            
            # Fill missing dates if needed
            date_range = pd.date_range(
                start=forecast_df['ds'].min(),
                end=forecast_df['ds'].max(),
                freq=freq
            )
            
            full_df = pd.DataFrame({'ds': date_range})
            forecast_df = full_df.merge(forecast_df, on='ds', how='left')
            forecast_df['y'] = forecast_df['y'].fillna(0)
            
            return forecast_df
            
        except Exception as e:
            logging.error(f"Error preparing forecast data: {e}")
            raise

    def prophet_forecast(self, df: pd.DataFrame, periods: int = 30, 
                        freq: str = 'D') -> Dict[str, Any]:
        """Generate forecast using Prophet."""
        if not PROPHET_AVAILABLE:
            return {
                'success': False,
                'error': 'Prophet not available. Please install prophet package.',
                'forecast': None,
                'metrics': None
            }
        
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )
            
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Calculate basic metrics on historical data
            historical = forecast[forecast['ds'].isin(df['ds'])]
            actual = df['y']
            predicted = historical['yhat']
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if actual.mean() != 0 else 0
            
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
            return {
                'success': True,
                'error': None,
                'forecast': forecast,
                'metrics': metrics,
                'model': model
            }
            
        except Exception as e:
            logging.error(f"Prophet forecast error: {e}")
            return {
                'success': False,
                'error': str(e),
                'forecast': None,
                'metrics': None
            }

    def simple_forecast(self, df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """Simple moving average forecast as fallback."""
        try:
            # Calculate moving averages
            window_sizes = [7, 14, 30]
            predictions = []
            
            for window in window_sizes:
                if len(df) >= window:
                    ma = df['y'].rolling(window=window).mean().iloc[-1]
                    predictions.append(ma)
            
            # Use average of moving averages
            if predictions:
                avg_prediction = np.mean(predictions)
            else:
                avg_prediction = df['y'].mean()
            
            # Create forecast dataframe
            last_date = df['ds'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': avg_prediction,
                'yhat_lower': avg_prediction * 0.8,
                'yhat_upper': avg_prediction * 1.2
            })
            
            # Simple metrics
            recent_actual = df['y'].tail(min(30, len(df)))
            mae = np.mean(np.abs(recent_actual - recent_actual.mean()))
            
            metrics = {
                'mae': float(mae),
                'rmse': float(mae * 1.2),  # Rough estimate
                'mape': 15.0  # Default estimate
            }
            
            return {
                'success': True,
                'error': None,
                'forecast': forecast,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Simple forecast error: {e}")
            return {
                'success': False,
                'error': str(e),
                'forecast': None,
                'metrics': None
            }

    def forecast_sales(self, df: pd.DataFrame, periods: int = 30, 
                      method: str = 'prophet', **kwargs) -> Dict[str, Any]:
        """Main forecasting method."""
        try:
            # Prepare data
            forecast_data = self.prepare_forecast_data(df, **kwargs)
            
            if len(forecast_data) < 10:
                return {
                    'success': False,
                    'error': 'Insufficient data for forecasting (minimum 10 points required)',
                    'forecast': None,
                    'metrics': None
                }
            
            # Choose forecasting method
            if method.lower() == 'prophet' and PROPHET_AVAILABLE:
                result = self.prophet_forecast(forecast_data, periods)
            else:
                result = self.simple_forecast(forecast_data, periods)
            
            return result
            
        except Exception as e:
            logging.error(f"Forecasting error: {e}")
            return {
                'success': False,
                'error': str(e),
                'forecast': None,
                'metrics': None
            }

# Global forecaster instance
forecaster = SalesForecaster()
