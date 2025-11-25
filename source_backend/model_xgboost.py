"""
Defines the XGBoostModels class for automated time series forecasting using XGBoost. This class facilitates the 
initialization with a dataset and model configurations, supporting both general and item-specific predictions. The main 
functionality includes creating the XGBoost model, generating predictions, and extracting optimal hyperparameters while 
maintaining consistency with the existing model structure.
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import json
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from source_backend.bayesian_optimizer import BayesianOptimization
from source_database.transformations import nature_encode
from util import get_device_config, is_cpu_mode

########################################################################################################################
#
# MODEL
#
########################################################################################################################
class XGBoostModels:
    def __init__(self,
                X: Optional[pd.DataFrame] = None,
                y: Optional[pd.Series] = None,
                **kwargs
                ):
        """
        """
        # Create copy to avoid modifying the original datasets;
        self.X = X.copy() if X is not None else None
        self.y = y.copy() if y is not None else None

        # Standardize date column if data is provided;
        if self.X is not None:
            self._add_calendar_features()

    def _add_calendar_features(self) -> None:
        """Transforms the 'Data_Hora_Medicao' into features for GBM type of models"""
        if 'Data_Hora_Medicao' not in self.X.columns:
            return None
        
        # Extract datetime components;
        ts = pd.to_datetime(self.X['Data_Hora_Medicao'])
        
        # Year as regular numeric feature (not cyclical - it doesn't repeat);
        self.X['year'] = ts.dt.year.astype(float)
        
        # Cyclical features that benefit from nature_encode;
        self.X['month'] = ts.dt.month
        self.X['day'] = ts.dt.day
        self.X['hour'] = ts.dt.hour
        self.X['dayofyear'] = ts.dt.dayofyear
        self.X['dayofweek'] = ts.dt.dayofweek
        
        # Special handling for day of month (varies by month length);
        days_in_month = ts.dt.days_in_month
        
        # Apply cyclical encoding to other periodic features;
        cyclical_features = [
            ('month', 12),       # Monthly cycle;
            ('hour', 24),         # Daily cycle;
            ('dayofyear', 365),  # Annual cycle;
            ('dayofweek', 7),    # Weekly cycle;
            ('day', ts.dt.days_in_month) # Monthly cycle;
        ]
        
        for col, period in cyclical_features:
            nature_encode(df=self.X, col=col, div_period=period)
            self.X.drop(columns=[col], inplace=True)
        
        # Drop the original datetime column;
        self.X.drop(columns=['Data_Hora_Medicao'], inplace=True)

    def _create_model(self, use_best_params: bool = False) -> XGBRegressor: 
        """
        Creates and configures the XGBoost model;
        
        Parameters:
            - use_best_params (bool): Whether to use the best parameters from previous training
            
        Returns:
            - XGBRegressor: Configured XGBoost model
        """
        if use_best_params and self.best_param:
            # Merge device config with saved parameters;
            merged_params = {**self.best_param, **self.device_config}
            return XGBRegressor(**merged_params)
        else:
            # Default parameters if no best parameters are available;
            default_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42,
                **self.device_config  # Include device configuration
            }
            return XGBRegressor(**default_params)

    def _get_best_params(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Performs Bayesian optimization to find the best hyperparameters;
        
        Parameters:
            - X_train: Training features;
            - y_train: Training target;
            
        Returns:
            - dict: Best hyperparameters;
        """
        optimizer = BayesianOptimization(
            model_name="xgb",
            n_trials=self.n_trials,
            X_train=X_train,
            y_train=y_train,
            cpu=is_cpu_mode(self.mode)
        )
        
        # Get best parameters from optimizer Class;
        best_params = optimizer.optimize()
        return best_params

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits the model. If X and y are provided, they update the internal datasets.
        """
        if X is not None:
            self.X = X.copy()
        if y is not None:
            self.y = y.copy()
            
        if self.X is not None:
            self._add_calendar_features()
            
        # ... existing fit logic ...
        """
        Finds optimal hyperparameters using Bayesian optimization while respecting the time series structure
        (TimeSeriesSplit). Also, it evaluates model performance on the last forecasting_horizon periods for each
        division;
        
        Returns:
            - Tuple[pd.DataFrame, pd.DataFrame]: Model info and predictions on test periods
        """
        divisions = self.df['unique_id'].unique()
        all_results = []
        all_predictions = []
        
        for division in divisions:
            division_df = self.df[self.df['unique_id'] == division].copy()
            division_df['ds'] = pd.to_datetime(division_df['ds'])
            division_df = division_df.sort_values('ds')
            
            # Prepare features and target
            X_with_dates = division_df.drop(['y', 'unique_id'], axis=1).copy()
            y = division_df['y']
            
            # Separate ds from the dataset;
            ds_values = X_with_dates['ds'].copy()
            X = X_with_dates.drop(['ds'], axis=1)
            
            # Get or find best parameters -> Bayesian Optimizer;
            if not self.best_param:
                self.best_param = self._get_best_params(X, y)
                # Save the best parameters for future use;
                self._save_best_params(self.best_param)
            
            # Create model with optimized parameters;
            model = self._create_model(use_best_params=True)
            
            # Train model on full dataset;
            model.fit(X, y)
            
            # Store model information;
            model_info = pd.DataFrame({
                'unique_id': [division],
                'model_type': ['XGBRegressor'],
                'hyperparameters': [str(self.best_param)],
                'training_samples': [len(X)]
            })
            all_results.append(model_info)
            
            # Evaluate on last forecasting_horizon periods if enough data;
            if len(division_df) >= self.forecasting_horizon:
                test_indices = range(len(division_df) - self.forecasting_horizon, len(division_df))
                test_ds = ds_values.iloc[test_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
                
                # Make predictions;
                predictions = model.predict(X_test)
                
                # Create prediction DataFrame - match neural models format;
                pred_df = pd.DataFrame({
                    'ds': test_ds.values,
                    'unique_id': division,
                    'XGBRegressor': predictions,
                    'y': y_test.values
                })
                all_predictions.append(pred_df)
        
        # Combine results;
        model_info_df = pd.concat(all_results, axis=0).reset_index(drop=True)
        
        if all_predictions:
            predictions_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
        else:
            # Create empty DataFrame with expected columns if no predictions;
            predictions_df = pd.DataFrame(columns=['ds', 'unique_id', 'XGBRegressor', 'y'])
        return predictions_df, model_info_df

    def predict_future(self) -> pd.DataFrame:
        """
        Generates forecasts for future time periods.
        
        Returns:
            - pd.DataFrame: DataFrame containing forecasts for future periods
        """
        # Create future DataFrame with engineered features;
        future_df = self._create_future_df(self.df)
        
        # Make predictions for each division;
        divisions = self.df['unique_id'].unique() #TODO Check if this is consistent for items;
        all_predictions = []
        
        for division in divisions:
            # Get training data for this division;
            division_df = self.df[self.df['unique_id'] == division].copy()
            division_df['ds'] = pd.to_datetime(division_df['ds'])
            
            # Prepare features;
            X_train = division_df.drop(['ds', 'y', 'unique_id'], axis=1)
            y_train = division_df['y']
            X_train = X_train[future_df.drop('ds', axis=1).columns] #Reorder columns to match future_df;
            
            # Train model with best parameters;     
            model = self._create_model(use_best_params=True)
            model.fit(X_train, y_train)
            
            # Make predictions;
            predictions = model.predict(future_df.drop('ds', axis=1))
            
            # Create prediction DataFrame;
            pred_df = pd.DataFrame({
                'ds': future_df['ds'],
                'unique_id': division,
                'XGBRegressor': predictions
            })
            all_predictions.append(pred_df)
        
        # Combine all predictions;
        future_predictions = pd.concat(all_predictions, axis=0)
        return future_predictions

    def last_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the last cutoff as the final result. Window evaluation still needs to be implemented for XGBoost;
        """
        cutoffs = df['ds'].sort_values().unique().tolist()
        last_cutoff = cutoffs[-1]

        window = df.loc[df.ds == last_cutoff]
        window = window.reset_index(drop=True)
        return window

    def _create_future_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a future dataframe with engineered features and forecasted external variables;
        
        Parameters:
            - df (pd.DataFrame): Input DataFrame containing training data;
            
        Returns:
            - pd.DataFrame: DataFrame with future dates, engineered features and forecasted external variables;
        """
        unique_ids = df['unique_id'].unique().tolist()
        last_date = df['ds'].max()
        
        # Create a single dataframe from the first unique_id to use for forecasting external variables;
        single_df = df.loc[df['unique_id'] == unique_ids[0]].copy()
        single_df = single_df.drop('unique_id', axis=1).drop_duplicates()
        
        # Ensure dates are sorted and unique;
        single_df = single_df.sort_values('ds').drop_duplicates('ds')
        single_df.set_index('ds', inplace=True)
        
        # Create future dates;
        offset = pd.tseries.frequencies.to_offset(self.freq)
        next_date = last_date + offset
        future_dates = pd.date_range(start=next_date, periods=self.forecasting_horizon, freq=self.freq)
        
        # Initialize the future DataFrame;
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['year'] = future_df['ds'].dt.year
        future_df['month'] = future_df['ds'].dt.month
        #future_df['day'] = future_df['ds'].dt.day
        nature_encode(df=future_df, col='month', div_period=12)
        future_df.drop('month', axis=1, inplace=True)
        
        # Define standard columns that shouldn't be forecasted;
        standard_cols = ['unique_id', 'ds', 'y', 'year', 'month', 'month_cos', 'month_sin', 'day', 'region']
        pred_cols = [col for col in df.columns.to_list() if col not in standard_cols]
        
        # Forecast external variables if any exist;
        for col in pred_cols:
            print(f"Forecasting external variable: {col}")
            
            # Prepare training data - check what features are available;
            available_features = []
            if 'year' in single_df.columns:
                available_features.append('year')
            if 'month_sin' in single_df.columns:
                available_features.append('month_sin')
            if 'month_cos' in single_df.columns:
                available_features.append('month_cos')
            if 'month' in single_df.columns:
                available_features.append('month')
            
            # Use available features for training;
            X_train = single_df[available_features]
            y_train = single_df[col]
            
            # Use corresponding features for testing;
            test_features = []
            if 'year' in future_df.columns:
                test_features.append('year')
            if 'month_sin' in future_df.columns:
                test_features.append('month_sin')
            if 'month_cos' in future_df.columns:
                test_features.append('month_cos')
            if 'month' in future_df.columns:
                test_features.append('month')
            
            X_test = future_df[test_features]
            
            # Handle null and inf values;
            if X_train.isnull().values.any():
                X_train = X_train.fillna(method='ffill')
            if X_test.isnull().values.any():
                X_test = X_test.fillna(method='ffill')
            if y_train.isnull().values.any():
                y_train = y_train.fillna(method='ffill')
            
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            y_train = y_train.replace([np.inf, -np.inf], 0)
            
            try:
                # Use ARIMA to forecast the external variable;
                model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
                model = model.fit()
                forecast_values = model.predict(
                    start=len(y_train), 
                    end=len(y_train)+len(X_test)-1, 
                    exog=X_test
                )
                forecast_values = pd.DataFrame(index=future_dates, data=forecast_values.values)
                forecast_values = forecast_values.reset_index(drop=False).rename(columns={'index': 'ds', 0: col})
            except Exception as e:
                print(f"Failed to forecast {col}: {str(e)}")
                # If ARIMA fails, use zeros as fallback;
                forecast_values = pd.DataFrame(index=future_dates, data=np.zeros(len(X_test)))
                forecast_values = forecast_values.reset_index(drop=False).rename(columns={'index': 'ds', 0: col})
            
            future_df = future_df.merge(forecast_values, how='left', on='ds')
            print(future_df)
        return future_df