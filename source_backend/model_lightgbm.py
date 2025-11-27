"""
Defines the LightGBMModels class for automated time series forecasting using LightGBM. This class facilitates the 
initialization with a dataset and model configurations, supporting both general and item-specific predictions. The main 
functionality includes creating the LightGBM model, generating predictions, and extracting optimal hyperparameters while 
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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from source_backend.optimize_params import BayesianOptimization
from source_database.transformations import nature_encode
from source_backend.mlflow_utils import MLFlowHandler
from util import get_device_config, is_cpu_mode
from source_backend.metrics import nse as nash_sutcliffe_efficiency

########################################################################################################################
#
# MODEL
#
########################################################################################################################
class LightGBMModels:
    def __init__(self,
                random_state: int = 42,
                n_trials: int = 10,
                batch: int = 128,
                steps: int = 12,
                mode: str = 'CPU',
                **kwargs
                ):
        """
        Initialize the model by defining the variables;
        """
        # Define arguments;
        self.model_name = 'lightgbm'
        self.random_state = random_state
        self.n_trials = n_trials
        self.batch = batch
        self.steps = steps
        self.mode = mode
        self.device_config = get_device_config(self.mode, self.model_name)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparameters: bool = True):
        """
        Fits the model with the provided X and y;
        Parameters:
            X: Features
            y: Target
            optimize_hyperparameters: If True, run Bayesian Optimization. If False, load best from MLflow.
        """
        # Validate existence and content;
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        if (hasattr(X, 'empty') and X.empty) or (hasattr(y, 'empty') and y.empty):
            raise ValueError("Input data (X and y) cannot be empty.")

        # Create copy to avoid modifying the original datasets;
        self.X = X.copy()
        self.y = y.copy()
        
        # Standardize date column;
        if self.X is not None:
            self.X = self._add_calendar_features(X=self.X)
            
        # Hyperparameter handling;
        best_params = {}
        if optimize_hyperparameters:
            print("Running Bayesian Optimization...")
            best_params = self._get_best_params()
            
        else:
            print("Loading best parameters from MLflow...")
            mlflow_handler = MLFlowHandler()
            best_params = mlflow_handler.load_best_params(metric_name="score", mode="max")
            
            if not best_params:
                print("No best params found in MLflow, using defaults.")
        
        # Convert numeric params that might be strings from MLflow;
        for k, v in best_params.items():
            try:
                if float(v).is_integer():
                    best_params[k] = int(float(v))
                else:
                    best_params[k] = float(v)
            except (ValueError, TypeError):
                pass

        # Create model with params;
        print(f"Training LightGBM with params: {best_params}")
        self.model = LGBMRegressor(**best_params)

        # Fit model with Training data;
        self.model.fit(self.X, self.y)
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts the model with the provided test dataset.
        
        Parameters:
            X_test: Test features DataFrame. Must contain 'Data_Hora_Medicao' column.
        
        Returns:
            np.ndarray: Model predictions.
        
        Raises:
            ValueError: If model has not been fitted or X_test is None.
        """
        # Validate model and dataframe;
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        if X_test is None:
            raise ValueError("X_test cannot be None. Please provide test features.")
        
        # Create copy to avoid modifying the original dataset;
        X_test_processed = X_test.copy()
        
        # Apply calendar feature engineering (same as training);
        X_test_processed = self._add_calendar_features(X=X_test_processed)

        # Make predictions;
        self.y_pred = self.model.predict(X_test_processed)
        return self.y_pred

    def score(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Calculates the Nash-Sutcliffe Efficiency (NSE) score for the model predictions.
        
        Parameters:
            y_true: True target values for test set
            y_pred: Predicted target values for test set
        
        Returns:
            float: NSE score (higher is better, range: -inf to 1.0)
        """
        if y_pred == None:
            y_pred = self.y_pred

        # # Calculate NSE using hydroeval;
        # from hydroeval import evaluator, nse
        # nse_score = evaluator(nse, simulations=y_pred, evaluation=y_true, axis=0)
        
        # Calculate Scores;
        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        nse = nash_sutcliffe_efficiency(y_true=y_true, y_pred=y_pred)
        return rmse, mae, nse

    def _add_calendar_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the 'Data_Hora_Medicao' into features for GBM type of models.
        Works for both training and test datasets.
        
        Parameters:
            X: DataFrame containing 'Data_Hora_Medicao' column.
        
        Returns:
            pd.DataFrame: DataFrame with calendar features engineered and original date column removed.
        """
        if 'Data_Hora_Medicao' not in X.columns:
            raise ValueError(f"The dataset doesn't have a 'Data_Hora_Medicao' date column")
        
        # Create copy to avoid modifying original DataFrame;
        X_cpy = X.copy()
        
        # Extract datetime components;
        ts = pd.to_datetime(X_cpy['Data_Hora_Medicao'])
        
        # Year as regular numeric feature (not cyclical - it doesn't repeat);
        X_cpy['year'] = ts.dt.year.astype(float)
        
        # Cyclical features that benefit from nature_encode;
        X_cpy['month'] = ts.dt.month
        X_cpy['day'] = ts.dt.day
        X_cpy['hour'] = ts.dt.hour
        X_cpy['dayofyear'] = ts.dt.dayofyear
        X_cpy['dayofweek'] = ts.dt.dayofweek
        
        # Apply cyclical encoding to periodic features;
        cyclical_features = [
            ('month', 12),       # Monthly cycle;
            ('hour', 24),         # Daily cycle;
            ('dayofyear', 365),  # Annual cycle;
            ('dayofweek', 7),    # Weekly cycle;
            ('day', ts.dt.days_in_month) # Monthly cycle (varies by month length);
        ]
        
        for col, period in cyclical_features:
            nature_encode(df=X_cpy, col=col, div_period=period)
            X_cpy.drop(columns=[col], inplace=True)
        
        # Drop the original datetime column;
        X_cpy.drop(columns=['Data_Hora_Medicao'], inplace=True)
        return X_cpy

    def _get_best_params(self) -> dict:
        """
        Performs Bayesian optimization to find the best hyperparameters.
        
        Parameters:
            - X_train: Training features
            - y_train: Training target
            
        Returns:
            - dict: Best hyperparameters
        """
        # Get best parameters from optimizer;
        optimizer = BayesianOptimization(
            model_name=self.model_name,
            n_trials=self.n_trials, 
            X_train=self.X,
            y_train=self.y,
            mode=self.mode
        )
        return optimizer.optimize()