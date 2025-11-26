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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from source_backend.bayesian_optimizer import BayesianOptimization
from source_database.transformations import nature_encode
from source_backend.mlflow_utils import MLFlowHandler
from util import get_device_config, is_cpu_mode

########################################################################################################################
#
# MODEL
#
########################################################################################################################
class LightGBMModels:
    def __init__(self,
                X: Optional[pd.DataFrame] = None,
                y: Optional[pd.Series] = None,
                random_state: int = 42,
                n_trials: int = 10,
                batch: int = 128,
                steps: int = 12,
                **kwargs
                ):
        """
        """
        # Define arguments;
        self.model_name = 'lightgbm'
        self.random_state = random_state
        self.n_trials = n_trials
        self.batch = batch
        self.steps = steps
        self.mode = kwargs.get('mode', 'CPU')
        self.device_config = get_device_config(self.mode, self.model_name)
        
        # Create copy to avoid modifying the original datasets;
        self.X = X.copy() if X is not None else None
        self.y = y.copy() if y is not None else None
        
        # Standardize date column if data is provided;
        if self.X is not None:
            self._add_calendar_features()
    
    def fit(self, 
            X: Optional[pd.DataFrame] = None, 
            y: Optional[pd.Series] = None, 
            optimize_hyperparameters: bool = False
        ):
        """
        Fits the model. If X and y are provided, they update the internal datasets.
        Parameters:
            X: Features
            y: Target
            optimize_hyperparameters: If True, run Bayesian Optimization. If False, load best from MLflow.
        """
        if X is not None:
            self.X = X.copy()
        if y is not None:
            self.y = y.copy()
            
        if self.X is not None:
            self._add_calendar_features()
            
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
        self.model.fit(self.X, self.y)
        return self
    
    #TODO: Implement;
    def predict(self) -> pd.DataFrame:
        """
        """
        return None

    def _add_calendar_features(self) -> None:
        """Transforms the 'Data_Hora_Medicao' into features for GBM type of models"""
        if 'Data_Hora_Medicao' not in self.X.columns:
            raise ValueError(f"The dataset doesn't have a 'Data_Hora_Medicao' date column")
        
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