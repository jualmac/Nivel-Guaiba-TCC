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
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from src.models.optimize_params import BayesianOptimization
from src.etl.transformations import nature_encode
from src.mlflow_utils import MLFlowHandler
from src.util import get_device_config, is_cpu_mode, configure_logging

logger = configure_logging(__name__)
from src.metrics import (
    nse as nash_sutcliffe_efficiency,
    kge as kling_gupta_efficiency,
)

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
        self.log_mlflow = kwargs.get('log_mlflow', True)  # Default to True for backward compatibility;
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            optimize_hyperparameters: bool = True,
            early_stopping: int = 50,
            feature_pipeline=None,
            X_raw: Optional[pd.DataFrame] = None,
            cv_n_splits: int = 5,
            cv_gap: int = 24,
            ):
        """
        Fits the model with the provided X and y, optionally using validation data for early stopping;
        Parameters:
            X: Features
            y: Target
            optimize_hyperparameters: If True, run Bayesian Optimization. If False, load best from MLflow.
            X_val: Optional validation features for early stopping
            y_val: Optional validation target for early stopping
        """
        # Validate X and y contents;
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        if (hasattr(X, 'empty') and X.empty) or (hasattr(y, 'empty') and y.empty):
            raise ValueError("Input data (X and y) cannot be empty.")

        # Create copy to avoid modifying the original datasets;
        self.X = X.copy()
        self.y = y.copy()
        self.X_raw = X_raw
        self.feature_pipeline = feature_pipeline
        self.cv_n_splits = cv_n_splits
        self.cv_gap = cv_gap
        
        # Standardize date column;
        if self.X is not None:
            self.X = self._add_calendar_features(X=self.X)
        
        # Process validation data if provided;
        eval_set = None
        if X_val is not None and y_val is not None:
            # Validate validation data;
            if (hasattr(X_val, 'empty') and X_val.empty) or (hasattr(y_val, 'empty') and y_val.empty):
                raise ValueError("Provided evaluation data (X_val and y_val) cannot be empty.")
            
            # Process validation features (add calendar features);
            X_val_processed = self._add_calendar_features(X=X_val.copy())
            eval_set = [(X_val_processed, y_val)]
            logger.info("Using validation set for early stopping.")
            
        # Hyperparameter handling;
        best_params = {}
        if optimize_hyperparameters:
            logger.info("Running Bayesian Optimization...")
            best_params = self._get_best_params()
            
        else:
            logger.info("Loading best parameters from MLflow...")
            mlflow_handler = MLFlowHandler()
            best_params = mlflow_handler.load_best_params(metric_name="score", mode="max")
            
            if not best_params:
                logger.info("No best params found in MLflow, using defaults.")
        
        # Convert numeric params that might be strings from MLflow;
        for k, v in best_params.items():
            try:
                if float(v).is_integer():
                    best_params[k] = int(float(v))
                else:
                    best_params[k] = float(v)
            except (ValueError, TypeError):
                pass

        # Quiet LightGBM warnings by default unless explicitly overridden;
        best_params.setdefault("verbosity", -1)

        # Add early stopping parameters if validation set is provided;
        if eval_set is not None:
            # Set early stopping parameters if not already in best_params;
            if 'early_stopping_rounds' not in best_params:
                best_params['early_stopping_rounds'] = early_stopping

            # Use KGE as validation metric and keep LightGBM aware it should maximize;
            def _lightgbm_kge_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
                score = kling_gupta_efficiency(y_true=y_true, y_pred=y_pred)
                return 'kge', score, True

            best_params['eval_metric'] = _lightgbm_kge_eval

        # Create model with params;
        logger.info("Training LightGBM with params: %s", best_params)
        self.model = LGBMRegressor(**best_params)

        # Fit model with Training data (and validation set for early stopping if provided);
        if eval_set is not None:
            self.model.fit(self.X, self.y, eval_set=eval_set)
        else:
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

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Calculates evaluation metrics for the model predictions.
        
        Parameters:
            y_true: True target values for test set
            y_pred: Predicted target values for test set
        
        Returns:
            Dict[str, float]: Dictionary containing rmse, mae, nse, kge and r2 metrics
        """
        if y_pred is None:
            y_pred = self.y_pred

        # Calculate all metrics;
        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        nse = nash_sutcliffe_efficiency(y_true=y_true, y_pred=y_pred)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        kge = kling_gupta_efficiency(y_true=y_true, y_pred=y_pred)
        
        # Return metrics as dictionary for easier logging;
        return {
            "rmse": rmse,
            "mae": mae,
            "nse": nse,
            "r2": r2,
            "kge": kge,
        }

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
            mode=self.mode,
            feature_pipeline=self.feature_pipeline,
            X_raw=self.X_raw,
            n_splits=self.cv_n_splits,
            gap=self.cv_gap,
            postprocess_fn=self._add_calendar_features,
            log_mlflow=self.log_mlflow
        )
        return optimizer.optimize()