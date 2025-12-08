"""
SARIMAModels: Scikit-Learn compatible wrapper for SARIMAX time series forecasting.

Uses pmdarima for efficient Stepwise Hyperparameter tuning, with additional
performance optimizations for handling a high number of exogenous variables (X).
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
import warnings
import gc
from typing import Optional, Union, Tuple
import pmdarima as pm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from source_backend.metrics import (
    nse as nash_sutcliffe_efficiency,
    kge as kling_gupta_efficiency,
)

########################################################################################################################
#                                                                  
# MODEL
#
########################################################################################################################
class SARIMAModels(BaseEstimator, RegressorMixin):
    """
    SARIMA wrapper using AutoARIMA (pmdarima) for accelerated hyperparameter tuning.
    Compatible with Scikit-Learn Pipelines.
    """
    def __init__(self,
                random_state: int = 42,
                n_trials: int = 10, # Kept for compatibility, not used;
                batch: int = 128,
                steps: int = 12,
                mode: str = 'CPU',
                max_exog_features: int = 20, # SAFETY BRAKE: Hard limit on features to prevent crash
                search_sample_size: int = 10000, # Optimization: Limit samples for stepwise search
                **kwargs
                ):
        self.model_name = 'sarima'
        self.random_state = random_state
        self.n_trials = n_trials 
        self.batch = batch
        self.steps = steps
        self.mode = mode
        self.max_exog_features = max_exog_features
        self.search_sample_size = search_sample_size
        self.log_mlflow = kwargs.get('log_mlflow', True)  # Default to True for backward compatibility; SARIMA doesn't use MLFlow but accepts for consistency;
        self.kwargs = kwargs
        
        # Default initialization
        self.model = None
        self.y_pred = None
        self.X_train = None 
        self.feature_selector = None # To track which features we kept

    def _preprocess_exog(self, X, training: bool = True, y=None):
        """
        Handles Exogenous variables: Fills NaNs and limits number of features.
        """
        if X is None:
            return None
            
        # Standardize to numpy/numeric and fill NaNs (ARIMA requirement)
        if isinstance(X, pd.DataFrame):
            X_clean = X.select_dtypes(include=[np.number]).fillna(0).values
        else:
            X_clean = np.nan_to_num(X)
            
        # Feature Selection (Critical for stability)
        if training:
            # If we have more columns than allowed, select top K
            if X_clean.shape[1] > self.max_exog_features:
                print(f"[model_sarima.py] SARIMA: Reducing features from {X_clean.shape[1]} to {self.max_exog_features} for stability.")
                # Use f_regression to select features based on linear correlation with target (y)
                self.feature_selector = SelectKBest(score_func=f_regression, k=self.max_exog_features)
                X_reduced = self.feature_selector.fit_transform(X_clean, y)
                return X_reduced
            else:
                return X_clean
        else:
            # Transform Test data using stored selector
            if self.feature_selector is not None:
                return self.feature_selector.transform(X_clean)
            else:
                # If no selection was needed during training, use all
                return X_clean

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            optimize_hyperparameters: bool = True,
            **kwargs
            ):
        """
        Fit AutoARIMA model.
        
        Parameters:
            X (pd.DataFrame): Exogenous variables.
            y (pd.Series): Target variable.
            optimize_hyperparameters (bool): If True, runs stepwise search. 
                                             If False, fits a default ARIMA(1,1,1).
        """
        # optimize_hyperparameters = False
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
            
        # Ensure Target is numeric;
        y_clean = y.astype(np.float32) if isinstance(y, pd.Series) else y.astype(np.float32)
        
        # Data Cleaning and Feature Selection (The crash fix);
        self.X_train = self._preprocess_exog(X, training=True, y=y_clean)
        
        # Fit Model;
        print(f"[model_sarima.py] Fitting SARIMA (AutoARIMA)... Optimization: {optimize_hyperparameters}")
        print(f"[model_sarima.py] Exogenous features used: {self.X_train.shape[1] if self.X_train is not None else 0}")
        
        # We use a try-except block because SARIMA is prone to LinAlgErrors with high feature counts
        try:
            if optimize_hyperparameters:
                # OPTIMIZATION: Use subset for search if data is too large to prevent RAM explosion
                if len(y_clean) > self.search_sample_size:
                    print(f"[model_sarima.py] SARIMA: Using last {self.search_sample_size} samples for hyperparameter search to save memory.")
                    y_search = y_clean[-self.search_sample_size:]
                    X_search = self.X_train[-self.search_sample_size:] if self.X_train is not None else None
                else:
                    y_search = y_clean
                    X_search = self.X_train

                # Optimized Stepwise Search;
                search_model = pm.auto_arima(
                    y=y_search,
                    X=X_search,
                    start_p=1, start_q=1, start_d=1,
                    max_p=3, max_q=3, max_d=2,
                    max_order=None,
                    m=12,                                   # Seasonality (Monthly) - Adjust if needed;
                    start_P=0, seasonal=True,
                    max_P=1, max_Q=1,                       # Constrained seasonal range for speed;
                    D=1,                                    # Force seasonal difference if needed, or set None;;
                    test='kpss',                            # Faster stationarity test;
                    trace=True,                             # Prints progress;
                    error_action='ignore',      
                    suppress_warnings=True,     
                    stepwise=False,                         # Impacts performance -> Controls full grid search;
                    approximation=True,                     # Uses CSS instead of MLE for search;
                    maxiter=25,                             # Stop solver if not converging quickly;
                    n_jobs=1,                               # Set to 1 for stability with exog variables
                    random_state=self.random_state,
                )
                
                # Refit best model on FULL data
                print(f"[model_sarima.py] Refitting best order {search_model.order}{search_model.seasonal_order} on full dataset ({len(y_clean)} rows)...")
                self.model = pm.ARIMA(
                    order=search_model.order, 
                    seasonal_order=search_model.seasonal_order,
                    suppress_warnings=True
                )
                
                # Clear memory from search
                del search_model
                gc.collect()

                self.model.fit(X=self.X_train, y=y_clean)

            else:
                # Fast fallback for no optimization;
                self.model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), suppress_warnings=True)
                self.model.fit(y_clean, X=self.X_train)
                
            print(f"[model_sarima.py] SARIMA Best Fit Order: {self.model.order}, Seasonal: {self.model.seasonal_order}")
            
        except Exception as e:
            print(f"[model_sarima.py] SARIMA Auto-Fit Failed: {e}")
            # Fallback to a very simple model to prevent pipeline crash
            print("[model_sarima.py] Falling back to simple ARIMA(1,0,0) without Exog due to failure.")
            try:
                self.model = pm.ARIMA(order=(1, 0, 0), suppress_warnings=True)
                self.model.fit(y_clean) # Fit without exog as fallback
                self.X_train = None # Flag that we aren't using exog
            except:
                raise ValueError("Critical SARIMA failure. Data may be unsuitable.")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted AutoARIMA model.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        if X_test is None and self.X_train is not None:
             raise ValueError("X_test required for model trained with Exogenous variables.")

        n_steps = len(X_test)
        
        # Prepare Exogenous var (applies the same feature selection as in fit);
        # Only call preprocess if we actually used exog during fit (self.X_train is not None)
        X_test_clean = self._preprocess_exog(X_test, training=False) if self.X_train is not None else None

        try:
            # Predict;
            if X_test_clean is not None:
                self.y_pred = self.model.predict(n_periods=n_steps, X=X_test_clean)
            else:
                self.y_pred = self.model.predict(n_periods=n_steps)
                
        except Exception as e:
            print(f"[model_sarima.py] SARIMA Prediction failed: {e}")
            self.y_pred = np.zeros(n_steps)
        
        # Handle Series return type;
        if isinstance(self.y_pred, pd.Series):
            self.y_pred = self.y_pred.values
            
        return self.y_pred

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Calculate evaluation metrics.
        """
        if y_pred is None:
            y_pred = self.y_pred

        # Handle NaNs in prediction (rare but possible in SARIMA);
        if np.isnan(y_pred).any():
            y_pred = np.nan_to_num(y_pred)

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