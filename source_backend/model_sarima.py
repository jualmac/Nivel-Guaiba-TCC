"""
SARIMAModels: Scikit-Learn compatible wrapper for SARIMAX time series forecasting.
Uses pmdarima for efficient Stepwise Hyperparameter tuning.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union, Tuple
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from source_backend.metrics import nse as nash_sutcliffe_efficiency

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
                **kwargs
                ):
        self.model_name = 'sarima'
        self.random_state = random_state
        self.n_trials = n_trials 
        self.batch = batch
        self.steps = steps
        self.mode = mode
        self.kwargs = kwargs
        
        # Default initialization
        self.model = None
        self.y_pred = None
        self.X_train = None # Store for shape validation

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
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        
        # Data Cleaning;
        if isinstance(X, pd.DataFrame):
            # Select numeric and fill NaN with 0 (ARIMA cannot handle NaN in exog);
            self.X_train = X.select_dtypes(include=[np.number]).fillna(0)
        else:
            self.X_train = np.nan_to_num(X)
            
        # Ensure Target is numeric;
        y_clean = y.astype(float) if isinstance(y, pd.Series) else y.astype(float)
        
        # Fit Model;
        print(f"Fitting SARIMA (AutoARIMA)... Optimization: {optimize_hyperparameters}")
        
        # We use a try-except block because SARIMA is prone to LinAlgErrors with high feature counts
        try:
            if optimize_hyperparameters:
                self.model = pm.auto_arima(
                    y=y_clean,
                    X=self.X_train,
                    start_p=1, start_q=1,
                    max_p=3, max_q=3,
                    m=12,                               # Seasonality (Monthly) - Adjust if needed;
                    start_P=0, seasonal=True,
                    d=None,                             # Let model determine 'd';
                    D=1,                                # Force seasonal difference if needed, or set None;
                    trace=True,                         # Prints progress;
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True,                      # Performance -> Avoids a full grid search;
                    n_jobs=-1,                          # Parallelize grid search (if stepwise=False);
                    random_state=self.random_state,
                    max_order=None                      # Allow wider search if needed;
                )
            else:
                # Fast fallback for no optimization;
                self.model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                self.model.fit(y_clean, X=self.X_train)
                
            print(f"SARIMA Best Fit Order: {self.model.order}, Seasonal: {self.model.seasonal_order}")
            
        except Exception as e:
            print(f"SARIMA Auto-Fit Failed: {e}")
            # Fallback to a very simple model to prevent pipeline crash
            print("Falling back to simple ARIMA(1,0,0) without Exog due to failure.")
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
        
        # Prepare Exogenous var;
        X_test_clean = None
        if self.X_train is not None:
            if isinstance(X_test, pd.DataFrame):
                X_test_clean = X_test.select_dtypes(include=[np.number]).fillna(0)
            else:
                X_test_clean = np.nan_to_num(X_test)

        try:
            # Predict;
            if X_test_clean is not None:
                self.y_pred = self.model.predict(n_periods=n_steps, X=X_test_clean)
            else:
                self.y_pred = self.model.predict(n_periods=n_steps)
                
        except Exception as e:
            print(f"SARIMA Prediction failed: {e}")
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
        return {
            "rmse": rmse,
            "mae": mae,
            "nse": nse,
            "r2": r2
        }