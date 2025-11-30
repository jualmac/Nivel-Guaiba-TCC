"""
Defines the SARIMAModels class for automated time series forecasting.
Wraps statsmodels.tsa.statespace.sarimax.SARIMAX to be compatible with 
Scikit-Learn Pipelines and the existing project structure.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from source_backend.mlflow_utils import MLFlowHandler
from source_backend.metrics import nse as nash_sutcliffe_efficiency

########################################################################################################################
#                                                                  
# MODEL
#
########################################################################################################################
class SARIMAModels:
    def __init__(self,
                random_state: int = 42,
                n_trials: int = 10,
                 batch: int = 128, # Unused in SARIMA, kept for API consistency
                 steps: int = 12,  # Prediction horizon
                 mode: str = 'CPU',
                **kwargs
                ):
        """
        Initialize the SARIMA model wrapper.
        """
        self.model_name = 'sarima'
        self.random_state = random_state
        self.n_trials = n_trials
        self.steps = steps
        self.mode = mode
        
        # Default order if optimization fails or is skipped
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 12) 
        self.trend = 'c'
        
        self.model_results: Optional[SARIMAXResultsWrapper] = None

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            optimize_hyperparameters: bool = True,
            early_stopping: int = 50, # SARIMA doesn't support early stopping in the GBM sense
            ):
        """
        Fits the SARIMAX model. Compatible with sklearn Pipeline.
        """
        # 1. Validation
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        
        # Scikit-Learn Pipeline usually converts X to numpy array, stripping the Index.
        # We store them for fit.
        self.X = X # Exogenous variables
        self.y = y # Endogenous variable (Target)

        # 2. Hyperparameter Handling
        best_params = {}
        if optimize_hyperparameters:
            print("Running Bayesian Optimization for SARIMA...")
            # Note: SARIMA optimization is slower than GBMs
            best_params = self._get_best_params()
        else:
            print("Loading best parameters from MLflow...")
            mlflow_handler = MLFlowHandler()
            best_params = mlflow_handler.load_best_params(metric_name="sarima_best_rmse", mode="min")
            
            if not best_params:
                print("No best params found, using defaults.")
                best_params = {
                    'p': 1, 'd': 1, 'q': 1,
                    'P': 1, 'D': 1, 'Q': 1, 's': 12,
                    'trend': 'c'
                }

        # Parse flattened params back to tuples
        try:
            self.order = (
                int(best_params.get('p', 1)), 
                int(best_params.get('d', 1)), 
                int(best_params.get('q', 1))
            )
            self.seasonal_order = (
                int(best_params.get('P', 1)), 
                int(best_params.get('D', 1)), 
                int(best_params.get('Q', 1)), 
                int(best_params.get('s', 12))
            )
            self.trend = best_params.get('trend', 'c')
        except Exception as e:
            print(f"Error parsing params: {e}. Using defaults.")

        print(f"Training SARIMAX with Order: {self.order} x {self.seasonal_order}")

        # 3. Fit Model
        # SARIMAX requires strict handling of indices if available, but works with arrays too.
        # enforce_stationarity=False allows fitting on wider range of data without crashing.
        model = SARIMAX(
            endog=self.y,
            exog=self.X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Disp=False suppresses convergence messages
        self.model_results = model.fit(disp=False)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the SARIMAX model.
        In this pipeline context, X_test is treated as the 'exogenous' variables 
        for the future time steps.
        """
        if self.model_results is None:
            raise ValueError("Model has not been fitted.")
        if X_test is None:
            raise ValueError("X_test cannot be None (Exogenous variables required).")

        # Determine number of steps to forecast based on length of X_test
        n_steps = len(X_test)
        
        # Forecast
        # We use get_forecast() which is meant for out-of-sample prediction.
        # We must provide the exogenous variables (X_test) for the forecast horizon.
        forecast = self.model_results.get_forecast(steps=n_steps, exog=X_test)
        
        self.y_pred = forecast.predicted_mean
        
        # Handle cases where result is a Pandas Series
        if isinstance(self.y_pred, pd.Series):
            self.y_pred = self.y_pred.values
            
        return self.y_pred

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Calculates evaluation metrics. Identical to GBM/LSTM implementations.
        """
        if y_pred is None:
            y_pred = self.y_pred

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

    def _get_best_params(self) -> dict:
        """
        Calls the optimization module.
        """
        # Ensure we pass DataFrame/Series for column access if possible (Though Pipeline might have converted X to numpy)
        X_in = self.X
        if isinstance(X_in, np.ndarray):
            # If pipeline stripped dataframe, we can't recover column names easily,
            # but BayesianOptimization handles numpy input for SARIMAX check.
            pass

        # Import here to avoid circular import;
        from source_backend.optimize_params import BayesianOptimization
        
        optimizer = BayesianOptimization(
            model_name=self.model_name,
            n_trials=self.n_trials, 
            X_train=X_in,
            y_train=self.y,
            mode=self.mode
        )
        return optimizer.optimize()