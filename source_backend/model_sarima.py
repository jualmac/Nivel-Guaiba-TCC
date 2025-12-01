"""
SARIMAModels: Scikit-Learn compatible wrapper for SARIMAX time series forecasting.

This module provides a SARIMAModels class that wraps statsmodels.tsa.statespace.sarimax.SARIMAX
to be compatible with Scikit-Learn Pipelines. It includes internal grid search for hyperparameter
optimization and handles exogenous variables for SARIMAX models.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from source_backend.metrics import nse as nash_sutcliffe_efficiency

########################################################################################################################
#                                                                  
# MODEL
#
########################################################################################################################
class SARIMAModels(BaseEstimator, RegressorMixin):
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) model wrapper for time series forecasting.
    
    Inherits from BaseEstimator and RegressorMixin to ensure full Scikit-Learn Pipeline compatibility.
    Supports exogenous variables (SARIMAX) and includes internal grid search for hyperparameter optimization.
    
    Parameters:
        random_state (int): Random seed for reproducibility (default: 42).
        n_trials (int): Number of optimization trials (kept for API consistency, default: 10).
        batch (int): Batch size (kept for API consistency with other models, default: 128).
        steps (int): Prediction horizon in time steps (default: 12).
        mode (str): Training device mode, e.g., 'CPU' (default: 'CPU').
        **kwargs: Additional keyword arguments passed to the model.
    
    Attributes:
        order (tuple): ARIMA order (p, d, q) for non-seasonal components.
        seasonal_order (tuple): Seasonal order (P, D, Q, s) for seasonal components.
        trend (str): Trend component ('c'=constant, 't'=linear, 'n'=none, 'ct'=both).
        model_results (SARIMAXResultsWrapper): Fitted SARIMAX model results.
        y_pred (np.ndarray): Latest predictions from the model.
        X (pd.DataFrame): Training exogenous variables.
        y (pd.Series): Training target variable.
    """
    def __init__(self,
                random_state: int = 42,
                n_trials: int = 10,
                 batch: int = 128,
                 steps: int = 12,
                 mode: str = 'CPU',
                **kwargs
                ):
        # Model identification and configuration
        self.model_name = 'sarima'
        self.random_state = random_state
        self.n_trials = n_trials  # Kept for API consistency with other models
        self.batch = batch  # Kept for API consistency with other models
        self.steps = steps  # Prediction horizon
        self.mode = mode
        self.kwargs = kwargs
        
        # Default SARIMA parameters: ARIMA(1,1,1) x SARIMA(1,1,1,12) with constant trend
        self.order = (1, 1, 1)  # (p, d, q): autoregressive, differencing, moving average
        self.seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s): seasonal components with period 12
        self.trend = 'c'  # Constant trend term
        
        # Model state variables (initialized as None, set during fit)
        self.model_results: Optional[SARIMAXResultsWrapper] = None
        self.y_pred = None
        self.X = None
        self.y = None

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            optimize_hyperparameters: bool = True,
            **kwargs
            ):
        """
        Fit the SARIMAX model on training data.
        
        Performs data validation, optional hyperparameter optimization via grid search,
        constant detection in exogenous variables, and final model fitting. Compatible with
        Scikit-Learn Pipeline.fit() interface.
        
        Parameters:
            X (pd.DataFrame): Exogenous variables (features) for training.
            y (pd.Series): Endogenous variable (target) for training.
            optimize_hyperparameters (bool): If True, run grid search to find best parameters (default: True).
            **kwargs: Additional keyword arguments (for Pipeline compatibility).
        
        Returns:
            self: Returns self for method chaining (Scikit-Learn convention).
        
        Raises:
            ValueError: If X or y is None.
            Exception: If SARIMAX model fitting fails.
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        
        # Store training data (Pipeline may convert to numpy, but we preserve original structure)
        self.X = X  # Exogenous variables (features)
        self.y = y  # Endogenous variable (target)

        # Data type cleaning
        # Pipeline may pass datetime or non-numeric columns; SARIMAX requires numeric data only
        if self.X is not None and isinstance(self.X, pd.DataFrame):
            self.X = self.X.select_dtypes(include=[np.number])
        
        # Convert target to float if possible (handles string or object dtypes)
        try:
            self.y = self.y.astype(float)
        except ValueError:
            pass  # If conversion fails, proceed with original dtype

        # Hyperparameter optimization
        if optimize_hyperparameters:
            print("Running Simple Grid Search for SARIMA...")
            self._run_grid_search()  # Updates self.order, self.seasonal_order, self.trend
        else:
            print(f"Using default SARIMA parameters: Order={self.order}, Seasonal={self.seasonal_order}, Trend={self.trend}")

        # Check for constant columns in exogenous variables
        # SARIMAX raises ValueError if exog contains constants and trend includes 'c'
        self._check_constants_in_exog()

        # Fit final SARIMAX model with selected parameters
        print(f"Training SARIMAX with Order: {self.order} x {self.seasonal_order}, Trend: {self.trend}")
        
        try:
            # Create SARIMAX model instance
            model = SARIMAX(
                endog=self.y,  # Endogenous (target) variable
                exog=self.X,  # Exogenous (feature) variables
                order=self.order,  # Non-seasonal ARIMA order
                seasonal_order=self.seasonal_order,  # Seasonal ARIMA order
                trend=self.trend,  # Trend component
                enforce_stationarity=False,  # Allow non-stationary data
                enforce_invertibility=False  # Allow non-invertible models
            )
            # Fit model (disp=False suppresses convergence messages)
            self.model_results = model.fit(disp=False)
        except Exception as e:
            print(f"SARIMAX Fit Failed: {e}")
            raise e
            
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted SARIMAX model.
        
        Requires exogenous variables (X_test) for each forecast step. The number of forecast
        steps equals the length of X_test. Compatible with Scikit-Learn Pipeline.predict() interface.
        
        Parameters:
            X_test (pd.DataFrame): Exogenous variables for forecast horizon. Must have same
                number of rows as desired forecast steps.
        
        Returns:
            np.ndarray: Array of predicted values for the forecast horizon.
        
        Raises:
            ValueError: If model not fitted or X_test is None.
        """
        # Validate model state
        if self.model_results is None:
            raise ValueError("Model has not been fitted.")
        if X_test is None:
            raise ValueError("X_test cannot be None (Exogenous variables required).")

        # Clean test data: remove non-numeric columns (same preprocessing as fit)
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.select_dtypes(include=[np.number])

        # Number of forecast steps equals number of rows in X_test
        n_steps = len(X_test)
        
        # Generate forecast using fitted model
        try:
            # get_forecast() returns forecast object with predicted_mean, conf_int, etc.
            forecast = self.model_results.get_forecast(steps=n_steps, exog=X_test)
            self.y_pred = forecast.predicted_mean  # Extract point predictions
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Fallback to zeros if prediction fails (prevents pipeline crash)
            self.y_pred = np.zeros(n_steps)
        
        # Convert to numpy array if result is Pandas Series (for consistency)
        if isinstance(self.y_pred, pd.Series):
            self.y_pred = self.y_pred.values
            
        return self.y_pred

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Calculate evaluation metrics for model predictions.
        
        Computes RMSE, MAE, NSE (Nash-Sutcliffe Efficiency), and R² score. If y_pred is not
        provided, uses the latest predictions stored in self.y_pred.
        
        Parameters:
            y_true (pd.Series): True target values (ground truth).
            y_pred (Optional[pd.Series]): Predicted values. If None, uses self.y_pred (default: None).
        
        Returns:
            dict: Dictionary containing metric names and values:
                - "rmse": Root Mean Squared Error
                - "mae": Mean Absolute Error
                - "nse": Nash-Sutcliffe Efficiency coefficient
                - "r2": R-squared (coefficient of determination)
        """
        # Use stored predictions if y_pred not provided
        if y_pred is None:
            y_pred = self.y_pred

        # Calculate standard regression metrics
        rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        nse = nash_sutcliffe_efficiency(y_true=y_true, y_pred=y_pred)  # Custom metric for hydrology
        r2 = r2_score(y_true=y_true, y_pred=y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
            "nse": nse,
            "r2": r2
        }

    def _run_grid_search(self):
        """
        Internal grid search for hyperparameter optimization using TimeSeriesSplit cross-validation.
        
        Evaluates all parameter combinations in the grid using time series cross-validation.
        Selects parameters with lowest average RMSE across CV folds. Updates self.order,
        self.seasonal_order, and self.trend with best found parameters.
        
        Uses a simplified parameter grid to balance exploration and computational cost.
        Fits models with reduced maxiter for speed during grid search.
        """
        # Define simplified parameter grid: small search space for efficiency
        param_grid = {
            'p': [1, 2],  # Autoregressive order
            'd': [0, 1],  # Differencing order
            'q': [1, 2],  # Moving average order
            'P': [0, 1],  # Seasonal autoregressive order
            'D': [0, 1],  # Seasonal differencing order
            'Q': [0, 1],  # Seasonal moving average order
            's': [12],  # Seasonal period (fixed at 12 for monthly seasonality)
            'trend': ['c', 't']  # Constant or linear trend
        }
        
        grid = ParameterGrid(param_grid)  # Generate all parameter combinations
        tscv = TimeSeriesSplit(n_splits=5)  # Time series cross-validation (preserves temporal order)
        
        best_score = float('inf')  # Track best RMSE score
        best_params = None  # Track best parameter combination
        
        total_combos = len(list(grid))
        print(f"Grid Search: Evaluating {total_combos} combinations...")

        # Evaluate each parameter combination
        for i, params in enumerate(grid):
            print (i, params)
            scores = []  # Store CV scores for this parameter set
            try:
                # Cross-validation loop: split data maintaining temporal order
                for train_index, val_index in tscv.split(self.X):
                    # Handle both DataFrame and numpy array inputs (Pipeline may convert)
                    if isinstance(self.X, pd.DataFrame):
                        X_train_cv = self.X.iloc[train_index]
                        X_val_cv = self.X.iloc[val_index]
                    else:
                        X_train_cv = self.X[train_index]
                        X_val_cv = self.X[val_index]
                        
                    if isinstance(self.y, pd.Series):
                        y_train_cv = self.y.iloc[train_index]
                        y_val_cv = self.y.iloc[val_index]
                    else:
                        y_train_cv = self.y[train_index]
                        y_val_cv = self.y[val_index]
                    
                    # Adjust trend if constants detected in this CV fold
                    trend_cv = params['trend']
                    if self._has_constant(X_train_cv) and 'c' in trend_cv:
                        trend_cv = 'n' if trend_cv == 'c' else 't'  # Remove constant from trend

                    # Fit SARIMAX model with current parameters
                    model = SARIMAX(
                        endog=y_train_cv,
                        exog=X_train_cv,
                        order=(params['p'], params['d'], params['q']),
                        seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
                        trend=trend_cv,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    # Fit with reduced maxiter for speed (grid search doesn't need full convergence)
                    model_fit = model.fit(disp=False, maxiter=50)
                    
                    # Generate forecast for validation set
                    forecast = model_fit.get_forecast(steps=len(X_val_cv), exog=X_val_cv)
                    pred = forecast.predicted_mean
                    
                    # Calculate RMSE for this CV fold
                    score = root_mean_squared_error(y_val_cv, pred)
                    scores.append(score)
                
                # Average RMSE across all CV folds
                avg_score = np.mean(scores)
                
                # Update best parameters if this combination is better
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
                    
            except Exception:
                # Skip parameter combinations that fail (e.g., convergence issues)
                continue
        
        # Update model parameters with best found combination
        if best_params:
            print(f"Best Params: {best_params} (RMSE: {best_score:.4f})")
            self.order = (best_params['p'], best_params['d'], best_params['q'])
            self.seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
            self.trend = best_params['trend']
        else:
            # Fallback to defaults if no valid parameters found
            print("Grid Search found no valid parameters. Using defaults.")

    def _check_constants_in_exog(self):
        """
        Check for constant columns in exogenous variables and adjust trend parameter accordingly.
        
        SARIMAX raises ValueError if exogenous variables contain constants and trend includes 'c'.
        This method detects constants and removes the constant term from trend to avoid duplication.
        Called before final model fitting to prevent runtime errors.
        """
        if self.X is not None:
            has_constant = self._has_constant(self.X)
            
            if has_constant:
                # Remove constant from trend if detected (exog already provides constant)
                if self.trend == 'c':
                    print("Warning: Constant detected in exogenous variables. Changing trend from 'c' to 'n'.")
                    self.trend = 'n'  # No trend (constant provided by exog)
                elif self.trend == 'ct':
                    print("Warning: Constant detected in exogenous variables. Changing trend from 'ct' to 't'.")
                    self.trend = 't'  # Linear trend only (constant provided by exog)

    def _has_constant(self, X):
        """
        Check if input array/DataFrame contains any constant columns (columns with no variation).
        
        Parameters:
            X: Input data (pd.DataFrame or np.ndarray).
        
        Returns:
            bool: True if any column has constant values (no variation), False otherwise.
        """
        try:
            if isinstance(X, pd.DataFrame):
                # Check if any column has only one unique value (constant)
                return (X.nunique() <= 1).any()
            else:
                # For numpy arrays: check if peak-to-peak (max - min) is zero for any column
                return (np.ptp(X, axis=0) == 0).any()
        except Exception:
            # Return False on any error (safer fallback)
            return False
