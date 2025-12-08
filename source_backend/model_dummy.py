"""
Lightweight baseline model using sklearn's DummyRegressor with mean strategy.
Provides the same interface (.fit, .predict, .metric) as other model classes
so it can be plugged into the shared training pipeline without special casing.
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from source_backend.metrics import (
    nse as nash_sutcliffe_efficiency,
    kge as kling_gupta_efficiency,
)

########################################################################################################################
#
# MODEL
#
########################################################################################################################
class DummyModels:
    def __init__(
        self,
        strategy: str = "mean",
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize DummyRegressor wrapper with pipeline-compatible signature.;
        """
        self.model_name = "dummy"
        self.strategy = strategy
        self.random_state = random_state
        self.log_mlflow = kwargs.get('log_mlflow', True)  # Default to True for backward compatibility; DUMMY doesn't use MLFlow but accepts for consistency;

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize_hyperparameters: bool = False,
        **kwargs,
    ):
        """
        Fit DummyRegressor on provided data.; hyperparameter flag ignored by design.;
        """
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        if (hasattr(X, "empty") and X.empty) or (hasattr(y, "empty") and y.empty):
            raise ValueError("Input data (X and y) cannot be empty.")

        self.X = X.copy()
        self.y = y.copy()

        # DummyRegressor provides a quick baseline using simple strategies.;
        self.model = DummyRegressor(strategy=self.strategy)
        self.model.fit(self.X, self.y)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted DummyRegressor.;
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        if X_test is None:
            raise ValueError("X_test cannot be None. Please provide test features.")

        X_test_processed = X_test.copy()
        self.y_pred = self.model.predict(X_test_processed)
        return self.y_pred

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics for baseline comparison.;
        """
        if y_pred is None:
            if not hasattr(self, "y_pred"):
                raise ValueError("No predictions available. Call predict() first or pass y_pred.")
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
