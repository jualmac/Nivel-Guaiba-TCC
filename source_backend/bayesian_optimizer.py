"""
This module provides functionality to optimize regression models by tuning hyperparameters and evaluating 
performance metrics. The module implements various optimization techniques such as grid search, random search, or 
Bayesian optimization to find the best hyperparameters for regression models.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import json
import logging
import optuna
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import os

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
class OptimizeRegressor:
    """
    Performs Bayesian Optimization on specified regression models.

    Attributes
    ----------
    n_trials : int
        Number of trials for the optimization process.
    X_train : DataFrame
        Training data features.
    y_train : Series
        Training data target variable.
    logger : Logger
        Logger for logging messages.
    file_name : str
        Name of the file to save the best hyperparameters.
    """

    def __init__(
            self, 
            model_name: str, 
            n_trials: int, 
            X_train: DataFrame, 
            y_train: Series, 
            cpu: bool, 
            ):
        """
        Initializes the OptimizeRegressor class with the specified model and parameters.

        Parameters
        ----------
        n_trials : int
            Number of trials for the optimization process.
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data target variable.
        """
        # Initialize Logger;
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        self.model_name = model_name
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        self.random_state = 42
        self.n_jobs = -1
        self.cpu = cpu
        self.file_name = f"source_backend/parameters/params_{model_name}.json"

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization to tune hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single trial of an optimization experiment.

        Returns
        -------
        float
            The negative RMSE score for the given trial.
        """
        # XGBoost;
        if self.model_name == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "tree_method": 'hist',
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "tree_method": 'hist',  # Required for GPU acceleration
                "device": 'cpu'  # Default to CPU
                }
            if not self.cpu:  # Use GPU if not CPU mode
                params['device'] = 'cuda'
            model = XGBRegressor(**params)
            return self.evaluate(model)
        
        # LightGBM;
        elif self.model_name == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "num_threads": self.n_jobs if self.n_jobs != -1 else 0,
                "device": 'cpu'  # Default to CPU to avoid OpenCL device errors
            }
            if not self.cpu:  # Only use GPU if explicitly requested (cpu=False)
                params['device'] = 'gpu'
            model = LGBMRegressor(**params)
            return self.evaluate(model)
        
        # Random Forest;
        elif self.model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]), 
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                }
            model = RandomForestRegressor(**params)
            return self.evaluate(model)

        else:
            self.logger.error("Please provide a supported model: RandomForest (rf), XGBoost (xgb) or LightGBM (lgbm)")
            raise TypeError()

    def evaluate(self, model) -> float:
        """
        Evaluates the regression model using TimeSeriesSplit cross-validation.

        Parameters
        ----------
        model : The regression model to evaluate.

        Returns
        -------
        float
            The negative RMSE score across TimeSeriesSplit splits.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(
            model, 
            X=self.X_train, 
            y=self.y_train, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=self.n_jobs
        )
        return scores.mean()

    def optimize(self) -> dict:
        """
        Conducts Bayesian optimization to find the best hyperparameters for the specified model.
        Saves the best hyperparameters to a JSON file and returns them.
        
        Returns:
            dict: The best hyperparameters found during optimization
        """
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=3,
                interval_steps=1
            )
        )
        
        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=600  # 10 minutes timeout;
        )

        # Add model-specific parameters
        best_params = study.best_params.copy()
        best_params['random_state'] = self.random_state
        
        # Add n_jobs only for models that support it
        if self.model_name in ['xgb', 'rf']:
            best_params['n_jobs'] = self.n_jobs
        elif self.model_name == 'lgb':
            # LightGBM uses num_threads instead of n_jobs
            best_params['num_threads'] = self.n_jobs if self.n_jobs != -1 else 0
        
        # Ensure parameters directory exists
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        
        with open(self.file_name, "w") as f:
            json.dump(best_params, f, indent=4)

        self.logger.info(f"Best parameters saved to {self.file_name}")        
        return best_params