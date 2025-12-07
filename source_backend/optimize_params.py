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
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gc
import mlflow
from util import get_device_config

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
class BayesianOptimization:
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
            mode: str = 'CPU', 
        ):
        """
        Initializes the BayesianOptimization class with the specified model and parameters.

        Parameters
        ----------
        n_trials : int
            Number of trials for the optimization process.
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data target variable.
        mode : str
            Training mode - 'CPU', 'GPU', or 'CUDA'.
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
        self.mode = mode

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
        if self.model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "subsample": trial.suggest_float("subsample", 0.3, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                }
            
            # Get device config
            device_config = get_device_config(self.mode, 'xgboost')
            params.update(device_config)

            model = XGBRegressor(**params)
            return self.evaluate(model)
        
        # LightGBM;
        elif self.model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "subsample": trial.suggest_float("subsample", 0.3, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
                "random_state": self.random_state,
                "num_threads": self.n_jobs if self.n_jobs != -1 else 0,
            }
            
            # Get device config
            device_config = get_device_config(self.mode, 'lightgbm')
            params.update(device_config)

            model = LGBMRegressor(**params)
            return self.evaluate(model)

        # LSTM;
        elif self.model_name == "lstm":
            # Hyperparameters;
            params = {
                # Architecture Tuning;
                "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 6), 
                "dropout": trial.suggest_float("dropout", 0.0, 0.6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epochs": 50, 
                "sequence_length": trial.suggest_categorical("sequence_length", [12, 24, 36, 48, 72, 96, 168, 336]) 
            }
            return self.evaluate_lstm(params)

        else:
            self.logger.error("Please provide a supported model: XGBoost (xgboost), LightGBM (lightgbm) or LSTM (lstm)")
            raise TypeError()

    def evaluate_lstm(self, params) -> float:
        """
        Evaluates the LSTM model using TimeSeriesSplit cross-validation.
        Custom implementation for PyTorch model.
        """
        # Import here to avoid circular import;
        from source_backend.model_lstm import _LSTMRegressor as LSTMRegressor
        
        # Define Splits;        
        tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for deep learning speed
        scores = []
        
        # Prepare data - handle both DataFrame and numpy array inputs;
        if isinstance(self.X_train, pd.DataFrame):
            if 'Data_Hora_Medicao' in self.X_train.columns:
                X = self.X_train.sort_values('Data_Hora_Medicao').drop(columns=['Data_Hora_Medicao']).values
            else:
                X = self.X_train.values
        else:
            # Already a numpy array (from preprocessor pipeline)
            X = self.X_train
            
        # Handle y_train similarly
        if isinstance(self.y_train, pd.Series):
            y = self.y_train.values
        else:
            y = self.y_train
            
        # Remove non-numeric columns (e.g., Timestamps) from object arrays
        if hasattr(X, 'dtype') and X.dtype == object:
            numeric_cols = []
            for i in range(X.shape[1]):
                try:
                    # Check if the first element can be converted to float
                    float(X[0, i])
                    numeric_cols.append(i)
                except (ValueError, TypeError):
                    # Likely a Timestamp or non-numeric string; skip this column
                    continue
            
            # Filter X if columns were removed
            if len(numeric_cols) < X.shape[1]:
                X = X[:, numeric_cols]
            
        # Ensure proper dtype for PyTorch (convert object dtype to float64);
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Device
        device = torch.device('cuda' if self.mode in ['GPU', 'CUDA'] and torch.cuda.is_available() else 'cpu')
        
        for train_index, val_index in tscv.split(X):
            # Split data
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            
            # Create sequences (simplified version of _create_sequences logic)
            seq_len = params['sequence_length']
            
            def create_seq(data_x, data_y):
                seqs, targs = [], []
                for i in range(len(data_x) - seq_len):
                    seqs.append(data_x[i:i+seq_len])
                    targs.append(data_y[i+seq_len])
                return np.array(seqs), np.array(targs)
            
            if len(X_train_fold) <= seq_len or len(X_val_fold) <= seq_len:
                continue # Skip if not enough data
                
            X_train_seq, y_train_seq = create_seq(X_train_fold, y_train_fold)
            X_val_seq, y_val_seq = create_seq(X_val_fold, y_val_fold)
            
            if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                continue

            # Create PyTorch datasets manually (avoid importing TimeSeriesDataset)
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val_seq, dtype=torch.float32),
                torch.tensor(y_val_seq, dtype=torch.float32)
            )
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Model
            model = LSTMRegressor(
                input_size=X.shape[1],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                output_size=1,
                dropout=params['dropout']
            ).to(device)
            
            # Train
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            model.train()
            for epoch in range(10): # Reduced epochs for optimization speed
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)
            
            scores.append(-np.sqrt(val_loss / len(val_dataset))) # Negative RMSE
            
            # Cleanup memory
            del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset
            del X_train_seq, y_train_seq, X_val_seq, y_val_seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        return np.mean(scores) if scores else -float('inf')

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

        # Add model-specific parameters;
        best_params = study.best_params.copy()
        best_params['random_state'] = self.random_state
        
        # Add n_jobs only for models that support it;
        if self.model_name in ['xgboost', 'random_forest']:
            best_params['n_jobs'] = self.n_jobs
        elif self.model_name == 'lightgbm':
            # LightGBM uses num_threads instead of n_jobs;
            best_params['num_threads'] = self.n_jobs if self.n_jobs != -1 else 0
        elif self.model_name == 'lstm':
            pass # LSTM doesn't use n_jobs parameter in this context;
        
        # Log the final best metric and corresponding parameters to the current active MLflow run;
        mlflow.log_metric(f"train_best_rmse", study.best_value)
        mlflow.log_params(best_params)

        self.logger.info(f"Best parameters logged to MLflow for {self.model_name}.")
        return best_params