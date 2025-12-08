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
from optuna.trial import TrialState
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gc
import mlflow
from util import get_device_config
from source_backend.metrics import kge

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
class BayesianOptimization:
    """
    Performs Bayesian Optimization on specified regression models.
    Optimization maximizes Kling Gupta Efficiency (KGE), which ranges from (-inf, 1] and peaks at 1.0.

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
            feature_pipeline=None,
            X_raw: DataFrame | None = None,
            n_splits: int = 3,
            gap: int = 24,
            postprocess_fn=None,
            log_mlflow: bool = True,
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
        self.feature_pipeline = feature_pipeline
        self.X_raw = X_raw
        self.n_splits = n_splits
        self.gap = gap
        self.postprocess_fn = postprocess_fn
        self.log_mlflow = log_mlflow

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
            The KGE score for the given trial (higher is better; max is 1.0).
        """
        # XGBoost;
        if self.model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
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
                "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.05),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
                "random_state": self.random_state,
                "num_threads": self.n_jobs if self.n_jobs != -1 else 0,
            }
            
            # Get device config;
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
                "num_layers": trial.suggest_int("num_layers", 1, 3), 
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epochs": 50, 
                "sequence_length": trial.suggest_categorical("sequence_length", [12, 24, 36, 48, 72, 96, 168]) 
            }
            return self.evaluate_lstm(params)

        else:
            self.logger.error("Please provide a supported model: XGBoost (xgboost), LightGBM (lightgbm) or LSTM (lstm)")
            raise TypeError()

    def evaluate_lstm(self, params) -> float:
        """
        Evaluates the LSTM model using TimeSeriesSplit cross-validation.
        Custom implementation for PyTorch model returning mean KGE across splits.
        """
        # Import here to avoid circular import;
        from source_backend.model_lstm import _LSTMRegressor as LSTMRegressor
        
        # Use raw features when available to refit preprocessing per fold;
        X_source = self.X_raw if self.X_raw is not None else self.X_train
        y_source = self.y_train

        # Sort once to keep temporal order consistent;
        if isinstance(X_source, pd.DataFrame) and 'Data_Hora_Medicao' in X_source.columns:
            X_source = X_source.sort_values('Data_Hora_Medicao')
            if isinstance(y_source, (pd.Series, pd.DataFrame)):
                y_source = y_source.loc[X_source.index]

        # Define Splits;
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        scores = []
        
        # Device
        device = torch.device('cuda' if self.mode in ['GPU', 'CUDA'] and torch.cuda.is_available() else 'cpu')
        
        # Helper to slice arrays/DataFrames;
        def _slice(data, idx):
            return data.iloc[idx] if hasattr(data, "iloc") else data[idx]

        for train_index, val_index in tscv.split(X_source):
            # Split data
            X_train_fold = _slice(X_source, train_index)
            X_val_fold = _slice(X_source, val_index)
            y_train_fold = _slice(y_source, train_index)
            y_val_fold = _slice(y_source, val_index)

            # Fit feature pipeline on the fold to avoid leakage;
            if self.feature_pipeline is not None:
                fold_pipeline = clone(self.feature_pipeline)
                X_train_fold = fold_pipeline.fit_transform(X_train_fold, y_train_fold)
                X_val_fold = fold_pipeline.transform(X_val_fold)

            # Remove non-numeric columns and enforce float dtype;
            if isinstance(X_train_fold, pd.DataFrame):
                X_train_fold = X_train_fold.select_dtypes(include=[np.number])
                X_val_fold = X_val_fold.select_dtypes(include=[np.number])
            
            X = np.asarray(X_train_fold, dtype=np.float64)
            y = np.asarray(y_train_fold, dtype=np.float64)
            X_val_np = np.asarray(X_val_fold, dtype=np.float64)
            y_val_np = np.asarray(y_val_fold, dtype=np.float64)
            
            # Create sequences (simplified version of _create_sequences logic)
            seq_len = params['sequence_length']
            
            def create_seq(data_x, data_y):
                seqs, targs = [], []
                for i in range(len(data_x) - seq_len):
                    seqs.append(data_x[i:i+seq_len])
                    targs.append(data_y[i+seq_len])
                return np.array(seqs), np.array(targs)
            
            if len(X) <= seq_len or len(X_val_np) <= seq_len:
                continue # Skip if not enough data
                
            X_train_seq, y_train_seq = create_seq(X, y)
            X_val_seq, y_val_seq = create_seq(X_val_np, y_val_np)
            
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
            # Collect validation predictions to compute KGE per fold;
            y_val_true_all = []
            y_val_pred_all = []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    y_val_true_all.append(targets.cpu().numpy().ravel())
                    y_val_pred_all.append(outputs.cpu().numpy().ravel())
            
            if y_val_true_all and y_val_pred_all:
                y_true_np = np.concatenate(y_val_true_all)
                y_pred_np = np.concatenate(y_val_pred_all)
                kge_score = kge(y_true=y_true_np, y_pred=y_pred_np)
                scores.append(kge_score)
            
            # Cleanup memory
            del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset
            del X_train_seq, y_train_seq, X_val_seq, y_val_seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect() 
        return np.mean(scores) if scores else -float('inf')

    def evaluate(self, model) -> float:
        """
        Evaluates the regression model using TimeSeriesSplit cross-validation with proper
        refitting of preprocessing inside each fold to avoid leakage.
        Returns the mean KGE across splits (higher is better; max is 1.0).
        """
        # Select raw data when provided so scalers/encoders are refit per fold;
        X_source = self.X_raw if self.X_raw is not None else self.X_train
        y_source = self.y_train

        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        scores = []

        # Helper to index DataFrame/ndarray consistently;
        def _slice(data, idx):
            return data.iloc[idx] if hasattr(data, "iloc") else data[idx]

        for train_idx, val_idx in tscv.split(X_source):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue  # Skip degenerate splits;

            X_train_fold = _slice(X_source, train_idx)
            X_val_fold = _slice(X_source, val_idx)
            y_train_fold = _slice(y_source, train_idx)
            y_val_fold = _slice(y_source, val_idx)

            # Fit feature pipeline on the fold to prevent leakage;
            if self.feature_pipeline is not None:
                fold_pipeline = clone(self.feature_pipeline)
                X_train_transformed = fold_pipeline.fit_transform(X_train_fold, y_train_fold)
                X_val_transformed = fold_pipeline.transform(X_val_fold)
            else:
                X_train_transformed = X_train_fold
                X_val_transformed = X_val_fold

            # Optional postprocessing hook (e.g., calendar features) after pipeline;
            if self.postprocess_fn is not None:
                X_train_transformed = self.postprocess_fn(X_train_transformed.copy())
                X_val_transformed = self.postprocess_fn(X_val_transformed.copy())

            model_fold = clone(model)
            model_fold.fit(X_train_transformed, y_train_fold)
            preds = model_fold.predict(X_val_transformed)

            y_true_np = np.asarray(y_val_fold, dtype=float).ravel()
            preds_np = np.asarray(preds, dtype=float).ravel()

            # Compute KGE for the validation fold; 
            kge_score = kge(y_true=y_true_np, y_pred=preds_np)
            scores.append(kge_score)
        return np.mean(scores) if scores else -float('inf')

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
            # timeout=600  # 10 minutes timeout;
        )

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise ValueError("No trials are completed yet; check preprocessing or model errors during CV.")

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
        
        # Log the final best metric and corresponding parameters to the current active MLflow run (only if logging is enabled);
        if self.log_mlflow:
            mlflow.log_metric(f"train_best_kge", study.best_value)
            mlflow.log_params(best_params)
            self.logger.info(f"Best parameters logged to MLflow for {self.model_name}.")
        else:
            self.logger.info(f"MLflow logging disabled. Best parameters found for {self.model_name}.")
        
        return best_params