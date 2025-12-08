"""
Defines the LSTMModels class for automated time series forecasting using PyTorch. 
This class acts as a wrapper to make a PyTorch LSTM compatible with Scikit-Learn Pipelines 
and mirrors the structure of the existing XGBoost/LightGBM models.
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from source_backend.mlflow_utils import MLFlowHandler
from util import get_device_config
from source_backend.metrics import (
    nse as nash_sutcliffe_efficiency,
    kge as kling_gupta_efficiency,
)

########################################################################################################################
#
# INNER PYTORCH MODULE
#
########################################################################################################################
class _LSTMRegressor(nn.Module):
    """
    Standard PyTorch LSTM implementation.
    Hidden class used internally by LSTMModels.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(_LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)

        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

########################################################################################################################
#                                                                  
# MODEL WRAPPER
#
########################################################################################################################
class LSTMModels:
    def __init__(self,
                random_state: int = 42,
                n_trials: int = 10,
                batch: int = 128,
                mode: str = 'CPU',
                **kwargs
                ):
        """
        Initialize the model wrapper.
        """
        self.model_name = 'lstm'
        self.random_state = random_state
        self.n_trials = n_trials
        self.batch_size = batch
        self.sequence_length = None
        self.mode = mode
        self.log_mlflow = kwargs.get('log_mlflow', True)  # Default to True for backward compatibility;
        
        # Determine device;
        self.device = torch.device('cuda' if self.mode in ['GPU', 'CUDA'] and torch.cuda.is_available() else 'cpu')
        
        # Placeholders;
        self.model = None
        self.X_train_shape = None

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            optimize_hyperparameters: bool = True,
            early_stopping: int = 10, # Epochs for patience
            epochs: int = 100,
            feature_pipeline=None,
            X_raw: Optional[pd.DataFrame] = None,
            cv_n_splits: int = 3,
            cv_gap: int = 24
            ):
        """
        Fits the LSTM model. Compatible with sklearn Pipeline.
        """
        # Validation Logic;
        if X is None or y is None:
            raise ValueError("Input data (X and y) cannot be None.")
        
        # Check for empty inputs (handling DataFrames or Numpy arrays);
        if hasattr(X, 'empty') and X.empty: raise ValueError("X cannot be empty.")
        if hasattr(y, 'empty') and y.empty: raise ValueError("y cannot be empty.")

        # Force keeping DataFrames to perform column filtering later if needed
        self.X = X 
        self.y = y
        self.X_train_shape = self.X.shape
        self.feature_pipeline = feature_pipeline
        self.X_raw = X_raw
        self.cv_n_splits = cv_n_splits
        self.cv_gap = cv_gap
        
        # Hyperparameter Optmization;
        best_params = {}
        if optimize_hyperparameters:
            print("[model_lstm.py] Running Bayesian Optimization for LSTM...")
            best_params = self._get_best_params()
        else:
            print("[model_lstm.py] Loading best parameters from MLflow...")
            mlflow_handler = MLFlowHandler()
            best_params = mlflow_handler.load_best_params(metric_name="lstm_best_rmse", mode="min")
            if not best_params:
                print("[model_lstm.py] No best params found, using defaults.")
                best_params = {
                    "hidden_size": 64, "num_layers": 1, 
                    "dropout": 0.0, "learning_rate": 0.001,
                    "sequence_length": 24  # Default for hydrological data (24 hours);
                }

        # Clean params types;
        for k, v in best_params.items():
            if k in ['hidden_size', 'num_layers', 'sequence_length']: best_params[k] = int(v)
        
        # Set sequence_length from optimized/loaded params;
        self.sequence_length = best_params.get('sequence_length', 24)

        print(f"[model_lstm.py] Training LSTM with params: {best_params}")

        # Convert inputs to float32 numpy arrays to ensure TensorDataset compatibility
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.select_dtypes(include=[np.number]).values
        
        if hasattr(self.X, 'astype'):
            self.X = self.X.astype(np.float32)
        else:
            self.X = np.array(self.X, dtype=np.float32)
            
        if hasattr(self.y, 'astype'):
            self.y = self.y.astype(np.float32)
        else:
            self.y = np.array(self.y, dtype=np.float32)

        # Data Preparation - Reshape 2D into 3D Sequences;
        X_seq, y_seq = self._create_sequences(self.X, self.y)
        
        # Create DataLoader;
        train_dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        # Prepare Validation Data if present;
        val_loader = None
        if X_val is not None and y_val is not None:
            # Pre-clean validation data: drop non-numeric columns
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.select_dtypes(include=[np.number])

            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            # Convert Validation to float32
            if hasattr(X_val_np, 'astype'):
                X_val_np = X_val_np.astype(np.float32)
            else:
                X_val_np = np.array(X_val_np, dtype=np.float32)
                
            if hasattr(y_val_np, 'astype'):
                y_val_np = y_val_np.astype(np.float32)
            else:
                y_val_np = np.array(y_val_np, dtype=np.float32)

            X_val_seq, y_val_seq = self._create_sequences(X_val_np, y_val_np)
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                print("[model_lstm.py] Using validation set for early stopping.")
        
        # Initialize Inner Model;
        self.model = _LSTMRegressor(
            input_size=self.X.shape[1],
            hidden_size=best_params.get('hidden_size', 64),
            num_layers=best_params.get('num_layers', 1),
            output_size=1,
            dropout=best_params.get('dropout', 0.0)
        ).to(self.device)

        # Training Loop with Early Stopping;
        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_params.get('learning_rate', 0.001))
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation / Early Stopping;
            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X, val_y = val_X.to(self.device), val_y.to(self.device).unsqueeze(1)
                        outputs = self.model(val_X)
                        val_loss += criterion(outputs, val_y).item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"[model_lstm.py][LSTM][Epoch {epoch+1}/{epochs}] train_loss={train_loss/len(train_loader):.4f}; val_loss={avg_val_loss:.4f};")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Ideally save model state dict here
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        print(f"[model_lstm.py] Early stopping triggered at epoch {epoch}")
                        break
                self.model.train() # Switch back to train mode
            else:
                print(f"[model_lstm.py][LSTM][Epoch {epoch+1}/{epochs}] train_loss={train_loss/len(train_loader):.4f}; (no val loader)")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        if X_test is None:
            raise ValueError("X_test cannot be None.")

        # Pre-clean: drop non-numeric columns
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.select_dtypes(include=[np.number])

        # Convert to numpy
        X_np = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Ensure float32
        if hasattr(X_np, 'astype'):
            X_np = X_np.astype(np.float32)
        else:
            X_np = np.array(X_np, dtype=np.float32)

        # CRITICAL: LSTM needs sequences. 
        # If X_test is just a 2D chunk, we treat it as the raw data to be sequenced.
        # NOTE: This simple sequence generation loses the first 'sequence_length' predictions
        # because we don't have history for them in X_test alone.
        X_seq, _ = self._create_sequences(X_np, None)

        if len(X_seq) == 0:
            # Fallback for very small test sets: duplicate last known row to force a prediction?
            # Or raise error. For strict comparability, we proceed with what we have.
            return np.array([])

        self.model.eval()
        predictions = []
        
        # Batch processing for prediction;
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                out = self.model(batch_X)
                predictions.append(out.cpu().numpy())

        # Flatten results;
        self.y_pred = np.concatenate(predictions).flatten()
        
        # Pad the beginning with NaNs or first prediction to match X_test length (Since creating sequences consumes the first L rows);
        padding = np.full(self.sequence_length, self.y_pred[0]) 
        self.y_pred = np.concatenate([padding, self.y_pred])
        
        # Truncate if padding made it too long (rare) or fit to exactly X_test length;
        return self.y_pred[:len(X_test)]

    def metric(self, y_true: pd.Series, y_pred: Optional[pd.Series] = None):
        """
        Identical metric method to GBM models.
        """
        if y_pred is None:
            y_pred = self.y_pred

        # Ensure lengths match (handle the sequence shortening);
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

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

    def _create_sequences(self, data, target=None):
        """
        Converts 2D array (N, F) into 3D array (N, Seq_Len, F).
        """
        xs = []
        ys = []
        length = len(data)
        
        if length <= self.sequence_length:
            return np.array([]), np.array([])

        for i in range(length - self.sequence_length):
            x_chunk = data[i : i + self.sequence_length]
            xs.append(x_chunk)
            
            if target is not None:
                ys.append(target[i + self.sequence_length])
        
        xs = np.array(xs)
        if target is not None:
            ys = np.array(ys)
            return xs, ys
        return xs, None

    def _get_best_params(self) -> dict:
        """
        Performs Bayesian optimization to find the best hyperparameters.
        
        Parameters:
            - X_train: Training features
            - y_train: Training target
            
        Returns:
            - dict: Best hyperparameters
        """
        # Import here to avoid circular import;
        from source_backend.optimize_params import BayesianOptimization
        
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
            log_mlflow=self.log_mlflow
        )
        return optimizer.optimize()