"""
Main backend orchestrator for model training and evaluation;

Reads clean data from database, performs train/test split, preprocessing,
model training, and evaluation;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import argparse
import time
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Optional
from db_handler import DBConnection
from src.pipe_preparation import data_division, encoding_pipeline
from src.train_models import training_pipeline
from src.mlflow_utils import MLFlowHandler
from src.data_io import save_to_database
from src.main_data import main_database
from src.data_imputation import feature_imputation
from util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# MAIN BACKEND PIPELINE
#
########################################################################################################################
def main_backend(
    target_column: str,
    models_to_use: Optional[list] = None,
    train_size: float = 0.7,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    n_trials: int = 10,
    batch: int = 128,
    steps: int = 12,
    freq: str = 'h',
    mode: str = 'CPU',
    optimize: bool = True,
    early_stopping: int = 50,
    save_to_db: bool = False,
    use_lags: bool = False,
    use_rolling_stats: bool = False,
    use_cumulative: bool = False,
    use_feature_selection: bool = False,
    n_features: Optional[int] = None,
    log_mlflow: bool = True
) -> None:
    """
    Execute complete model training pipeline: data loading, splitting, preprocessing, and training;
    
    Reads clean data from database (produced by src ETL), performs train/test split,
    applies feature encoding and preprocessing, trains the selected model, and evaluates performance;
    Optionally saves predictions and metrics to database;
    
    Parameters:
        target_column (str): Name of target column to predict;
        models_to_use (Optional[list]): List of models to train ('SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM', 'DUMMY').
            If None, trains all models (default: None);
        train_size (float): Proportion of data for training set (default: 0.7);
        test_size (float): Proportion of data for test set (default: 0.2);
        val_size (float): Proportion of data for validation set (default: 0.1);
        random_state (int): Random seed for reproducibility (default: 42);
        n_trials (int): Number of trials for hyperparameter optimization (default: 10);
        batch (int): Training batch size (default: 128);
        steps (int): Prediction horizon in time steps (default: 12);
        freq (str): Frequency of predictions (pandas offset) (default: 'h');
        mode (str): Training device mode ('CPU', 'GPU', 'CUDA') (default: 'CPU');
        optimize (bool): Whether to perform hyperparameter optimization (default: True);
        early_stopping (int): Number of rounds for early stopping (default: 50);
        save_to_db (bool): If True, save predictions and metrics to database (default: False);
        use_lags (bool): If True, add lag features transformer to pipeline (default: False).
        use_rolling_stats (bool): If True, add rolling statistics transformer to pipeline (default: False).
        use_cumulative (bool): If True, add cumulative rolling-sum transformer to pipeline (default: False).
        use_feature_selection (bool): If True, add feature selection based on RandomForest importance (default: False);
        n_features (Optional[int]): Number of top features to select if use_feature_selection=True.
            If None, uses default (50) (default: None);
        log_mlflow (bool): If True, log experiments to MLFlow (default: True);
    
    Returns:
        None: Function performs training and optionally persists results to database;
    """
    # Initialize MLFlow Handler and log initial parameters (only if logging is enabled);
    mlflow_handler = None
    if log_mlflow:
        mlflow_handler = MLFlowHandler(experiment_name="river_level_forecasting")
        mlflow_handler.start_run(run_name="initial_config")
        
        # Prepare parameters for logging (convert list to string if needed);
        log_params = {
            "target_column": target_column,
            "models_to_use": str(models_to_use) if models_to_use is not None else "all",
            "train_size": train_size,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "n_trials": n_trials,
            "batch": batch,
            "steps": steps,
            "freq": freq,
            "mode": mode,
            "optimize": optimize,
            "early_stopping": early_stopping,
            "save_to_db": save_to_db,
            "use_lags": use_lags,
            "use_rolling_stats": use_rolling_stats,
            "use_cumulative": use_cumulative,
            "use_feature_selection": use_feature_selection,
            "n_features": n_features if n_features is not None else "None"
        }
        mlflow_handler.log_params(log_params)
        mlflow_handler.end_run()
    
    # Execute data pipeline and read clean data;
    logger.info("Executing ETL pipeline...")
    df = main_database(save_to_db=save_to_db)
    logger.info("Loaded %s rows from data pipeline", len(df))

    # Enforce chronological order to avoid temporal leakage before splitting;
    if "Data_Hora_Medicao" in df.columns:
        df = df.sort_values("Data_Hora_Medicao").reset_index(drop=True)
    else:
        logger.warning("Column 'Data_Hora_Medicao' not found; skipping chronological sort;");

    # Split data into train/test sets;
    logger.info("Splitting data into train/test sets...")
    if val_size is not None and val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = data_division(
            df=df,
            target_column=target_column,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state)
        logger.info("Train set: %s | Validation set: %s | Test set: %s", X_train.shape, X_val.shape, X_test.shape)

        logger.info("Applying feature imputation post-split...")
        X_train = feature_imputation(X_train)
        X_val = feature_imputation(X_val)
        X_test = feature_imputation(X_test)

    else:
        X_train, X_test, y_train, y_test = data_division(
            df=df,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state)
        logger.info("Train set: %s | Test set: %s", X_train.shape, X_test.shape)
        
        logger.info("Applying feature imputation post-split...")
        X_train = feature_imputation(X_train)
        X_test = feature_imputation(X_test)
    
    # Create preprocessing pipeline;
    logger.info("Creating preprocessing pipeline...")
    preprocessor = encoding_pipeline(target_column=target_column)
    
    # Pre-fit the preprocessor so Validation data can be processed;
    preprocessor.fit(X_train, y_train)

    # Create full training pipeline (preprocessing + models);
    logger.info("Building training pipeline...")
    pipelines = training_pipeline(
        preprocessor=preprocessor,
        models_to_use=models_to_use,
        random_state=random_state,
        n_trials=n_trials,
        batch=batch,
        steps=steps,
        freq=freq,
        mode=mode,
        use_lags=use_lags,
        use_rolling_stats=use_rolling_stats,
        use_cumulative=use_cumulative,
        use_feature_selection=use_feature_selection,
        n_features=n_features
    ) 
    
    # Train each pipeline independently;
    logger.info("Training model...")
    results = {}
    all_predictions = []
    all_metrics = []
    all_features = []
    
    for name, pipeline in pipelines.items():
        logger.info("Training %s...", name)
        model_start_time = time.time()  # Track total runtime per model;
        
        # Start MLFlow run for this model (only if logging is enabled);
        if log_mlflow:
            mlflow_handler.start_run(run_name=f"train_{name}")
            
            # Log general parameters;
            mlflow_handler.log_params({
                "target_column": target_column,
                "train_size": train_size,
                "test_size": test_size,
                "val_size": val_size,
                "random_state": random_state,
                "model_type": name
            })
        
        # Build feature pipeline reference (preprocessor + optional steps) for CV leakage-free folds;
        feature_steps = pipeline.steps[:-1]
        feature_pipeline = Pipeline(feature_steps)

        # Fit validation data for early stopping if available (only for XGBOOST and LIGHTGBM);
        fit_params = {f"{name}__optimize_hyperparameters": optimize}
        if name in ['XGBOOST', 'LIGHTGBM', 'LSTM']:
            cv_folds = 5 if name != 'LSTM' else 3
            fit_params.update({
                f"{name}__feature_pipeline": feature_pipeline,
                f"{name}__X_raw": X_train,
                f"{name}__cv_n_splits": cv_folds,
                f"{name}__cv_gap": 24,  # One-day gap on hourly data to reduce leakage;
            })
        
        if val_size is not None and val_size > 0 and name in ['XGBOOST', 'LIGHTGBM']:
            logger.info("Preparing validation data for %s early stopping...", name)
            
            # Split pipeline into feature engineering and model steps;
            model_step_name, model_instance = pipeline.steps[-1]
            
            # Create and fit feature pipeline on training data;
            # This ensures all transformations (preprocessor, lags, selection) are learned from training data;
            feature_pipeline.fit(X_train, y_train)
            
            # Transform both train and validation sets using the fitted feature pipeline;
            X_train_processed = feature_pipeline.transform(X_train)
            X_val_processed = feature_pipeline.transform(X_val)

            # Prepare parameters for manual model fitting (remove pipeline prefixes);
            model_params = {
                "optimize_hyperparameters": optimize,
                "X_val": X_val_processed,
                "y_val": y_val,
                "early_stopping": early_stopping,
                "feature_pipeline": feature_pipeline,
                "X_raw": X_train,
                "cv_n_splits": 5,
                "cv_gap": 24,
            }
            
            # Fit the model instance directly with processed data -> This avoids "double fitting" the transformers which would happen if we called pipeline.fit();
            model_instance.fit(X_train_processed, y_train, **model_params)
            
            # Reconstruct the pipeline with the fitted steps for future use (prediction, logging);
            pipeline = Pipeline(feature_pipeline.steps + [(model_step_name, model_instance)])
        
        else:
            # Standard Pipeline fit for other models or when no validation set is used;
            pipeline.fit(X_train, y_train, **fit_params)
        
        # Collect transformed feature matrices (post-preprocessor + feature engineering + selection);
        if save_to_db:
            def collect_features(split_name, X_split, y_split):
                orig_index = X_split.index if hasattr(X_split, "index") else None
                features = X_split
                for step_name, step in pipeline.steps[:-1]:
                    features = step.transform(features)
                if not isinstance(features, pd.DataFrame):
                    features = pd.DataFrame(features, index=orig_index)
                features = features.copy()
                target_series = y_split.reindex(features.index) if isinstance(y_split, pd.Series) else pd.Series(y_split, index=features.index, name='target')
                features['target'] = target_series
                features['model_name'] = name
                features['split'] = split_name
                return features

            all_features.append(collect_features('train', X_train, y_train))
            if val_size is not None and val_size > 0:
                all_features.append(collect_features('val', X_val, y_val))
            all_features.append(collect_features('test', X_test, y_test))
        
        # Predict with fitted Pipeline;
        y_pred = pipeline.predict(X_test)

        # Capture original datetime before any pivoting to keep human-readable dates;
        date_col = None
        if hasattr(X_test, "columns") and "Data_Hora_Medicao" in X_test.columns:
            date_col = X_test["Data_Hora_Medicao"].copy()
        elif hasattr(X_test, "index"):
            date_col = X_test.index

        results[name] = {
            'pipeline': pipeline,
            'predictions': y_pred
        }

        # Evaluate model;
        model_step = pipeline.named_steps[name] 
        metrics = model_step.metric(y_true=y_test, y_pred=y_pred)
        if log_mlflow:
            mlflow_handler.log_metrics(metrics)
        logger.info("Model performance for %s: %s", name, metrics)

        # Log the model with input example (only if logging is enabled);
        if log_mlflow:
            mlflow_handler.log_model(pipeline, artifact_path=f"model_{name}", input_example=(X_test.iloc[:1] if hasattr(X_test, 'iloc') else X_test[:1]))
        
        # Store predictions and metrics for database saving;
        # Create predictions dataframe for this model;
        pred_df = pd.DataFrame({
            'model_name': name,
            'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
            'y_pred': y_pred,
            'target_column': target_column
        })
        # Add index if available from X_test; align datetime column when present;
        if hasattr(X_test, 'index'):
            pred_df.index = X_test.index
        if date_col is not None:
            pred_df['date'] = date_col
        all_predictions.append(pred_df)
        
        # Store metrics for this model;
        model_duration_seconds = time.time() - model_start_time  # End-to-end duration for this model;
        metrics_row = {
            'model_name': name,
            'target_column': target_column,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'nse': metrics['nse'],
            'r2': metrics['r2'],
            'kge': metrics['kge'],
            'train_size': train_size,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state,
            'duration_seconds': model_duration_seconds
        }
        all_metrics.append(metrics_row)
        
        # End MLFlow run (only if logging is enabled);
        if log_mlflow:
            mlflow_handler.end_run()
    
    # Create combined results dataframe with predictions and metrics;
    if all_predictions:
        # Concatenate all predictions;
        full_preds = pd.concat(all_predictions, ignore_index=False)
        
        # Pivot to have columns per model, preferring real datetime over integer index;
        if 'date' in full_preds.columns:
            df_predictions = full_preds.pivot(index='date', columns='model_name', values='y_pred')
            y_true = full_preds.groupby('date')['y_true'].first()
            df_predictions['y_true'] = y_true
            df_predictions = df_predictions.reset_index()
        else:
            df_predictions = full_preds.pivot_table(index=full_preds.index, columns='model_name', values='y_pred')
            y_true = full_preds.groupby(level=0)['y_true'].first()
            df_predictions['y_true'] = y_true
            df_predictions = df_predictions.reset_index().rename(columns={'index': 'date'})
        
        # Metrics DataFrame;
        df_metrics = pd.DataFrame(all_metrics)
        
        # Save ML results to database if flag is set;
        if save_to_db:
            logger.info("Saving ML results to database...")
            save_to_database(
                df_predictions=df_predictions,
                df_metrics=df_metrics,
                df_features=pd.concat(all_features, ignore_index=True) if all_features else None
            )
    logger.info("Backend pipeline completed!")

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main backend training pipeline")
    
    # Main backend parameters;
    parser.add_argument('--target_column', type=str, default='Cota_Adotada_87450004', help='Name of target column to predict')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for training set (0.0 to 1.0)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set (0.0 to 1.0)')
    parser.add_argument('--val_size', type=float, default=0.0, help='Proportion of data for validation set (0.0 to 1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['CPU', 'GPU', 'CUDA'], default='GPU', help='Training device mode: CPU (default), GPU (OpenCL), or CUDA')
    parser.add_argument('--models_to_use', type=str, nargs='+',
                        choices=['SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM', 'DUMMY'],
                        default=['LSTM', 'XGBOOST', 'LIGHTGBM', 'DUMMY'],
                        help='List of models to train (e.g., --models_to_use XGBOOST LIGHTGBM). If None, trains all models'
                        )
    
    # Additional pipeline parameters;
    parser.add_argument('--batch', type=int, default=128, help='Training batch size')
    parser.add_argument('--steps', type=int, default=12, help='The amount of forward steps to be predicted')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials for hyperparameter optimization') # Testing=10, Initial=100, Deep=500;
    parser.add_argument('--early_stopping', type=int, default=50, help='Number of rounds for early stopping (default: 50)')
    parser.add_argument('--n_features', type=int, default=100, help='Number of top features to select if use_feature_selection=True (default: 50)')
    parser.add_argument('--freq', type=str, 
                        choices=['h', 'bh', 'min', 's', 'D', 'B', 'W', 'M', 'MS', 'SMS'], 
                        default='h', 
                        help='Frequency of predictions (pandas offset)'
                        )
    
    # Bool arguments;
    parser.add_argument('--save_to_db', action='store_true', help='Save the results to the database')
    parser.add_argument('--no_save_to_db', dest='save_to_db', action='store_false', help='Do not save the results to the database')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter Optimization')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false', help='Do not perform hyperparameter Optimization')
    parser.add_argument('--use_lags', action='store_true', help='Add lag features transformer to pipeline')
    parser.add_argument('--no_use_lags', dest='use_lags', action='store_false', help='Do not add lag features transformer')
    parser.add_argument('--use_rolling_stats', action='store_true', help='Add rolling statistics transformer to pipeline')
    parser.add_argument('--no_use_rolling_stats', dest='use_rolling_stats', action='store_false', help='Do not add rolling statistics transformer')
    parser.add_argument('--use_cumulative', action='store_true', help='Add cumulative rolling-sum transformer to pipeline')
    parser.add_argument('--no_use_cumulative', dest='use_cumulative', action='store_false', help='Do not add cumulative rolling-sum transformer')
    parser.add_argument('--use_feature_selection', action='store_true', help='Add feature selection based on RandomForest importance')
    parser.add_argument('--no_use_feature_selection', dest='use_feature_selection', action='store_false', help='Do not add feature selection')
    parser.add_argument('--log_mlflow', action='store_true', help='Log experiments to MLFlow')
    parser.add_argument('--no_log_mlflow', dest='log_mlflow', action='store_false', help='Do not log experiments to MLFlow')

    # Set default values for booleans;
    parser.set_defaults(
        save_to_db=True,
        optimize=True,
        use_lags=True,
        use_rolling_stats=True,
        use_cumulative=True,
        use_feature_selection=True,
        log_mlflow=True
    )
    
    args = parser.parse_args()
    logger.info("Arguments: %s", args)

    # Execute main backend pipeline with parsed arguments;
    main_backend(
        target_column=args.target_column,
        models_to_use=args.models_to_use,
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_trials=args.trials,
        batch=args.batch,
        steps=args.steps,
        freq=args.freq,
        mode=args.mode,
        optimize=args.optimize,
        early_stopping=args.early_stopping,
        save_to_db=args.save_to_db,
        use_lags=args.use_lags,
        use_rolling_stats=args.use_rolling_stats,
        use_cumulative=args.use_cumulative,
        use_feature_selection=args.use_feature_selection,
        n_features=args.n_features,
        log_mlflow=args.log_mlflow
        )
    logger.info("All Done!")