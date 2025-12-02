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
import pandas as pd
from typing import Optional
from db_handler import DBConnection
from source_backend.data_preparation import data_division, encoding_pipeline
from source_backend.train_models import training_pipeline
from source_backend.mlflow_utils import MLFlowHandler
from source_database.data_io import save_to_database

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
    save_to_db: bool = False
) -> None:
    """
    Execute complete model training pipeline: data loading, splitting, preprocessing, and training;
    
    Reads clean data from database (produced by source_database ETL), performs train/test split,
    applies feature encoding and preprocessing, trains the selected model, and evaluates performance;
    Optionally saves predictions and metrics to database;
    
    Parameters:
        target_column (str): Name of target column to predict;
        models_to_use (Optional[list]): List of models to train ('SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM').
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
    
    Returns:
        None: Function performs training and optionally persists results to database;
    """
    # Initialize MLFlow Handler and log initial parameters;
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
        "early_stopping": early_stopping
    }
    mlflow_handler.log_params(log_params)
    mlflow_handler.end_run()
    
    # Read clean data from database;
    print("Loading data from database...")
    db = DBConnection()
    df = db.run("SELECT * FROM data_stations")['result']
    print(f"Loaded {len(df)} rows from database")

    # Split data into train/test sets;
    print("Splitting data into train/test sets...")
    if val_size is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = data_division(
            df=df,
            target_column=target_column,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state)
        print(f"Train set: {(X_train.shape)}\nValidation set: {(X_val.shape)}\nTest set: {(X_test.shape)}")

    else:
        X_train, X_test, y_train, y_test = data_division(
            df=df,
            target_column=target_column,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state)
        print(f"Train set: {(X_train.shape)}\nTest set: {(X_test.shape)}")
    
    # Create preprocessing pipeline;
    print("Creating preprocessing pipeline...")
    preprocessor = encoding_pipeline(target_column=target_column)
    
    # Pre-fit the preprocessor so Validation data can be processed;
    preprocessor.fit(X_train, y_train)

    # Create full training pipeline (preprocessing + models);
    print("Building training pipeline...")
    pipelines = training_pipeline(
        preprocessor=preprocessor,
        models_to_use=models_to_use,
        random_state=random_state,
        n_trials=n_trials,
        batch=batch,
        steps=steps,
        freq=freq,
        mode=mode
    ) 
    
    # Train each pipeline independently;
    print("Training model...")
    results = {}
    all_predictions = []
    all_metrics = []
    
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        
        # Start MLFlow run for this model;
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
        
        # Fit validation data for early stopping if available (only for XGBOOST and LIGHTGBM);
        fit_params = {f"{name}__optimize_hyperparameters": optimize}
        if val_size is not None and name in ['XGBOOST', 'LIGHTGBM']:
            # Retrieve the preprocessor step from the current pipeline and apply it manually to validation dataset;
            preprocessor_step = pipeline.named_steps['preprocessor']
            X_val_processed = preprocessor_step.transform(X_val)

            # Add Validation dataset and early stopping to fit;
            fit_params = {f"{name}__optimize_hyperparameters": optimize}
            fit_params[f"{name}__X_val"] = X_val_processed
            fit_params[f"{name}__y_val"] = y_val
            fit_params[f"{name}__early_stopping"] = early_stopping
        
        # Fit Pipeline;
        pipeline.fit(X_train, y_train, **fit_params)
        
        # Predict with fitted Pipeline;
        y_pred = pipeline.predict(X_test)

        results[name] = {
            'pipeline': pipeline,
            'predictions': y_pred
        }

        # Evaluate model;
        model_step = pipeline.named_steps[name] 
        metrics = model_step.metric(y_true=y_test, y_pred=y_pred)
        mlflow_handler.log_metrics(metrics)
        print(f"Model performance: {metrics}")

        # Log the model with input example;
        mlflow_handler.log_model(pipeline, artifact_path=f"model_{name}", input_example=(X_test.iloc[:1] if hasattr(X_test, 'iloc') else X_test[:1]))
        
        # Store predictions and metrics for database saving;
        # Create predictions dataframe for this model;
        pred_df = pd.DataFrame({
            'model_name': name,
            'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
            'y_pred': y_pred,
            'target_column': target_column
        })
        # Add index if available from X_test;
        if hasattr(X_test, 'index'):
            pred_df.index = X_test.index
        all_predictions.append(pred_df)
        
        # Store metrics for this model;
        metrics_row = {
            'model_name': name,
            'target_column': target_column,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'nse': metrics['nse'],
            'r2': metrics['r2'],
            'train_size': train_size,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': random_state
        }
        all_metrics.append(metrics_row)
        
        # End MLFlow run;
        mlflow_handler.end_run()
    
    # Create combined results dataframe with predictions and metrics;
    if all_predictions:
        df_predictions = pd.concat(all_predictions, ignore_index=False)
        df_metrics = pd.DataFrame(all_metrics)
        # Merge predictions with metrics for comprehensive results;
        df_ml_results = df_predictions.merge(df_metrics, on='model_name', how='left')
        
        # Save ML results to database if flag is set;
        if save_to_db:
            print("Saving ML results to database...")
            save_to_database(df_ml_results=df_ml_results)
    
    print("Backend pipeline completed!")

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main backend training pipeline")
    
    # Main backend parameters;
    parser.add_argument('--target_column', type=str, default='Cota_Adotada_87450004', help='Name of target column to predict')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion of data for training set (0.0 to 1.0)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set (0.0 to 1.0)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data for validation set (0.0 to 1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['CPU', 'GPU', 'CUDA'], default='CPU', help='Training device mode: CPU (default), GPU (OpenCL), or CUDA')
    parser.add_argument('--models_to_use', type=str, nargs='+',
                        choices=['SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM'],
                        default=['SARIMA'],
                        help='List of models to train (e.g., --models_to_use XGBOOST LIGHTGBM). If None, trains all models'
                        )
    
    # Additional pipeline parameters;
    parser.add_argument('--batch', type=int, default=128, help='Training batch size')
    parser.add_argument('--steps', type=int, default=12, help='The amount of forward steps to be predicted')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials for hyperparameter optimization') # Testing=10, Initial=100, Deep=500;
    parser.add_argument('--early_stopping', type=int, default=50, help='Number of rounds for early stopping (default: 50)')
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

    # Set default values for booleans;
    parser.set_defaults(
        save_to_db=True,
        optimize=True
    )
    
    args = parser.parse_args()
    print(f'Arguments: {args}')

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
        save_to_db=args.save_to_db
        )
    print('All Done!')