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
from sklearn.pipeline import Pipeline
from typing import Optional
from db_handler import DBConnection
from source_backend.pipe_preparation import data_division, encoding_pipeline
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
    save_to_db: bool = False,
    use_lags: bool = False,
    use_feature_selection: bool = False,
    n_features: Optional[int] = None
) -> None:
    """
    Execute complete model training pipeline: data loading, splitting, preprocessing, and training;
    
    Reads clean data from database (produced by source_database ETL), performs train/test split,
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
            Warning: LagFeaturesTransformer requires 'value' column which may not exist after preprocessing;
        use_feature_selection (bool): If True, add feature selection based on RandomForest importance (default: False);
        n_features (Optional[int]): Number of top features to select if use_feature_selection=True.
            If None, uses default (50) (default: None);
    
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
        "early_stopping": early_stopping,
        "save_to_db": save_to_db,
        "use_lags": use_lags,
        "use_feature_selection": use_feature_selection,
        "n_features": n_features if n_features is not None else "None"
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
    if val_size is not None and val_size > 0:
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
        mode=mode,
        use_lags=use_lags,
        use_feature_selection=use_feature_selection,
        n_features=n_features
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
            print(f"Preparing validation data for {name} early stopping...")
            
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
            'kge': metrics['kge'],
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
        # Concatenate all predictions;
        full_preds = pd.concat(all_predictions, ignore_index=False)
        
        # Pivot to have columns per model (index is date);
        df_predictions = full_preds.pivot_table(index=full_preds.index, columns='model_name', values='y_pred')
        
        # Add y_true (should be same for all models for same index);
        # Group by index and take the first value of y_true;
        y_true = full_preds.groupby(level=0)['y_true'].first()
        df_predictions['y_true'] = y_true
        
        # Reset index to make date a column;
        df_predictions = df_predictions.reset_index().rename(columns={'index': 'date', 'Data_Hora_Medicao': 'date'})
        
        # Metrics DataFrame;
        df_metrics = pd.DataFrame(all_metrics)
        
        # Save ML results to database if flag is set;
        if save_to_db:
            print("Saving ML results to database...")
            save_to_database(
                df_predictions=df_predictions,
                df_metrics=df_metrics
            )
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
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for training set (0.0 to 1.0)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set (0.0 to 1.0)')
    parser.add_argument('--val_size', type=float, default=0.0, help='Proportion of data for validation set (0.0 to 1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['CPU', 'GPU', 'CUDA'], default='CPU', help='Training device mode: CPU (default), GPU (OpenCL), or CUDA')
    parser.add_argument('--models_to_use', type=str, nargs='+',
                        choices=['SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM', 'DUMMY'],
                        default=['DUMMY'],
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
    parser.add_argument('--use_lags', action='store_true', help='Add lag features transformer to pipeline (Warning: requires "value" column)')
    parser.add_argument('--no_use_lags', dest='use_lags', action='store_false', help='Do not add lag features transformer')
    parser.add_argument('--use_feature_selection', action='store_true', help='Add feature selection based on RandomForest importance')
    parser.add_argument('--no_use_feature_selection', dest='use_feature_selection', action='store_false', help='Do not add feature selection')   

    # Set default values for booleans;
    parser.set_defaults(
        save_to_db=True,
        optimize=True,
        use_lags=False,
        use_feature_selection=True
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
        save_to_db=args.save_to_db,
        use_lags=args.use_lags,
        use_feature_selection=args.use_feature_selection,
        n_features=args.n_features
        )
    print('All Done!')