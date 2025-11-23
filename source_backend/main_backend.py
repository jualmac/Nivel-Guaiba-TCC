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

########################################################################################################################
#                                                                  
# MAIN BACKEND PIPELINE
#
########################################################################################################################
def main_backend(
    target_column: str,
    model_name: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> None:
    """
    Execute complete model training pipeline: data loading, splitting, preprocessing, and training;
    
    Reads clean data from database (produced by source_database ETL), performs train/test split,
    applies feature encoding and preprocessing, trains the selected model, and evaluates performance;
    
    Parameters:
        target_column (str): Name of target column to predict;
        model_name (Optional[str]): Model to train ('SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM').
            If None, trains all models (default: None);
        test_size (float): Proportion of data for test set (default: 0.2);
        random_state (int): Random seed for reproducibility (default: 42);
    
    Returns:
        None: Function performs training and persists results;
    """
    # Read clean data from database;
    print("Loading data from database...")
    db = DBConnection()
    df = db.run("SELECT * FROM data_stations_melted")['result']
    print(f"Loaded {len(df)} rows from database")
    
    # Split data into train/test sets;
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = data_division(
        df=df,
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state)
    
    print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    
    # Create preprocessing pipeline;
    #TODO: Define actual column names based on data structure;
    print("Creating preprocessing pipeline...")
    preprocessor = encoding_pipeline()
    
    # Create full training pipeline (preprocessing + model);
    print("Building training pipeline...")
    pipeline = training_pipeline(
        preprocessor=preprocessor,
        model_name=model_name
    )
    
    # Train the model;
    #TODO: Implement actual training logic;
    print("Training model...")
    # pipeline.fit(X_train, y_train)
    
    # Evaluate model;
    #TODO: Implement evaluation and metrics calculation;
    print("Evaluating model...")
    # score = pipeline.score(X_test, y_test)
    # print(f"Model performance: {score}")
    
    # Save model and results;
    #TODO: Implement model persistence (MLFlow, pickle, etc);
    print("Saving model...")
    
    print("Backend pipeline completed!")
    return None

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main backend training pipeline")
    
    # Main backend parameters;
    parser.add_argument('--target_column', type=str, default='value', help='Name of target column to predict')
    parser.add_argument('--model_name', type=str, choices=['SARIMA', 'LSTM', 'XGBOOST', 'LIGHTGBM'], default=None, help='Model to train. If None, trains all models')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set (0.0 to 1.0)')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data for validation set (0.0 to 1.0)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    # Additional pipeline parameters;
    parser.add_argument('--batch', type=int, default=128, help='Training batch size')
    parser.add_argument('--steps', type=int, default=12, help='The amount of forward steps to be predicted')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--freq', type=str, choices=['h', 'bh', 'min', 's', 'D', 'B', 'W', 'M', 'MS', 'SMS'], default='MS', help='Frequency of predictions (pandas offset)')
    parser.add_argument('--mode', type=str, choices=['CPU', 'GPU', 'CUDA'], default='CPU', help='Training device mode: CPU (default), GPU (OpenCL), or CUDA')
    
    # Bool arguments;
    parser.add_argument('--save_to_db', action='store_true', help='Save the results to the database')
    parser.add_argument('--no_save_to_db', dest='save_to_db', action='store_false', help='Do not save the results to the database')
    
    # Set default values for booleans;
    parser.set_defaults(
        save_to_db=True,
    )
    
    args = parser.parse_args()
    print(f'Arguments: {args}')

    # Execute main backend pipeline with parsed arguments;
    main_backend(
        target_column=args.target_column,
        model_name=args.model_name,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
        )
    print('All Done!')