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
    print(f"Loaded {len(df)} rows from database.")
    
    # Split data into train/test sets;
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = data_division(
        df=df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    
    # Create preprocessing pipeline;
    #TODO: Define actual column names based on data structure;
    print("Creating preprocessing pipeline...")
    preprocessor = encoding_pipeline(
        numerical_cols=[],  #TODO: Define numerical columns;
        categorical_cols=[],  #TODO: Define categorical columns;
        missing_indicator_cols=['value']
    )
    
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
    #TODO: Define actual target column name;
    main_backend(
        target_column='value',  #TODO: Update with correct target column;
        model_name=None,
        test_size=0.2,
        random_state=42
    )