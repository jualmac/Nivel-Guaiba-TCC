"""
Model training pipeline orchestrator;

Constructs sklearn pipelines that combine preprocessing and model training steps;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import joblib
import pandas as pd
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models;
from source_backend.model_sarima import SARIMAModels
from source_backend.model_lstm import LSTMModels
from source_backend.model_xgboost import XGBoostModels
from source_backend.model_lightgbm import LightGBMModels

########################################################################################################################
#                                                                  
# TRAINING PIPELINE
#
########################################################################################################################
#TODO: Add rolling lags -> LagFeaturesTransformer or RollingStatsTransformer;
#TODO: Add Feature Selection -> SelectKBest (if needed);
def create_model_pipeline(model, preprocessor):
    """Create a pipeline for a single model"""
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipe

def training_pipeline(
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_train: pd.Series = None,
    y_test: pd.Series = None,
    models_to_use: Optional[list] = None,
    preprocessor: ColumnTransformer = None,
) -> None:
    """
    Construct sklearn Pipeline combining preprocessing and model training;
    
    Creates a pipeline that chains the preprocessing step (feature encoding, scaling)
    with the selected model. The preprocessor is applied first, then the model is trained;
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline from data_preparation;
        models_to_use (Optional[list]): List of model names to use. Options: 'SARIMA', 'LSTM', 
            'XGBOOST', 'LIGHTGBM'. If None, trains all models (default: None);
    
    Returns:
        Pipeline: sklearn Pipeline with preprocessing and model steps;
    """
    # Validate Dataset;
    if (
        X_train is None or
        y_train is None or
        X_test is None or
        y_test is None or
        (hasattr(X_train, 'empty') and X_train.empty) or
        (hasattr(y_train, 'empty') and y_train.empty) or
        (hasattr(X_test, 'empty') and X_test.empty) or
        (hasattr(y_test, 'empty') and y_test.empty) or
        len(y_train) == 0 or
        len(y_test) == 0
        ):
        raise ValueError("Training and Test data are required and must not be empty")

    # Create separate pipeline for each model;
    pipelines = {
        'SARIMA': create_model_pipeline(SARIMAModels(), preprocessor),
        'LSTM': create_model_pipeline(LSTMModels(), preprocessor),
        'XGBOOST': create_model_pipeline(XGBoostModels(), preprocessor),
        'LIGHTGBM': create_model_pipeline(LightGBMModels(), preprocessor)
    }

    # Remove models that are NOT in the models_to_use list;
    if models_to_use is not None:
        models_to_remove = [model for model in pipelines.keys() if model not in models_to_use]
        for model in models_to_remove:
            pipelines.pop(model)

    # Train each pipeline independently;
    results = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        results[name] = {
            'pipeline': pipeline,
            'predictions': pipeline.predict(X_test)
        }