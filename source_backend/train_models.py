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
def create_model_pipeline(model, preprocessor, model_name):
    """Create a pipeline for a single model"""
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        (model_name, model)
    ])
    return pipe


def training_pipeline(
    models_to_use: Optional[list] = None,
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
    n_trials: int = 10,
    batch: int = 128,
    steps: int = 12,
    freq: str = 'h',
    mode: str = 'CPU',
    **kwargs
) -> None:
    """
    Construct sklearn Pipeline combining preprocessing and model training;
    
    Creates a pipeline that chains the preprocessing step (feature encoding, scaling)
    with the selected model. The preprocessor is applied first, then the model is trained;
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline from data_preparation;
        models_to_use (Optional[list]): List of model names to use. Options: 'SARIMA', 'LSTM', 
            'XGBOOST', 'LIGHTGBM'. If None, trains all models (default: None);
        random_state (int): Random seed for reproducibility (default: 42);
        n_trials (int): Number of trials for hyperparameter optimization (default: 10);
        batch (int): Training batch size (default: 128);
        steps (int): The amount of forward steps to be predicted (default: 12);
        freq (str): Frequency of predictions (pandas offset) (default: 'h');
        **kwargs: Additional parameters to pass to models;
    
    Returns:
        Pipeline: sklearn Pipeline with preprocessing and model steps;
    """
    # Create separate pipeline for each model with arguments;
    pipelines = {
        'SARIMA': create_model_pipeline(
            SARIMAModels(random_state=random_state, n_trials=n_trials, batch=batch, steps=steps, mode=mode, **kwargs), 
            preprocessor,
            model_name='SARIMA'
        ),
        'LSTM': create_model_pipeline(
            LSTMModels(random_state=random_state, n_trials=n_trials, batch=batch, steps=steps, mode=mode, **kwargs), 
            preprocessor,
            model_name='LSTM'
        ),
        'XGBOOST': create_model_pipeline(
            XGBoostModels(random_state=random_state, n_trials=n_trials, batch=batch, steps=steps, mode=mode, **kwargs), 
            preprocessor,
            model_name='XGBOOST'
        ),
        'LIGHTGBM': create_model_pipeline(
            LightGBMModels(random_state=random_state, n_trials=n_trials, batch=batch, steps=steps, mode=mode, **kwargs), 
            preprocessor,
            model_name='LIGHTGBM'
        )
    }

    # Remove models that are NOT in the models_to_use list;
    if models_to_use is not None:
        models_to_remove = [model for model in pipelines.keys() if model not in models_to_use]
        for model in models_to_remove:
            pipelines.pop(model)
    return pipelines