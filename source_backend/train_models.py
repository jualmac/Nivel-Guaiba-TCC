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
        (f"{model}", model)
    ])
    return pipe


def training_pipeline(
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
    return pipelines