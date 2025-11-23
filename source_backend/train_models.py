"""
Model training pipeline orchestrator;

Constructs sklearn pipelines that combine preprocessing and model training steps;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models;
#TODO: Import actual models from source_backend modules;
# from source_backend.model_sarima import SARIMA_Model
# from source_backend.model_lstm import LSTM_Model
# from source_backend.model_xgboost import XGBoostModel
# from source_backend.model_lightgbm import LightGBMModel

########################################################################################################################
#                                                                  
# TRAINING PIPELINE
#
########################################################################################################################
def training_pipeline(
    preprocessor: ColumnTransformer,
    model_name: Optional[str] = None
) -> Pipeline:
    """
    Construct sklearn Pipeline combining preprocessing and model training;
    
    Creates a pipeline that chains the preprocessing step (feature encoding, scaling)
    with the selected model. The preprocessor is applied first, then the model is trained;
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline from data_preparation;
        model_name (Optional[str]): Name of model to use. Options: 'SARIMA', 'LSTM', 
            'XGBOOST', 'LIGHTGBM'. If None, uses default (default: None);
    
    Returns:
        Pipeline: sklearn Pipeline with preprocessing and model steps;
    """
    #TODO: Implement model selection logic;
    #TODO: Add models to pipeline based on model_name parameter;
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # ("SARIMA", SARIMAModels), #TODO: Define models here;
        # ("LSTM", LSTMModels), #TODO: Define models here;
        # ("XGBOOST", XGBoostModels), #TODO: Define models here;
        # ("LIGHTGBM", LightGBMModels), #TODO: Define models here;
    ])
    return pipeline