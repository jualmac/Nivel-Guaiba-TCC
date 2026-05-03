"""
Model training pipeline orchestrator;

Constructs sklearn pipelines that combine preprocessing and model training steps;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models;
from src.models.model_sarima import SARIMAModels
from src.models.model_lstm import LSTMModels
from src.models.model_xgboost import XGBoostModels
from src.models.model_lightgbm import LightGBMModels
from src.models.model_dummy import DummyModels
from src.pipelines.pipeline_transformations import FeatureImportanceSelector, LagFeaturesTransformer, RollingStatsTransformer, CumulativeFeaturesTransformer

########################################################################################################################
#                                                                  
# TRAINING PIPELINE
#
########################################################################################################################
def create_model_pipeline(
    model, 
    preprocessor, 
    model_name, 
    use_lags: bool = False, 
    use_rolling_stats: bool = False,
    use_cumulative: bool = False,
    use_feature_selection: bool = False, 
    n_features: Optional[int] = None,
    mode: str = 'CPU'
    ):
    """
    Create a pipeline for a single model with optional lag features and feature selection.
    
    Parameters:
        model: Model instance (SARIMAModels, LSTMModels, etc.).
        preprocessor: Fitted ColumnTransformer from data_preparation.
        model_name (str): Name of the model ('SARIMA', 'LSTM', etc.).
        use_lags (bool): If True, add lag features transformer (default: False).
        use_rolling_stats (bool): If True, add rolling statistics transformer (default: False).
        use_cumulative (bool): If True, add cumulative rolling-sum transformer (default: False).
        use_feature_selection (bool): If True, add feature selection step (default: False).
        n_features (Optional[int]): Number of features to select if use_feature_selection=True.
            If None, uses default (50) (default: None).
        mode (str): Device mode for feature selection GPU acceleration ('CPU', 'GPU', 'CUDA') (default: 'CPU').
    
    Returns:
        Pipeline: sklearn Pipeline with preprocessing, optional lags, optional feature selection, and model.
    """
    # Add preprocessor;
    steps = [("preprocessor", preprocessor)]

    # Add lag features if requested;
    if use_lags:
        steps.append(("lags", LagFeaturesTransformer()))

    # Add cumulative rolling-sum features if requested;
    if use_cumulative:
        steps.append(("cumulative", CumulativeFeaturesTransformer()))

    # Add rolling statistics if requested;
    if use_rolling_stats:
        steps.append(("rolling_stats", RollingStatsTransformer()))
    
    # Add feature selection if requested;
    if use_feature_selection:
        n_feat = n_features if n_features is not None else 50
        steps.append(("feature_selection", FeatureImportanceSelector(n_features=n_feat, mode=mode)))
    
    # Add model as final step;
    steps.append((model_name, model))
    return Pipeline(steps)


def training_pipeline(
    models_to_use: Optional[list] = None,
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
    n_trials: int = 10,
    batch: int = 128,
    freq: str = 'h',
    mode: str = 'CPU',
    use_lags: bool = False,
    use_rolling_stats: bool = False,
    use_cumulative: bool = False,
    use_feature_selection: bool = False,
    n_features: Optional[int] = None,
    log_mlflow: bool = True,
    **kwargs
) -> dict:
    """
    Construct sklearn Pipeline combining preprocessing and model training;
    
    Creates a pipeline that chains the preprocessing step (feature encoding, scaling)
    with optional lag features, optional feature selection, and the selected model;
    
    Parameters:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline from data_preparation;
        models_to_use (Optional[list]): List of model names to use. Options: 'SARIMA', 'LSTM', 
            'XGBOOST', 'LIGHTGBM', 'DUMMY'. If None, trains all models (default: None);
        random_state (int): Random seed for reproducibility (default: 42);
        n_trials (int): Number of trials for hyperparameter optimization (default: 10);
        batch (int): Training batch size (default: 128);
        freq (str): Frequency of predictions (pandas offset) (default: 'h');
        mode (str): Training device mode ('CPU', 'GPU', 'CUDA') (default: 'CPU');
        use_lags (bool): If True, add lag features transformer (default: False).
        use_rolling_stats (bool): If True, add rolling statistics transformer (default: False).
        use_cumulative (bool): If True, add cumulative rolling-sum transformer (default: False).
        use_feature_selection (bool): If True, add feature selection based on RandomForest importance (default: False);
        n_features (Optional[int]): Number of top features to select if use_feature_selection=True.
            If None, uses default (50) (default: None);
        log_mlflow (bool): If True, log experiments to MLFlow (default: True);
        **kwargs: Additional parameters to pass to models;
    
    Returns:
        dict: Dictionary of model names to Pipeline objects with preprocessing, optional lags, 
            optional feature selection, and model steps;
    """
    # Create separate pipeline for each model with arguments;
    pipelines = {
        'SARIMA': create_model_pipeline(
            SARIMAModels(random_state=random_state, n_trials=n_trials, batch=batch, mode=mode, log_mlflow=log_mlflow, **kwargs), 
            preprocessor,
            model_name='SARIMA',
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=20, #Harcoded due to slowness of SARIMA;
            mode=mode
        ),
        'LSTM': create_model_pipeline(
            LSTMModels(random_state=random_state, n_trials=n_trials, batch=batch, mode=mode, log_mlflow=log_mlflow, **kwargs), 
            preprocessor,
            model_name='LSTM',
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=n_features,
            mode=mode
        ),
        'XGBOOST': create_model_pipeline(
            XGBoostModels(random_state=random_state, n_trials=n_trials, batch=batch, mode=mode, log_mlflow=log_mlflow, **kwargs), 
            preprocessor,
            model_name='XGBOOST',
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=n_features,
            mode=mode
        ),
        'LIGHTGBM': create_model_pipeline(
            LightGBMModels(random_state=random_state, n_trials=n_trials, batch=batch, mode=mode, log_mlflow=log_mlflow, **kwargs), 
            preprocessor,
            model_name='LIGHTGBM',
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=n_features,
            mode=mode
        ),
        'DUMMY': create_model_pipeline(
            DummyModels(strategy='mean', random_state=random_state, log_mlflow=log_mlflow, **kwargs),
            preprocessor,
            model_name='DUMMY',
            use_lags=use_lags,
            use_rolling_stats=use_rolling_stats,
            use_cumulative=use_cumulative,
            use_feature_selection=use_feature_selection,
            n_features=n_features,
            mode=mode
        )
    }

    # Remove models that are NOT in the models_to_use list;
    if models_to_use is not None:
        models_to_remove = [model for model in pipelines.keys() if model not in models_to_use]
        for model in models_to_remove:
            pipelines.pop(model)
    return pipelines