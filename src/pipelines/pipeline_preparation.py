"""
Data preparation utilities for model training;

This module handles train/test splitting, feature encoding, and preprocessing operations
that are specific to machine learning workflows. These operations are separate from
the ETL pipeline in src;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.db_handler import DBConnection
from src.util import configure_logging

logger = configure_logging(__name__)


def encoding_pipeline(
    columns: list,
    target_column: Optional[str] = None
) -> ColumnTransformer:
    """
    Construct sklearn ColumnTransformer for feature preprocessing and encoding;
    
    Creates a preprocessing pipeline with separate processing for numerical and categorical
    features. Numerical features: median imputation + StandardScaler. Categorical features: constant
    imputation + OneHotEncoder. Includes MissingIndicator for missing value tracking;
    
    Parameters:
        numerical_cols (list): List of numerical column names to process (default: []);
        categorical_cols (list): List of categorical column names to process (default: []);
        missing_indicator_cols (list): List of columns to track missing values (default: ['value']);
    
    Returns:
        ColumnTransformer: Configured sklearn ColumnTransformer with preprocessing steps;
            - Numerical pipeline: SimpleImputer(strategy='median') -> StandardScaler();
            - Categorical pipeline: SimpleImputer(strategy='constant') -> OneHotEncoder();
            - MissingIndicator: Tracks missing values in specified columns;
    """
    # Use provided columns
    column_list = columns

    # Define numerical columns;
    exclude_prefixes = {'Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo', 'Data_Hora_Medicao'}
    numerical_cols = [
        col for col in column_list 
        if '_Status' not in col 
        and not any(col.startswith(prefix) for prefix in exclude_prefixes)
        and col != target_column
    ]

    # Define categorical columns;
    categorical_cols = [
        col for col in column_list 
        if ('_Status' in col or col.startswith('Rio_Codigo'))
        and col != target_column
    ]

    # Define numerical pipeline;
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define categorical pipeline;
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        # handle_unknown='ignore' prevents failures when validation/test folds contain categories unseen in the training split; leakage-safe CV needs this;
        ('scaler', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # Create ColumnTransformer;
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical",  categorical_pipeline, categorical_cols),
        ], remainder="passthrough", verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")
    return preprocessor