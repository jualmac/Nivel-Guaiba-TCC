"""
Data preparation utilities for model training;

This module handles train/test splitting, feature encoding, and preprocessing operations
that are specific to machine learning workflows. These operations are separate from
the ETL pipeline in source_database;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
from typing import Tuple, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split
from db_handler import DBConnection
from util import STATION_COLS

########################################################################################################################
#                                                                  
# DATA PREPARATION FUNCTIONS
#
########################################################################################################################
def data_division(
    df: pd.DataFrame, 
    target_column: str = None,
    train_size: float = 0.8, 
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    shuffle: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Split dataset into train/test or train/validation/test sets for time series;
    
    Separates features from target variable and performs train_test_split. Rows with missing target
    values are excluded before splitting. Uses random_state for reproducibility. For time series data,
    shuffle should be False to preserve temporal order;
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing features and target column;
        target_column (Optional[str]): Name of target column to predict. Required (raises ValueError if None);
        test_size (float): Proportion of dataset allocated to test set, range [0, 1] (default: 0.2);
        val_size (Optional[float]): Proportion of dataset allocated to validation set for early stopping.
            If None, performs two-way split (Train/Test). If provided, performs three-way split (Train/Val/Test).
            Range [0, 1]. Default: None;
        random_state (int): Random seed for reproducible splits (default: 42);
        shuffle (bool): Whether to shuffle data before splitting. Should be False for time series data
            to preserve temporal order (default: False);
    
    Returns:
        If val_size is None:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: (X_train, X_test, y_train, y_test);
        If val_size is provided:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
                (X_train, X_val, X_test, y_train, y_val, y_test);
    """
    # Validate target column;
    if target_column is None:
        raise ValueError("target_column must be provided (cannot be None)")
    
    # Validate split sizes;
    if val_size is not None and val_size > 0:
        if train_size + val_size + test_size != float(1.0):
            raise ValueError("Train + Validation + Test Size must equal to 1")
        if val_size < 0 or val_size >= 1:
            raise ValueError("val_size must be in range (0, 1)")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be in range (0, 1)")
        if (val_size + test_size) >= 1.0:
            raise ValueError(f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0")
    else:
        if train_size + test_size != float(1.0):
            raise ValueError("Train + Test Size must equal to 1")
        
    # Split features and target;
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Three-way split (Train/Val/Test) for early stopping;
    if val_size is not None and val_size > 0:
        # First split: Train vs (Val + Test);
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(val_size + test_size),
            random_state=random_state,
            shuffle=shuffle  # False for time series to preserve temporal order;
        )
        
        # Second split: Val vs Test;
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=random_state,
            shuffle=shuffle  # False for time series to preserve temporal order;
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Two-way split (Train/Test) - default behavior;
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle  # False for time series to preserve temporal order;
        )
        return X_train, X_test, y_train, y_test


def encoding_pipeline(
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
    # Get list of column names from database (the table should be selectable);
    print("Loading data from database...")
    db = DBConnection()
    column_names = db.run(
        """
        SELECT 
            column_name 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME='data_stations'
        """
        )['result']
    column_list = list(column_names['column_name'])

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
        ('scaler', OneHotEncoder(sparse_output=False))
    ])

    # Create ColumnTransformer;
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical",  categorical_pipeline, categorical_cols),
        ], remainder="passthrough", verbose_feature_names_out=False)
    preprocessor.set_output(transform="pandas")
    return preprocessor