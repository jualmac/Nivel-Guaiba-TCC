#TODO: Define Scikit-learn Pipeline here;
#TODO: Apply CuPy transformation to the data;
#TODO: Feature Importance;
#TODO: Feature Scaling;
#TODO: Implement MLFlow;
"""
AAA
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
# External libraries;
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Models;
#TODO: Define Models (internal import from source_backend)

########################################################################################################################
#                                                                  
# PIPELINE
#
########################################################################################################################
def data_division(
    df: pd.DataFrame, 
    target_column: Optional[str] = None, 
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide the dataset into train and test data;
    
    Parameters:
        df (pd.DataFrame): The dataframe to split;
        target_column (str): Name of the target column to predict. Must be present in dataframe columns;
        test_size (float): Proportion of dataset to include in the test split (default: 0.2);
        random_state (int): Random seed for reproducibility (default: 42);
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_train (pd.DataFrame): Training features;
            - X_test (pd.DataFrame): Test features;
            - y_train (pd.Series): Training target;
            - y_test (pd.Series): Test target;
    """
    # Validate target column;
    if target_column is None:
        raise ValueError("target_column must be provided (cannot be None)")
    
    # Split features and target;
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Split data;
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state
                                                        )
    return X_train, X_test, y_train, y_test


def encoding_pipeline(
) -> Pipeline:
    """
    Define the preprocessing pipeline the data. This method creates a scikit-learn pipeline with 
    appropriate transformers for different column types in analysis data. It handles numerical features, categorical 
    features of different cardinalities, and missing values.

    Parameters:
        df (pd.DataFrame): The dataframe to preprocess;

    Returns:
        Pipeline: The preprocessing pipeline;
    """
    # Define the columns; #TODO: Define the columns;
    numerical_cols = []
    categorical_cols = []

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols),
        ("missing_indicator", MissingIndicator(features="missing-only"), ['value'])
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # ("ARIMA", ARIMA_Model), #TODO: Define models here;
        # ("LSTM", LSTM_Model), #TODO: Define models here;
        # ("XGBOOST", XGB_Model), #TODO: Define models here;
        # ("LIGHTGBM", LGB_Model), #TODO: Define models here;
    ])
    return pipeline