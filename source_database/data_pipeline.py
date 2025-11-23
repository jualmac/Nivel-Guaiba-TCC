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
    Split dataset into train/test sets using stratified random sampling;
    
    Separates features from target variable and performs train_test_split. Rows with missing target
    values are excluded before splitting. Uses random_state for reproducibility;
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing features and target column;
        target_column (Optional[str]): Name of target column to predict. Required (raises ValueError if None);
        test_size (float): Proportion of dataset allocated to test set, range [0, 1] (default: 0.2);
        random_state (int): Random seed for reproducible splits (default: 42);
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple of (X_train, X_test, y_train, y_test);
            - X_train (pd.DataFrame): Training feature matrix;
            - X_test (pd.DataFrame): Test feature matrix;
            - y_train (pd.Series): Training target vector;
            - y_test (pd.Series): Test target vector;
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
    Construct sklearn Pipeline for feature preprocessing and encoding;
    
    Creates a ColumnTransformer-based pipeline with separate processing for numerical and categorical
    features. Numerical features: median imputation + StandardScaler. Categorical features: constant
    imputation + OneHotEncoder. Includes MissingIndicator for missing value tracking;
    
    Returns:
        Pipeline: Configured sklearn Pipeline with preprocessor step;
            - Numerical pipeline: SimpleImputer(strategy='median') -> StandardScaler();
            - Categorical pipeline: SimpleImputer(strategy='constant') -> OneHotEncoder();
            - MissingIndicator: Tracks missing values in 'value' column;
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
    return preprocessor