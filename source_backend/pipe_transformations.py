"""
Adds transformations to the dataset with lags and stats;
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor

########################################################################################################################
#
# TRANSFORMER
#
########################################################################################################################
class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection based on RandomForest feature importance.
    
    Selects top k features based on RandomForestRegressor importance scores.
    Useful for tree-based models and provides interpretable feature selection.
    """
    def __init__(self, n_features: int = 50, random_state: int = 42):
        """
        Initialize feature importance selector.
        
        Parameters:
            n_features (int): Number of top features to select (default: 50).
            random_state (int): Random seed for RandomForest (default: 42).
        """
        self.n_features = n_features
        self.random_state = random_state
        self.selector = None
        self.selected_features_ = None
        self.numeric_indices_ = None
    
    def fit(self, X, y):
        """
        Fit RandomForest and select top features based on importance.
        
        Parameters:
            X: Feature matrix (pd.DataFrame or np.ndarray).
            y: Target vector (pd.Series or np.ndarray).
        """
        # Ensure X contains only numeric columns for RandomForest
        X_numeric = X
        if isinstance(X, pd.DataFrame):
            # Select only numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            # Save original feature names corresponding to numeric columns
            numeric_feature_names = X_numeric.columns.tolist()
        else:
            # For numpy arrays, assume all are numeric or handle object dtype
            if X.dtype == object:
                pass
        
        # Use RandomForest to compute feature importance;
        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=10
        )
        
        # Fit only on numeric data
        rf.fit(X_numeric, y)
        
        # Select top n_features based on importance;
        self.selector = SelectFromModel(
            rf,
            prefit=True,
            max_features=self.n_features,
            threshold=-np.inf
        )
        
        # Store selected feature names if X is DataFrame;
        if isinstance(X, pd.DataFrame):
            # Get support boolean mask
            support = self.selector.get_support()
            # Map back to the original numeric column names
            selected_numeric_cols = [col for col, selected in zip(numeric_feature_names, support) if selected]
            self.selected_features_ = selected_numeric_cols
        return self
    
    def transform(self, X):
        """
        Transform X to include only selected features.
        
        Parameters:
            X: Feature matrix to transform.
        
        Returns:
            Transformed feature matrix with selected features only.
        """
        # Prepare data same as fit
        X_numeric = X
        preserved_data = None

        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=['number'])
            # Preserve Data_Hora_Medicao if it exists
            if 'Data_Hora_Medicao' in X.columns:
                preserved_data = X['Data_Hora_Medicao']
            
        # Transform using the selector
        X_transformed = self.selector.transform(X_numeric)
        
        # Return DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            df_transformed = pd.DataFrame(
                X_transformed,
                columns=self.selected_features_,
                index=X.index
            )
            
            # Add back Data_Hora_Medicao if it was preserved
            if preserved_data is not None:
                df_transformed['Data_Hora_Medicao'] = preserved_data
                
            return df_transformed
            
        return X_transformed

class LagFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for time series"""
    def __init__(self, lags=[1, 7, 30]):
        self.lags = lags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for lag in self.lags:
            X_copy[f'lag_{lag}'] = X_copy['value'].shift(lag)
        return X_copy.fillna(0)

class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window statistics"""
    def __init__(self, windows=[4, 16, 32, 96]):
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for window in self.windows:
            X_copy[f'rolling_mean_{window}'] = X_copy['value'].rolling(window).mean()
            X_copy[f'rolling_std_{window}'] = X_copy['value'].rolling(window).std()
        return X_copy.fillna(0)