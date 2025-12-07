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
        self.feature_names_ = None
    
    def fit(self, X, y):
        """
        Fit RandomForest and select top features based on importance.
        
        Parameters:
            X: Feature matrix (pd.DataFrame or np.ndarray).
            y: Target vector (pd.Series or np.ndarray).
        """
        # Ensure X contains only numeric columns for RandomForest;
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=['number'])
        else:
            # Wrap numpy array with synthetic column names so downstream selectors. Retain feature name metadata and avoid feature-name warnings;
            X_numeric = pd.DataFrame(X)
        
        # Track feature names for reuse in transform;
        self.feature_names_ = X_numeric.columns.tolist()

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
        
        # Store selected feature names (works for both DataFrame and synthetic names);
        support = self.selector.get_support()
        selected_numeric_cols = [col for col, selected in zip(self.feature_names_, support) if selected]
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
        # Prepare data same as fit;
        preserved_data = None
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=['number'])
            if 'Data_Hora_Medicao' in X.columns:
                preserved_data = X['Data_Hora_Medicao']
        else:
            # Ensure column names align with those used at fit time;
            X_numeric = pd.DataFrame(X, columns=self.feature_names_)
        
        # Transform using the selector;
        X_transformed = self.selector.transform(X_numeric)
        
        # Return DataFrame if input was DataFrame (or wrapped as such above);
        df_transformed = pd.DataFrame(
            X_transformed,
            columns=self.selected_features_,
            index=X_numeric.index
        )
        
        # Add back Data_Hora_Medicao if it was preserved;
        if preserved_data is not None:
            df_transformed['Data_Hora_Medicao'] = preserved_data
        # If original input was ndarray, keep ndarray output to avoid surprising type change;
        if not isinstance(X, pd.DataFrame):
            return df_transformed.to_numpy()
        return df_transformed

class LagFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for time series"""
    def __init__(self, lags=[1, 6, 12, 24, 48]):
        self.lags = lags
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        input_is_df = isinstance(X, pd.DataFrame)
        X_copy = X.copy() if input_is_df else pd.DataFrame(X)

        # Identify columns that should not be lagged (date/index-like columns);
        exclude_names = {"date", "datetime", "timestamp", "data_hora_medicao", "data_hora"};
        engineered_prefixes = ("lag_", "rolling_mean_", "rolling_std_", "cum_sum_")
        non_lag_cols = {
            col for col in X_copy.columns
            if pd.api.types.is_datetime64_any_dtype(X_copy[col])
            or (isinstance(col, str) and (col.lower() in exclude_names or col.startswith(engineered_prefixes)))
        }

        # Add lagged versions for every eligible feature column using concat to avoid fragmentation;
        feature_cols = [col for col in X_copy.columns if col not in non_lag_cols]
        lag_data = {}
        for col in feature_cols:
            for lag in self.lags:
                lag_data[f'lag_{lag}_{col}'] = X_copy[col].shift(lag)
        if lag_data:
            lag_df = pd.DataFrame(lag_data, index=X_copy.index)
            X_copy = pd.concat([X_copy, lag_df], axis=1)

        # Fill NaNs;
        X_copy = X_copy.fillna(0)
        return X_copy if input_is_df else X_copy.to_numpy()

class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window statistics"""
    def __init__(self, windows=[12, 24, 48, 72, 168]):
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        input_is_df = isinstance(X, pd.DataFrame)
        X_copy = X.copy() if input_is_df else pd.DataFrame(X)

        # Identify columns that should not be rolled (date/index-like columns);
        exclude_names = {"date", "datetime", "timestamp", "data_hora_medicao", "data_hora"};
        engineered_prefixes = ("lag_", "rolling_mean_", "rolling_std_", "cum_sum_")
        non_roll_cols = {
            col for col in X_copy.columns
            if pd.api.types.is_datetime64_any_dtype(X_copy[col])
            or (isinstance(col, str) and (col.lower() in exclude_names or col.startswith(engineered_prefixes)))
        }

        # Add rolling statistics for numeric eligible columns using concat to avoid fragmentation;
        feature_cols = [
            col for col in X_copy.columns
            if col not in non_roll_cols and pd.api.types.is_numeric_dtype(X_copy[col])
        ]
        roll_data = {}
        for col in feature_cols:
            for window in self.windows:
                roll_data[f'rolling_mean_{window}_{col}'] = X_copy[col].rolling(window).mean()
                roll_data[f'rolling_std_{window}_{col}'] = X_copy[col].rolling(window).std()
        if roll_data:
            roll_df = pd.DataFrame(roll_data, index=X_copy.index)
            X_copy = pd.concat([X_copy, roll_df], axis=1)

        # Fill NaNs;
        X_copy = X_copy.fillna(0)
        return X_copy if input_is_df else X_copy.to_numpy()

class CumulativeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Create cumulative/rolling-sum features across all eligible columns"""
    def __init__(self, windows=[96, 672, 2880]):
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        input_is_df = isinstance(X, pd.DataFrame)
        X_copy = X.copy() if input_is_df else pd.DataFrame(X)

        # Identify columns that should not be accumulated (date/index-like columns);
        exclude_names = {"date", "datetime", "timestamp", "data_hora_medicao", "data_hora"};
        engineered_prefixes = ("lag_", "rolling_mean_", "rolling_std_", "cum_sum_")
        non_accum_cols = {
            col for col in X_copy.columns
            if pd.api.types.is_datetime64_any_dtype(X_copy[col])
            or (isinstance(col, str) and (col.lower() in exclude_names or col.startswith(engineered_prefixes)))
        }

        # Add rolling-sum cumulative features for numeric eligible columns using concat to avoid fragmentation;
        feature_cols = [
            col for col in X_copy.columns
            if col not in non_accum_cols and pd.api.types.is_numeric_dtype(X_copy[col])
        ]
        accum_data = {}
        for col in feature_cols:
            for window in self.windows:
                accum_data[f'cum_sum_{window}_{col}'] = X_copy[col].rolling(window, min_periods=1).sum()
        if accum_data:
            accum_df = pd.DataFrame(accum_data, index=X_copy.index)
            X_copy = pd.concat([X_copy, accum_df], axis=1)

        # Fill NaNs;
        X_copy = X_copy.fillna(0)
        return X_copy if input_is_df else X_copy.to_numpy()