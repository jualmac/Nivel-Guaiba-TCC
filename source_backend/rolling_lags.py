"""
AAA
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from sklearn.base import BaseEstimator, TransformerMixin

########################################################################################################################
#
# TRANSFORMER
#
########################################################################################################################
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
    def __init__(self, windows=[7, 30]):
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for window in self.windows:
            X_copy[f'rolling_mean_{window}'] = X_copy['value'].rolling(window).mean()
            X_copy[f'rolling_std_{window}'] = X_copy['value'].rolling(window).std()
        return X_copy.fillna(0)