"""
Defines the SARIMAModels class for time series forecasting using SARIMA;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
# External libraries;
import pandas as pd

########################################################################################################################
#                                                                  
# MODEL
#
########################################################################################################################
class SARIMAModels:
    def __init__(self, **kwargs):
        """
        """
        self.X = None
        self.y = None
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model.
        """
        # Create copy to avoid modifying the original datasets;
        self.X = X.copy()
        self.y = y.copy()
        
        # Standardize date column;
        self._add_calendar_features()
        
        #TODO: Implement training logic
        return self

    def predict(self, X: pd.DataFrame):
        #TODO: Implement prediction logic
        return []

    def _add_calendar_features(self) -> None:
        """
        Sets 'Data_Hora_Medicao' as the index for time series models;
        """
        if self.X is not None and 'Data_Hora_Medicao' in self.X.columns:
            # Convert to datetime and set as index;
            self.X['Data_Hora_Medicao'] = pd.to_datetime(self.X['Data_Hora_Medicao'])
            self.X.set_index('Data_Hora_Medicao', inplace=True)
            self.X.sort_index(inplace=True)
            
            # Also set index for target if available;
            if self.y is not None:
                self.y.index = self.X.index
        return None