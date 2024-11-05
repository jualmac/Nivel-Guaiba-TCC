"""
This function calculates the errors of the results from the Regression Models for the models. These are separeted into 
diferent functions, given the different structure of the dataframes. Both functions will apply the error metrics to the
results from every single different model so the models can be compared and the best one determined;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
import numpy as np
from source_backend.metrics import relative_root_mean_squared_error
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

########################################################################################################################
#                                                                  
# ERRORS FUNCTION
#
########################################################################################################################
def errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Root Mean Squared Error (RMSE) for various models across different regions in the provided DataFrame. 
    The function iterates through unique regions, extracting true values and model predictions for each region. It 
    computes the RMSE for each model's predictions, consolidating the results into a single DataFrame that includes the 
    RMSE values along with corresponding regions, which can be used for performance evaluation;

    Parameters:
        - df (pandas DataFrame): Input DataFrame containing model predictions and actual values, with columns for 
        'date', 'region', 'y' (true values) and any number of 'model' columns;

    Returns:
        - full_rmse (pandas DataFrame): A DataFrame containing RMSE values for each model across different regions;
    """
    regions = df['region'].sort_values().unique().tolist()
    full_rmse = pd.DataFrame()

    # Calculate each models RMSE for every item in every rolling window:
    for region in regions:
        region_data = df.loc[df.region == region]
        y_true = region_data['y']

        model = region_data.drop(['date', 'region', 'y'], axis=1)
        rmse = model.apply(lambda x: root_mean_squared_error(y_true=y_true, y_pred=x))
        rmse = pd.DataFrame(rmse).transpose()
        rmse['region'] = region

        full_rmse = pd.concat([full_rmse, rmse])
    return full_rmse