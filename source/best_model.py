"""
Contains the algorithms to select the best models for the Regression. Both are based on the given errors calculated on 
the calculate_errors.py function. For the itens, the algorithm selects the best model based on the lowest error metric;
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import pandas as pd

########################################################################################################################
#
# BEST MODEL FUNCTION
#
########################################################################################################################
def best_model(df: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the best forecasting model based on minimum errors from the provided errors DataFrame, and prepares a new 
    DataFrame with relevant information;

    Parameters:
        - df (pd.DataFrame): A DataFrame containing forecast dates and regions;
        - errors (pd.DataFrame): A DataFrame containing error metrics for different models;

    Returns:
        - df (pd.DataFrame): A DataFrame containing the date, selected forecast, target values, region, and month;
    """
    regional_results = pd.DataFrame()
    df['month'] = df['date'].dt.month

    regions = errors['region'].sort_values().unique().tolist()
    float_columns = errors.select_dtypes(include=['float']).columns

    for region in regions:
        df_region = df.loc[df.region == region]
        errors_region = errors.loc[errors.region == region]

        # Detemine the model with minimum error;
        min_col = str(errors_region[float_columns].idxmin(axis=1).values)
        min_col = min_col.replace("'", "").replace("[", "").replace("]", "")

        df_region = df_region[['date', min_col, 'y', 'region', 'month']]
        df_region.rename(columns={df_region.columns[1]: 'forecast', 'y': 'target'}, inplace=True)
        df_region['model'] = min_col
        regional_results = pd.concat([regional_results, df_region])

    return regional_results