"""
Data cleaning and initial preprocessing operations;

Handles data type conversions, date parsing, and basic data quality fixes;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd

from util import convert_to_float, START_DATE, END_DATE

########################################################################################################################
#                                                                  
# DATA CLEANING FUNCTIONS
#
########################################################################################################################
def clean_dataframe(
    df: pd.DataFrame, 
    cut: bool = True
) -> pd.DataFrame:
    """
    Clean and standardize raw station data types and date ranges;
    
    Converts status columns to nullable Int64, creates temperature status column, converts value
    columns to float, parses datetime columns, and optionally filters to START_DATE/END_DATE range.
    Fills Cota_Adotada gaps using Cota_Manual fallback and updates status codes accordingly;
    
    Parameters:
        df (pd.DataFrame): Raw station dataframe to clean;
        cut (bool): If True, filter data to START_DATE/END_DATE range (default: True);
    
    Returns:
        pd.DataFrame: Cleaned dataframe with standardized types and date filtering applied;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy()

    # Cast'_Status' columns as 'int64';
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    for col in status_cols:
        df_cpy[col] = df_cpy[col].astype('Int64')

    # Create missing temperature status column as integer type;
    df_cpy['Temperatura_Interna_Status'] = pd.Series(dtype='Int64')  # Nullable integer type;
    df_cpy.loc[df_cpy['Temperatura_Interna'].notna(), 'Temperatura_Interna_Status'] = 0
    df_cpy.loc[df_cpy['Temperatura_Interna'].isna(), 'Temperatura_Interna_Status'] = 4

    # Convert the value columns to float (excluding status columns);
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    exclude_cols = {'Data_Atualizacao', 'Data_Hora_Medicao', 'codigoestacao'} | set(status_cols)
    for col in (set(df_cpy.columns) - exclude_cols):
        df_cpy[col] = df_cpy[col].apply(convert_to_float)

    # Convert the date column to datetime;
    df_cpy['Data_Hora_Medicao'] = pd.to_datetime(df_cpy['Data_Hora_Medicao'])
    df_cpy['Data_Atualizacao'] = pd.to_datetime(df_cpy['Data_Atualizacao'])

    # Cut the dataframe to a time range where most data is available;
    if cut:
        df_cpy = df_cpy[df_cpy['Data_Hora_Medicao'] >= START_DATE]
        df_cpy = df_cpy[df_cpy['Data_Hora_Medicao'] <= END_DATE]
    df_cpy = df_cpy.sort_values('Data_Hora_Medicao').reset_index(drop=True)
    
    # Use sensor data to fill the gaps in the level column and respective status;
    was_nan = df_cpy['Cota_Adotada'].isna()
    df_cpy['Cota_Adotada'] = df_cpy['Cota_Adotada'].fillna(df_cpy['Cota_Manual'])
    # df_cpy['Cota_Adotada'] = df_cpy['Cota_Adotada'].fillna(df_cpy['Cota_Sensor']) # This is creating many outliers. Better to remove it;
    
    # Set status to 4 for filled values;
    is_now_filled = was_nan & df_cpy['Cota_Adotada'].notna()
    df_cpy.loc[is_now_filled, 'Cota_Adotada_Status'] = 4
    return df_cpy

