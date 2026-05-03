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

from src.util import convert_to_float, START_DATE, END_DATE

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

# def make_acc_rain(
#     df: pd.DataFrame, 
#     cut: bool = True
# ) -> pd.DataFrame:
#     """
#     Entry dataframe has a 15 minutes frequency;
#     1 day = 96 steps;
#     7 days = 672 steps;
#     30 days = 2880 steps;
#     """
#     print("Correcting Acc Rain values...")
    
#     # Copy dataframe to not propagate changes;
#     df_cpy = df.copy()

#     # Drop incoming Acc Rain;
#     if 'Chuva_Acumulada' in df_cpy.columns:
#         df_cpy.drop(columns='Chuva_Acumulada', inplace=True)
#     if 'Chuva_Acumulada_Status' in df_cpy.columns:
#         df_cpy.drop(columns='Chuva_Acumulada_Status', inplace=True)

#     # Recreate the accumulated rain for certain time windows for each station;
#     stations_list = []
#     for station in df_cpy['codigoestacao'].unique():
#         df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
#         df_station['Chuva_Acumulada_1dia'] = round(df_station['Chuva_Adotada'].rolling(96, min_periods=1).sum(), 2)
#         df_station['Chuva_Acumulada_7dia'] = round(df_station['Chuva_Adotada'].rolling(672, min_periods=1).sum(), 2)
#         df_station['Chuva_Acumulada_30dia'] = round(df_station['Chuva_Adotada'].rolling(2880, min_periods=1).sum(), 2)

#         # Create Status for the accumulated rain columns based on the Status of Chuva_Adotada (Most common value);
#         # Optimized using one-hot encoding + rolling sum to avoid slow rolling().apply();
#         status_dummies = pd.get_dummies(df_station['Chuva_Adotada_Status']).astype(float)
        
#         windows = {
#             'Chuva_Acumulada_1dia_Status': 96,
#             'Chuva_Acumulada_7dia_Status': 672,
#             'Chuva_Acumulada_30dia_Status': 2880
#         }
        
#         if not status_dummies.empty:
#             for col_name, window in windows.items():
#                 # Calculate count of each status in the window
#                 counts = status_dummies.rolling(window, min_periods=1).sum()
#                 # Find status with max count (mode)
#                 modes = counts.idxmax(axis=1)
#                 df_station[col_name] = modes.astype('Int64')
                
#                 # If original status was all NaN/missing (dummies are 0), the sum is 0.
#                 # idxmax returns first column label, which is incorrect. Mask these out.
#                 valid_mask = counts.sum(axis=1) > 0
#                 df_station.loc[~valid_mask, col_name] = pd.NA
#         else:
#              for col_name in windows.keys():
#                  df_station[col_name] = pd.NA

#         # Append each station and concat at the end;
#         stations_list.append(df_station)
#     df_all_stations = pd.concat(stations_list, ignore_index=True)
#     return df_all_stations