"""
Data input/output operations for database reading and writing;

Handles all database interactions including reading raw station data and persisting
processed dataframes to DuckDB;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
from typing import Optional

from db_handler import DBConnection

########################################################################################################################
#                                                                  
# DATA I/O FUNCTIONS
#
########################################################################################################################
def get_data() -> pd.DataFrame:
    """
    Query and concatenate raw station data from DuckDB database;
    
    Executes SELECT queries for all station tables and merges guaiba_1 and guaiba_2 into a single
    continuous time series (guaiba_2 served as backup during 2024 floods). All stations are
    concatenated into a unified dataframe;
    
    Returns:
        pd.DataFrame: Combined dataframe with all station data, guaiba stations merged;
    """
    # Initialize the Connection;
    db = DBConnection()

    # Query data;
    query = {
        'cai_1':        'SELECT * FROM station_cai_1',
        'cai_2':        'SELECT * FROM station_cai_2',
        'gravatai_1':   'SELECT * FROM station_gravatai_1',
        'guaiba_1':     'SELECT * FROM station_guaiba_1',
        'guaiba_2':     'SELECT * FROM station_guaiba_2',
        'jacui_1':      'SELECT * FROM station_jacui_1',
        'sinos_1':      'SELECT * FROM station_sinos_1',
        'sinos_2':      'SELECT * FROM station_sinos_2',
        'taquari_1':    'SELECT * FROM station_taquari_1',
        'taquari_2':    'SELECT * FROM station_taquari_2',
        }
    dataframe = db.run(query=query)

    # Merge guaiba_1 and guaiba_2 into a single dataframe -> This is because the guaiba_2 station was setted as a backup to guaiba_1 during the 2024 floods. Therefore, their data should be considered as a continuous time series;
    guaiba_merged = pd.concat([dataframe['guaiba_1'], dataframe['guaiba_2']])
    guaiba_merged['codigoestacao'] = '87450004'
    
    # Update the dictionary with the merged guaiba and remove guaiba_2;
    dataframe['guaiba_1'] = guaiba_merged
    dataframe.pop('guaiba_2')
    
    # Concatenate all station dataframes;
    df = pd.concat(list(dataframe.values()))
    return df


def save_to_database(
    df_cleaned: Optional[pd.DataFrame] = None,
    df_filled: Optional[pd.DataFrame] = None,
    missing: Optional[pd.DataFrame] = None,
    df_out: Optional[pd.DataFrame] = None,
    df_agg: Optional[pd.DataFrame] = None,
    df_imp: Optional[pd.DataFrame] = None,
    df_melted: Optional[pd.DataFrame] = None,
    df_ml_results: Optional[pd.DataFrame] = None
) -> None:
    """
    Persist ETL-processed and ML results dataframes to DuckDB database tables;
    
    Writes each provided dataframe to its corresponding table using inplace=True (replaces existing
    data). Only dataframes that are not None are saved. Handles both ETL data and ML model results;
    
    Parameters:
        df_cleaned (Optional[pd.DataFrame]): Cleaned data -> 'data_stations_cleaned' (default: None);
        df_filled (Optional[pd.DataFrame]): Gap-filled data -> 'data_stations_filled' (default: None);
        missing (Optional[pd.DataFrame]): Missing values tracking -> 'data_stations_missing' (default: None);
        df_out (Optional[pd.DataFrame]): Outlier-removed data -> 'data_stations_outlier' (default: None);
        df_agg (Optional[pd.DataFrame]): Aggregated data -> 'data_stations_aggregated' (default: None);
        df_imp (Optional[pd.DataFrame]): Imputed data -> 'data_stations_imputed' (default: None);
        df_melted (Optional[pd.DataFrame]): Melted data -> 'data_stations' (default: None);
        df_ml_results (Optional[pd.DataFrame]): ML model predictions and metrics -> 'ml_model_results' (default: None);
    """
    # Initialize the Connection;
    db = DBConnection()

    # Save main pipeline dataframes only if provided;
    if df_cleaned is not None:
        db.write(df=df_cleaned, table_name='data_stations_cleaned', inplace=True)
    if df_filled is not None:
        db.write(df=df_filled, table_name='data_stations_filled', inplace=True)
    if missing is not None:
        db.write(df=missing, table_name='data_stations_missing', inplace=True)
    if df_out is not None:
        db.write(df=df_out, table_name='data_stations_outlier', inplace=True)
    if df_agg is not None:
        db.write(df=df_agg, table_name='data_stations_aggregated', inplace=True)
    if df_imp is not None:
        db.write(df=df_imp, table_name='data_stations_imputed', inplace=True)
    if df_melted is not None:
        db.write(df=df_melted, table_name='data_stations', inplace=True)
    if df_ml_results is not None:
        db.write(df=df_ml_results, table_name='ml_model_results', inplace=True)
    
    print("Data successfully saved to database.")