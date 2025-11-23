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


def melt_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Reshape dataframe from long to wide format with station-metric column naming;
    
    Melts dataframe to long format, creates station-metric column names (e.g., 'Cota_Adotada_87450004'),
    then pivots to wide format. Automatically excludes columns where all values are missing.
    Handles duplicate timestamps using pivot_table with aggfunc='first';
    
    Parameters:
        df (pd.DataFrame): Long-format dataframe with Data_Hora_Medicao, codigoestacao, and value columns;
    
    Returns:
        pd.DataFrame: Wide-format dataframe with Data_Hora_Medicao as index and station-metric columns;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy()
    
    # Get the value columns (excluding date and codigoestacao);
    value_cols = [col for col in df_cpy.columns if col not in ['Data_Hora_Medicao', 'codigoestacao']]
    
    # Melt the dataframe to long format first;
    melted = df_cpy.melt(id_vars=['Data_Hora_Medicao', 'codigoestacao'], 
                     value_vars=value_cols,
                     var_name='metric', 
                     value_name='value')
    
    # Create new column names combining metric and codigoestacao;
    melted['new_col'] = melted['metric'] + '_' + melted['codigoestacao'].astype(str)
    
    # Identify and exclude station-metric combinations that are 100% missing;
    # This automatically removes Vazao_Adotada_87450004 and similar problematic columns;
    missing_by_combination = melted.groupby('new_col')['value'].apply(lambda x: x.isna().all())
    fully_missing_combinations = missing_by_combination[missing_by_combination].index.tolist()
    
    if fully_missing_combinations:
        print(f"\nExcluding {len(fully_missing_combinations)} fully missing station-metric combinations:")
        for combo in fully_missing_combinations:
            print(f"  - {combo}")
        melted = melted[~melted['new_col'].isin(fully_missing_combinations)]
    
    # Check for duplicates in Data_Hora_Medicao + codigoestacao + metric combinations;
    duplicate_mask = melted.duplicated(subset=['Data_Hora_Medicao', 'codigoestacao', 'metric'], keep=False)
    duplicates_df = melted[duplicate_mask].sort_values(['Data_Hora_Medicao', 'codigoestacao', 'metric'])
    
    if len(duplicates_df) > 0:
        print(f"Found {len(duplicates_df)} duplicate records (Data_Hora_Medicao + codigoestacao + metric combinations):")
        print(f"Number of unique duplicate combinations: {len(duplicates_df.drop_duplicates(subset=['Data_Hora_Medicao', 'codigoestacao', 'metric']))}")
        print("\nFirst 20 duplicate records:")
        print(duplicates_df.head(20))
        print("\nDuplicate summary by combination:")
        duplicate_counts = melted.groupby(['Data_Hora_Medicao', 'codigoestacao', 'metric']).size()
        print(duplicate_counts[duplicate_counts > 1].head(10))
    else:
        print("No duplicates found in Data_Hora_Medicao + codigoestacao + metric combinations")
    
    # Pivot to wide format using pivot_table to handle duplicates;
    df_pivoted = melted.pivot_table(index='Data_Hora_Medicao', columns='new_col', values='value', aggfunc='first')
    
    # Reset index and return pivoted dataframe;
    df_pivoted = df_pivoted.reset_index()
    return df_pivoted


def save_to_database(
    df_cleaned: Optional[pd.DataFrame] = None,
    df_filled: Optional[pd.DataFrame] = None,
    missing: Optional[pd.DataFrame] = None,
    df_out: Optional[pd.DataFrame] = None,
    df_agg: Optional[pd.DataFrame] = None,
    df_imp: Optional[pd.DataFrame] = None,
    df_melted: Optional[pd.DataFrame] = None
) -> None:
    """
    Persist ETL-processed dataframes to DuckDB database tables;
    
    Writes each provided dataframe to its corresponding table using inplace=True (replaces existing
    data). Only dataframes that are not None are saved. This function is only for ETL data,
    not for ML-processed data (train/test splits, encodings, etc);
    
    Parameters:
        df_cleaned (Optional[pd.DataFrame]): Cleaned data -> 'data_stations_cleaned' (default: None);
        df_filled (Optional[pd.DataFrame]): Gap-filled data -> 'data_stations_filled' (default: None);
        missing (Optional[pd.DataFrame]): Missing values tracking -> 'data_stations_missing' (default: None);
        df_out (Optional[pd.DataFrame]): Outlier-removed data -> 'data_stations_outlier' (default: None);
        df_agg (Optional[pd.DataFrame]): Aggregated data -> 'data_stations_aggregated' (default: None);
        df_imp (Optional[pd.DataFrame]): Imputed data -> 'data_stations_imputed' (default: None);
        df_melted (Optional[pd.DataFrame]): Melted data -> 'data_stations_melted' (default: None);
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
        db.write(df=df_melted, table_name='data_stations_melted', inplace=True)
    
    print("ETL data successfully saved to database.")

