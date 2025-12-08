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
import logging
import pandas as pd
from typing import Optional

from db_handler import DBConnection
from util import configure_logging

logger = configure_logging(__name__)

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
    df_predictions: Optional[pd.DataFrame] = None,
    df_metrics: Optional[pd.DataFrame] = None,
    df_features: Optional[pd.DataFrame] = None
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
        df_predictions (Optional[pd.DataFrame]): ML model predictions -> 'models_predictions' (default: None);
        df_metrics (Optional[pd.DataFrame]): ML model metrics -> 'models_metrics' (default: None);
        df_features (Optional[pd.DataFrame]): Processed feature matrices (post-preprocessing and feature selection)
            for model inputs -> 'models_features' (default: None);
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
    
    # Save predictions;
    if df_predictions is not None:
        db.write(df=df_predictions, table_name='models_predictions', inplace=True)

    # Save and Update metrics;
    if df_metrics is not None:
        try:
            existing_metrics = db.run("SELECT * FROM models_metrics")['result']
        except:
            existing_metrics = pd.DataFrame()

        if not existing_metrics.empty:
            # Separate current run metrics (non-BEST) and existing BEST metrics;
            existing_best = existing_metrics[existing_metrics['model_name'].str.endswith('_BEST')]
            
            # List to hold updated rows;
            updated_rows = []
            
            # Add current run metrics (overwriting any non-BEST rows naturally by just inserting them);
            updated_rows.append(df_metrics)
            
            # Process BEST metrics;
            for _, row in df_metrics.iterrows():
                model = row['model_name']
                best_model_name = f"{model}_BEST"
                
                # Get existing best for this model if any;
                current_best_row = existing_best[existing_best['model_name'] == best_model_name]
                
                if not current_best_row.empty:
                    current_best_rmse = current_best_row.iloc[0]['rmse']
                    new_rmse = row['rmse']
                    
                    # Compare RMSE (lower is better);
                    if new_rmse < current_best_rmse:
                         # New record is better, create BEST row from current row;
                         best_row = row.copy()
                         best_row['model_name'] = best_model_name
                         updated_rows.append(pd.DataFrame([best_row]))
                    else:
                         # Keep existing best;
                         updated_rows.append(current_best_row)
                else:
                    # No existing best, create one;
                    best_row = row.copy()
                    best_row['model_name'] = best_model_name
                    updated_rows.append(pd.DataFrame([best_row]))
            
            # Concatenate all;
            final_metrics = pd.concat(updated_rows, ignore_index=True)
            
            # We also need to keep existing BEST rows for models NOT in the current run (if any);
            current_models_best = [f"{m}_BEST" for m in df_metrics['model_name'].unique()]
            other_best = existing_best[~existing_best['model_name'].isin(current_models_best)]
            if not other_best.empty:
                final_metrics = pd.concat([final_metrics, other_best], ignore_index=True)

        else:
            # No existing metrics, just create current + best;
            best_metrics = df_metrics.copy()
            best_metrics['model_name'] = best_metrics['model_name'] + '_BEST'
            final_metrics = pd.concat([df_metrics, best_metrics], ignore_index=True)
            
        db.write(df=final_metrics, table_name='models_metrics', inplace=True)
    
    # Save processed feature matrices for model inputs;
    if df_features is not None:
        db.write(df=df_features, table_name='models_features', inplace=True)
    logger.info("Data successfully saved to database.")