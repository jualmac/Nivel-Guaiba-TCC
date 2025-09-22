#TODO: Fix the data gaps -> interpolate() -> SPECIFC FUNCTION;
#TODO: Fix outlier values;
#TODO: Fix the missing level data -> Usar Cota Manual/Cota Sensor;

"""
Creates the datasets for the Machine Learning Models. For this purpose, in this file, there will be a cleaning function
for each river a grouping function and lastly a function to concatenate the main dataset with external data from
different sources;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from source_database.db_handler import DBConnection
from util import convert_to_float, STATIONS_COLS, START_DATE, END_DATE

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def collect_all_stations(save_to_db: bool = False, frequency: str = 'H'):
    """
    Retrieve the data from the stations and return a single dataframe concatenated;

    Parameters:
        save_to_db (bool): Whether to save the dataframe to the database;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'H' for hourly).
            Examples: 'min' (minutes), 'H' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...

    Returns:
        df (pd.DataFrame): The concatenated dataframe;
        df_cleaned (pd.DataFrame): The cleaned and aggregated dataframe;
    """

    # Initialize the Connection;
    db = DBConnection()

    # Query data;
    query = {
        'cai_1':        'SELECT * FROM station_cai_1',
        'cai_2':        'SELECT * FROM station_cai_2',
        'cai_3':        'SELECT * FROM station_cai_3',
        'gravatai_1':   'SELECT * FROM station_gravatai_1',
        'guaiba_1':     'SELECT * FROM station_guaiba_1', 
        'guaiba_2':     'SELECT * FROM station_guaiba_2',
        'jacui_1':      'SELECT * FROM station_jacui_1',
        'jacui_2':      'SELECT * FROM station_jacui_2',
        'sinos_1':      'SELECT * FROM station_sinos_1', 
        'sinos_2':      'SELECT * FROM station_sinos_2',
        'sinos_3':      'SELECT * FROM station_sinos_3',
        'taquari_1':    'SELECT * FROM station_taquari_1', 
        'taquari_2':    'SELECT * FROM station_taquari_2',
        'taquari_3':    'SELECT * FROM station_taquari_3',

        }
    dataframe = db.run(query=query)

    # Open query into single dfs;
    cai_1       =   dataframe.get('cai_1')
    cai_2       =   dataframe.get('cai_2')
    cai_3       =   dataframe.get('cai_3')
    gravatai_1  =   dataframe.get('gravatai_1')
    guaiba_1    =   dataframe.get('guaiba_1')
    guaiba_2    =   dataframe.get('guaiba_2')
    jacui_1     =   dataframe.get('jacui_1')
    jacui_2     =   dataframe.get('jacui_2')
    sinos_1     =   dataframe.get('sinos_1')
    sinos_2     =   dataframe.get('sinos_2')
    sinos_3     =   dataframe.get('sinos_3')
    taquari_1   =   dataframe.get('taquari_1')
    taquari_2   =   dataframe.get('taquari_2')
    taquari_3   =   dataframe.get('taquari_3')

    # Merge guaiba_1 and guaiba_2 into a single dataframe;
    guaiba_1 = pd.concat([guaiba_1, guaiba_2])
    guaiba_1['codigoestacao'] = '87450004'

    # Create a dictionary mapping station names to dataframes for easier identification;
    stations_dataframes = {
        'cai_1':        cai_1,
        'cai_2':        cai_2,
        'cai_3':        cai_3,
        'gravatai_1':   gravatai_1, 
        'guaiba_1':     guaiba_1,
        'jacui_1':      jacui_1,
        'jacui_2':      jacui_2,
        'sinos_1':      sinos_1,
        'sinos_2':      sinos_2,
        'sinos_3':      sinos_3,
        'taquari_1':    taquari_1,
        'taquari_2':    taquari_2,
        'taquari_3':    taquari_3,
    }
    
    # Concatenate the dataframes;
    df = pd.concat(list(stations_dataframes.values()))

    # Select only the columns that are needed;
    df = df[STATIONS_COLS.keys()]
    df.rename(columns=STATIONS_COLS, inplace=True)

    # Cut the dataframe to a time range where most data is available;
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= START_DATE]
    df = df[df['date'] <= END_DATE]

    # Convert the value columns to float;
    for col in (set(df.columns) - {'date', 'station_id'}):
        df[col] = df[col].apply(convert_to_float)

    # Fill the data gaps;
    df_cleaned = fill_gaps(df=df, frequency=frequency)

    # Clean the data using specified frequency;
    df_agg = aggregate_data(df=df_cleaned, frequency=frequency)

    # Melt the dataframe;
    df_melted = melt_dataframe(df=df_agg)

    # Save the dataframes to the database;
    if save_to_db:
        db.write(df=df_cleaned, table_name='data_stations_cleaned', inplace=True)
        db.write(df=df_agg, table_name='data_stations_aggregated', inplace=True)
        db.write(df=df_melted, table_name='data_stations_melted', inplace=True)

        db.write(df=frequency_results, table_name='data_stations_frequency', inplace=True)
        db.write(df=gaps_df, table_name='data_stations_gaps', inplace=True)
    return df, df_agg, df_melted

def aggregate_data(df: pd.DataFrame, frequency: str = 'H'):
    """
    Clean the concatenated data from the different stations. Cuts the dataframe to a time range where most data is
    available. Groups the data by specified frequency intervals and melts the dataframe to long format -> Better for the Machine
    Learning Models;

    Parameters:
        df (pd.DataFrame): The dataframe to clean;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'H' for hourly).
            Examples: 'min' (minutes), 'H' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...
        max_ffill_steps (int): The maximum number of steps to forward-fill the data;

    Returns:
        df_cleaned (pd.DataFrame): The cleaned dataframe;
    """
    # Data Aggregation using specified frequency;
    agg_dict = {'level': 'mean',
                'level_status': 'median',
                'rainfall': 'mean',
                'rainfall_status': 'median',
                'rainfall_accumulated': 'mean',
                'rainfall_accumulated_status': 'median',
                'flow': 'mean',
                'flow_status': 'median',
                'temperature': 'mean'
                }

    # Resample the data to the desired frequency to create continuous timeline and fill gaps;
    df_agg = (
        df.groupby('station_id', group_keys=True)
          .apply(lambda g: g.resample(frequency, on='date').agg(agg_dict))
          .reset_index()
    )

    # Convert the value columns to float and round to 3 decimal places;
    for col in (set(df_agg.columns) - {'date', 'station_id'}):
        df_agg[col] = df_agg[col].apply(convert_to_float)
        df_agg[col] = df_agg[col].round(3)
    return df_agg

def fill_gaps(df: pd.DataFrame, frequency: str = 'H', max_ffill_steps: int = 8):
    """
    Fill the gaps in the dataframe;

    Parameters:
        df (pd.DataFrame): The dataframe to fill the gaps;
        frequency (str): The frequency to fill the gaps;
        max_ffill_steps (int): The maximum number of steps to forward-fill the data;
    """
    # Fill short gaps;
    for col in df.columns:
        df[col] = df.groupby('station_id')[col].ffill(limit=max_ffill_steps)  
    return df

def melt_dataframe(df: pd.DataFrame):
    """
    Melt the dataframe to long format;

    Parameters:
        df (pd.DataFrame): The dataframe to melt;
    """
    # Get the value columns (excluding date and station_id);
    value_cols = [col for col in df.columns if col not in ['date', 'station_id']]
    
    # Melt the dataframe to long format;
    melted = df.melt(id_vars=['date', 'station_id'], 
                     value_vars=value_cols,
                     var_name='metric', 
                     value_name='value')
    
    # Create new column names combining metric and station_id;
    melted['new_col'] = melted['metric'] + '_' + melted['station_id'].astype(str)
    
    # Check for duplicates in date + station_id + metric combinations;
    duplicate_mask = melted.duplicated(subset=['date', 'station_id', 'metric'], keep=False)
    duplicates_df = melted[duplicate_mask].sort_values(['date', 'station_id', 'metric'])
    
    if len(duplicates_df) > 0:
        print(f"Found {len(duplicates_df)} duplicate records (date + station_id + metric combinations):")
        print(f"Number of unique duplicate combinations: {len(duplicates_df.drop_duplicates(subset=['date', 'station_id', 'metric']))}")
        print("\nFirst 20 duplicate records:")
        print(duplicates_df.head(20))
        print("\nDuplicate summary by combination:")
        duplicate_counts = melted.groupby(['date', 'station_id', 'metric']).size()
        print(duplicate_counts[duplicate_counts > 1].head(10))
    else:
        print("No duplicates found in date + station_id + metric combinations")
    
    # Pivot to wide format using pivot_table to handle duplicates;
    df_pivoted = melted.pivot_table(index='date', columns='new_col', values='value', aggfunc='first')
    
    # Reset index and return pivoted dataframe;
    df_cleaned = df_pivoted.reset_index()
    return df_cleaned

if __name__ == "__main__":
    df, df_agg, df_melted = collect_all_stations(save_to_db=False, frequency='H')
    print("All Done!")