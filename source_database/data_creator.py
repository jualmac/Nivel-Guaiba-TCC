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
from util import convert_to_float
########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def get_data():
    """
    AAA
    """
    # Initialize the Connection;
    db_handler = DBConnection()

    # Query data;
    query = {
        'gravatai_1':   'SELECT * FROM station_gravatai_1',
        'guaiba_1':     'SELECT * FROM station_guaiba_1', 
        'guaiba_2':     'SELECT * FROM station_guaiba_2',
        'sinos_1':      'SELECT * FROM station_sinos_1', 
        'sinos_2':      'SELECT * FROM station_sinos_2',
        'sinos_3':      'SELECT * FROM station_sinos_3',
        'taquari_1':    'SELECT * FROM station_taquari_1', 
        # 'taquari_2':    'SELECT * FROM station_taquari_2',
        # 'jacui_1':      'SELECT * FROM station_jacui_1',
        # 'cai_1':        'SELECT * FROM station_cai_1',
        # 'cai_2':        'SELECT * FROM station_cai_2',
        }
    dataframe = db_handler.run(query=query)

    # Open query into single dfs;
    gravatai_1  =   dataframe.get('gravatai_1')
    guaiba_1    =   dataframe.get('guaiba_1')
    guaiba_2    =   dataframe.get('guaiba_2')
    sinos_1     =   dataframe.get('sinos_1')
    sinos_2     =   dataframe.get('sinos_2')
    sinos_3     =   dataframe.get('sinos_3')
    taquari_1   =   dataframe.get('taquari_1')
    # taquari_2   =   dataframe.get('taquari_2')
    # jacui_1     =   dataframe.get('jacui_1')
    # cai_1       =   dataframe.get('cai_1')
    # cai_2       =   dataframe.get('cai_2')

    # Merge guaiba_1 and guaiba_2 into a single dataframe;
    guaiba_1 = pd.concat([guaiba_1, guaiba_2])
    guaiba_1['codigoestacao'] = '87450004'

    # Create a dictionary mapping station names to dataframes for easier identification;
    dataframes = {
        'gravatai_1':   gravatai_1, 
        'guaiba_1':     guaiba_1,
        'sinos_1':      sinos_1,
        'sinos_2':      sinos_2,
        'sinos_3':      sinos_3,
        'taquari_1':    taquari_1,
        # 'taquari_2':    taquari_2,
        # 'jacui_1':      jacui_1,
        # 'cai_1':        cai_1,
        # 'cai_2':        cai_2,
    }

    # Concatenate the dataframes;
    df = pd.concat(list(dataframes.values()))

    # Select only the columns that are needed;
    cols = {'Data_Hora_Medicao': 'date', 
            'codigoestacao': 'station_id', 
            'Cota_Adotada': 'level',
            'Cota_Adotada_Status': 'level_status',
            'Chuva_Acumulada': 'rainfall',
            'Chuva_Acumulada_Status': 'rainfall_status',
            'Chuva_Adotada': 'rainfall_adopted',
            'Chuva_Adotada_Status': 'rainfall_adopted_status',
            'Temperatura_Agua': 'temperature'
            }

    df = df[cols.keys()]

    # Rename the columns;
    df.rename(columns=cols, inplace=True)

    # Convert the date column to datetime;
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2018-08-01']

    # Convert the value columns to float;
    for col in ['level', 'rainfall', 'rainfall_adopted', 'temperature']:
        df[col] = df[col].apply(convert_to_float)

    # Group each station by 1 hour intervals;
    df['hour_timestamp'] = df['date'].dt.floor('H')
    df = df.groupby(['station_id', 'hour_timestamp']).agg({'level': 'mean', 
                                                           'rainfall': 'mean', 
                                                           'rainfall_adopted': 'mean', 
                                                           'temperature': 'mean',
                                                           'level_status': 'first',
                                                           'rainfall_status': 'first',
                                                           'rainfall_adopted_status': 'first'}).reset_index()
    df.rename(columns={'hour_timestamp': 'date'}, inplace=True)

    # Get the value columns (excluding date and station_id)
    value_cols = [col for col in df.columns if col not in ['date', 'station_id']]
    
    # Melt the dataframe to long format
    melted = df.melt(id_vars=['date', 'station_id'], 
                     value_vars=value_cols,
                     var_name='metric', 
                     value_name='value')
    
    # Create new column names combining metric and station_id
    melted['new_col'] = melted['metric'] + '_' + melted['station_id'].astype(str)
    
    # Check for duplicates in date + station_id + metric combinations
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
    
    # Pivot to wide format using pivot_table to handle duplicates
    df_pivoted = melted.pivot_table(index='date', columns='new_col', values='value', aggfunc='first')
    
    # Reset index and return pivoted dataframe
    df = df_pivoted.reset_index()
    
    db_handler.write(df=df, table_name='data_stations', inplace=True)
    return df

def get_external(db):
    """
    #TODO: Get the external data from the database;
    """
    return None

if __name__ == "__main__":
    df = get_data()
    print(df.head())