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

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def get_data():
    """
    Main function;
    """
    get_rivers()
    return None

def get_external(db):
    return None

def get_rivers():
    """
    AAA
    """
    # Initialize the Connection;
    db_handler = DBConnection()

    # Query data;
    query = {
        'cai_1': 'SELECT * FROM station_cai_1', 
        'cai_2': 'SELECT * FROM station_cai_2',
        'gravatai_1': 'SELECT * FROM station_gravatai_1',
        'guaiba_1': 'SELECT * FROM station_guaiba_1', 
        'guaiba_2': 'SELECT * FROM station_guaiba_2',
        'sinos_1': 'SELECT * FROM station_sinos_1', 
        'sinos_2': 'SELECT * FROM station_sinos_2',
        'sinos_3': 'SELECT * FROM station_sinos_3',
        'taquari_1': 'SELECT * FROM station_taquari_1', 
        'taquari_2': 'SELECT * FROM station_taquari_2'
        }
    dataframe = db_handler.run(query=query)

    # Open query into single dfs;
    cai_1       =   dataframe.get('cai_1')
    cai_2       =   dataframe.get('cai_2')
    gravatai_1  =   dataframe.get('gravatai_1')
    guaiba_1    =   dataframe.get('guaiba_1')
    guaiba_2    =   dataframe.get('guaiba_2')
    sinos_1     =   dataframe.get('sinos_1')
    sinos_2     =   dataframe.get('sinos_2')
    sinos_3     =   dataframe.get('sinos_3')
    taquari_1   =   dataframe.get('taquari_1')
    taquari_2   =   dataframe.get('taquari_2')
    
    # Close connection;
    db_handler.close()
    return None

get_data()