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
    # Initialize the Connection;
    db_handler = DBConnection()

    # Get data for each river;
    cai = river_cai(db=db_handler)
    gravatai = river_gravatai(db=db_handler)
    guaiba = river_guaiba(db=db_handler)
    sinos = river_sinos(db=db_handler)
    taquari = river_taquari(db=db_handler)

    # Concatenate data;

    # Close connection;
    db_handler.close()
    return None

def get_external(db):
    return None

########################################################################################################################
#                                                                  
# SINGLE RIVERS
#
########################################################################################################################
def river_cai(db):
    """
    AAA
    """
    # Query data;
    query = {
        'station_1': 'SELECT * FROM station_cai_1', 
        'station_2': 'SELECT * FROM station_cai_2'
        }
    dataframe = db.run(query=query)
    station_1 = dataframe.get('station_1')
    station_2 = dataframe.get('station_2')
    return None

def river_gravatai(db):
    """
    AAA
    """
    # Query data;
    query = {
        'station_1': 'SELECT * FROM station_gravatai_1', 
        }
    dataframe = db.run(query=query)
    station_1 = dataframe.get('station_1')
    return None

def river_guaiba(db):
    """
    AAA
    """
    # Query data;
    query = {
        'station_1': 'SELECT * FROM station_guaiba_1', 
        'station_2': 'SELECT * FROM station_guaiba_2'
        }
    dataframe = db.run(query=query)
    station_1 = dataframe.get('station_1')
    station_2 = dataframe.get('station_2')
    return None

def river_sinos(db):
    """
    AAA
    """
    # Query data;
    query = {
        'station_1': 'SELECT * FROM station_sinos_1', 
        'station_2': 'SELECT * FROM station_sinos_2',
        'station_3': 'SELECT * FROM station_sinos_3'
        }
    dataframe = db.run(query=query)
    station_1 = dataframe.get('station_1')
    station_2 = dataframe.get('station_2')
    station_3 = dataframe.get('station_3')
    return None

def river_taquari(db):
    """
    AAA
    """
    # Query data;
    query = {
        'station_1': 'SELECT * FROM station_taquari_1', 
        'station_2': 'SELECT * FROM station_taquari_2'
        }
    dataframe = db.run(query=query)
    station_1 = dataframe.get('station_1')
    station_2 = dataframe.get('station_2')
    return None

get_data()