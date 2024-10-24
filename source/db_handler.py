"""
This file is used to handle the connection and the interactions with the SQLite database;
"""

########################################################################################################################
import sqlite3
import datetime
import pandas as pd
import streamlit as st

# SQLite database name;
db_file = 'znivel.db'

########################################################################################################################
#
# DATABASE MANIPULATION FUNCTIONS
#
########################################################################################################################
def connect_to_database():
    try:
        sqlite_connection = sqlite3.connect(db_file)
        sqlite_connection.row_factory = sqlite3.Row
        cursor = sqlite_connection.cursor()
        print(f"\033[2;30;41m Connected to SQLite version: {sqlite3.sqlite_version} \033[0;0m")
        return sqlite_connection
    except sqlite3.Error as error:
        print("Error while openning sqlite connection", error)

def close_connection(sqlite_connection):
    try:
        if sqlite_connection:
            sqlite_connection.close()
            print(f"\033[2;30;43m Disconnected from SQLite version: {sqlite3.sqlite_version} \033[0;0m")
    except sqlite3.Error as error:
        print("Error while closing sqlite connection:", error)

#TODO Add warning when accessing a table that doesn't exist;
def create_dataframe(query, db_connection):
    #Open Connection;
    try:
        dataframe = pd.read_sql_query(query, db_connection)

    finally:
        close_connection(sqlite_connection)
    return dataframe

def write_dataframe(dataframe, table_name, inplace=False):
    # Save index in case it's a Time Series;
    if (dataframe.index.dtype == 'datetime64[ns]'):
        dataframe = dataframe.reset_index(drop=False)
    sqlite_connection = connect_to_database()
    try:
        # Check if table exists;
        cursor = sqlite_connection.cursor()
        cursor.execute(f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")

        if (cursor.fetchone()[0] == 0) or (inplace==True):
            # Table does not exist, create it and save dataframe OR automatically replace instruction;
            dataframe.to_sql(table_name, sqlite_connection, if_exists='replace', index=False)
            print(f"DataFrame successfully written to new SQLite table '{table_name}'")
        else:
            # Table exists, update or insert data;
            existing_data = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_connection)
            
            # Fix datetime columns if they're present for proper comparision;
            datetime_cols = ['date', 'ds']
            for col in datetime_cols:
                if col in existing_data.columns:
                    existing_data[col] = pd.to_datetime(existing_data[col])

            # Check for duplicates based on data row;
            new_data = pd.concat([existing_data, dataframe], axis=0, ignore_index=True)
            new_data = new_data.drop_duplicates(ignore_index=True)

            new_data.to_sql(table_name, sqlite_connection, if_exists='replace', index=False)

            if not new_data.equals(existing_data):
                print(f"New data appended to SQLite table '{table_name}'")
            else:
                print(f"Nothing to append to table '{table_name}'")
    except sqlite3.Error as error:
        print(f"Error while writing DataFrame to SQLite table '{table_name}':", error)
    finally:
        close_connection(sqlite_connection)

# TODO SIMPLYFY Usage -> Transform into a Class;
# from source.db_handler import connect_to_database, create_dataframe, write_dataframe, close_connection
# query = "a"
# db_connection = connect_to_database()
# dataframe = create_dataframe(query, db_connection)
# close_connection(db_connection)