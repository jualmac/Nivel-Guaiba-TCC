"""
This file is used to handle the connection and the interactions with the SQLite database;
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import sqlite3
import datetime
import pandas as pd

########################################################################################################################
#
# DATABASE MANIPULATION FUNCTIONS
#
########################################################################################################################
class SQLite_Handler:
    def __init__(self, db_file='znivel.db'):
        """Initialize the SQLite_Handler with the specified SQLite database file."""
        self.db_file = db_file
        self.connection = None

    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.connection.row_factory = sqlite3.Row
            print(f"\033[2;30;41m Connected to SQLite version: {sqlite3.sqlite_version} \033[0;0m")
        except sqlite3.Error as error:
            print("Error while opening SQLite connection:", error)

    def close(self):
        """Close the SQLite database connection."""
        try:
            if self.connection:
                self.connection.close()
                print(f"\033[2;30;43m Disconnected from SQLite version: {sqlite3.sqlite_version} \033[0;0m")
        except sqlite3.Error as error:
            print("Error while closing SQLite connection:", error)

    def read(self, query):
        """Create a DataFrame by executing the given SQL query on the database."""
        if not self.connection:
            self.connect()
        try:
            dataframe = pd.read_sql_query(query, self.connection)
            return dataframe
        except sqlite3.Error as error:
            print("Error executing query:", error)
        finally:
            self.close()

    def write(self, dataframe, table_name, inplace=False):
        """Write a DataFrame to the specified table in the database."""
        if not self.connection:
            self.connect()
        try:
            # Save index if it's a time series
            if dataframe.index.dtype == 'datetime64[ns]':
                dataframe = dataframe.reset_index(drop=False)
                
            # Check if table exists
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{table_name}'")

            if (cursor.fetchone()[0] == 0) or inplace:
                dataframe.to_sql(table_name, self.connection, if_exists='replace', index=False)
                print(f"DataFrame successfully written to new SQLite table '{table_name}'")
            else:
                # Append data without duplicates
                existing_data = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
                
                # Convert date columns to datetime for comparison
                datetime_cols = ['date', 'ds']
                for col in datetime_cols:
                    if col in existing_data.columns:
                        existing_data[col] = pd.to_datetime(existing_data[col])

                # Merge data and remove duplicates
                new_data = pd.concat([existing_data, dataframe], axis=0, ignore_index=True)
                new_data = new_data.drop_duplicates(ignore_index=True)
                new_data.to_sql(table_name, self.connection, if_exists='replace', index=False)

                if not new_data.equals(existing_data):
                    print(f"New data appended to SQLite table '{table_name}'")
                else:
                    print(f"No new data to append to table '{table_name}'")
        except sqlite3.Error as error:
            print(f"Error while writing DataFrame to SQLite table '{table_name}':", error)
        finally:
            self.close()

########################################################################################################################
# Usage Examples:
# # Initialize the SQLite_Handler
# db_handler = SQLite_Handler()

# # Query data
# query = "SELECT * FROM some_table"
# dataframe = db_handler.read(query)

# # Write data
# db_handler.write(dataframe, 'new_table', inplace=True)