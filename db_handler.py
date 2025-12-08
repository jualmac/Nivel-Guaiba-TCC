"""
This file contains a wrapper for the logic of CRUD operations on DuckDB;
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import logging
import duckdb
import pandas as pd
from typing import Tuple
from util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# CLASS
#
########################################################################################################################
class DBConnection:
    def __init__(self, path: str = "data/nivel_duck.db"):
        self.path = path
        self.connection = self.connect()

    def connect(self):
        """Creates a connection to a DuckDB database file and tests the connection"""
        try:
            # Connect to database and test connection;
            self.connection = duckdb.connect(self.path)
            test_query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                AND table_type = 'BASE TABLE';
            """
            test_connection = self.run(test_query)

            if not test_connection.get('result').empty:
                logger.info("Connection successful to database on file %s", self.path)
                return self.connection
            else:
                raise Exception(f"Connection failed to database on {self.path}, no tables found.")
        except Exception as e:
            logger.error("Error on connect(): %s", e)
            raise

    def run(self, query, params: Tuple = None) -> dict:
        """
        Run one or multiple query and return results as Pandas DataFrames;
        
        Args:
            query (str | dict): A single SQL query string or a dict of {key: query}.
            params (tuple, optional): Parameters for the query (applied only when query is a string).
        
        Returns:
            dict: Dictionary with DataFrames as values.
                  - {"result": DataFrame} if a single query string is passed;
                  - {key: DataFrame} if a dictionary of query is passed;
        """
        results = {}
        try:
            if isinstance(query, dict):
                for key, q in query.items():
                    if params:
                        res = self.connection.execute(q, params)
                        results[key] = pd.DataFrame(res.fetchall(), columns=[d[0] for d in res.description])
                    else:
                        results[key] = self.connection.sql(q).df()
            else:  # Single query string
                if params:
                    res = self.connection.execute(query, params)
                    results["result"] = pd.DataFrame(res.fetchall(), columns=[d[0] for d in res.description])
                else:
                    results["result"] = self.connection.sql(query).df()
            return results
        except Exception as e:
            logger.error("SQL execution failed due to: %s", e)
            return results

    def write(self, df: pd.DataFrame, table_name: str, inplace: bool = False) -> None:
        """Insert a Pandas DataFrame into DuckDB."""
        try:
            tables = self.run("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                AND table_name = ?
                """, (table_name,))

            if inplace or tables.empty:
                # Replace table or create if doesn't exist;
                self.connection.register("tmp_df", df)
                self.connection.sql(f"DROP TABLE IF EXISTS {table_name}")
                self.connection.sql(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
                self.connection.unregister("tmp_df")
                self.connection.table(f"{table_name}").show()
            else:
                # Append data to existing table;
                self.connection.register("tmp_df", df)
                self.connection.sql(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
                self.connection.unregister("tmp_df")
                self.connection.table(f"{table_name}").show()
            logger.info("Inserted data into %s", table_name)
        except Exception as e:
            logger.error("Failed to insert data into %s: %s", table_name, e)
            raise

    def drop(self, objects: dict) -> None:
        """
        Drops multiple tables and/or views at the same time.
        Example:
            objects = {
                "table": ["table_a", "table_b"],
                "view": ["view_a", "view_b"]
            }
        """
        try:
            for obj_type, names in objects.items():
                if not names:
                    continue
                if obj_type.lower() == "table":
                    query = f"DROP TABLE IF EXISTS {', '.join(names)}"
                elif obj_type.lower() == "view":
                    query = f"DROP VIEW IF EXISTS {', '.join(names)}"
                else:
                    logger.error("Unsupported object type: %s", obj_type)
                    continue
                self.connection.execute(query)
            logger.info("Specified tables and views dropped successfully.")
        except Exception as e:
            logger.error("Failed to drop objects due to: %s", e)

    def close(self):
        """Close the DuckDB connection"""
        self.connection.close()

########################################################################################################################
# # Usage Examples:
# # Initialize the Connection;
# db_handler = DBConnection()

# # Query data;
# query = "SELECT * FROM table"
# dataframe = db_handler.run(query=query)

# # Write data;
# db_handler.write(df=dataframe, table_name='new_table', inplace=True)

