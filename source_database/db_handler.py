"""
This file contains a wrapper for the logic of CRUD operations on DuckDB;
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import duckdb
import pandas as pd
from typing import Tuple

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

            if not test_connection.empty:
                print(f"Connection successful to database on file {self.path}")
                return self.connection
            else:
                raise Exception(f"Connection failed to database on {self.path}, no tables found.")
        except Exception as e:
            print(f"Error on connect(): {e}")
            raise

    def run(self, query: str, params: Tuple = None) -> pd.DataFrame:
        """
        Run a query and return a Pandas DataFrame (to mirror PostgreSQL handler).
        """
        if params:
            result = self.connection.execute(query, params)
            return pd.DataFrame(result.fetchall(), columns=[d[0] for d in result.description])
        else:
            return self.connection.sql(query).df()

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
            print(f"Inserted data into {table_name}")
        except Exception as e:
            print(f"Failed to insert data into {table_name}: {e}")
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
                    print(f"Unsupported object type: {obj_type}")
                    continue
                self.connection.execute(query)
            print("Specified tables and views dropped successfully.")
        except Exception as e:
            print(f"Failed to drop objects due to: {e}")

    def close(self):
        """Close the DuckDB connection"""
        self.connection.close()