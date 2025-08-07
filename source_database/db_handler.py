"""
Database Connection Handler for PostgreSQL. This module provides the `DBConnection` class to facilitate interactions 
with the PostgreSQL database. It supports connection management, execution of SQL queries, and data manipulation using 
Pandas DataFrames;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.url import URL

########################################################################################################################
#                                                                  
# DATABASE CONNECTION
#
########################################################################################################################
class DBConnection:
    def __init__(self, sql: str = "", params: dict = None, schema_name: str = "public"):
        self.sql = sql
        self.params = params
        self.pg_schema = schema_name
        self.engine = self.create_db_engine()

    def create_db_engine(self):
        """Creates a SQLAlchemy engine with the proper connection and schema."""
        db_url = URL.create(
            drivername="postgresql",
            username=os.environ["PG_USER"],
            password=os.environ["PG_PASS"],
            host=os.environ["PG_HOST"],
            port=os.environ["PG_PORT"],
            database=os.environ["PG_NAME"]
        )
        engine = create_engine(db_url, connect_args={"options": f"-c search_path={self.pg_schema}"})
        print(f"Connected to the Database {os.environ['PG_HOST']} on Schema {self.pg_schema}")
        return engine

    def run_sql(self) -> dict:
        """
        Executes SQL queries provided as either a string or a dictionary.
        Returns a dictionary where each key corresponds to a query's result DataFrame.
        """
        results = {}
        try:
            with self.engine.connect() as connection:
                if isinstance(self.sql, dict):
                    for key, query in self.sql.items():
                        res = connection.execute(text(query), self.params or {})
                        data = res.fetchall()
                        results[key] = pd.DataFrame(data, columns=res.keys())
                else:
                    res = connection.execute(text(self.sql), self.params or {})
                    data = res.fetchall()
                    results["result"] = pd.DataFrame(data, columns=res.keys())
            return results
        except Exception as e:
            print(f"SQL execute query failed due to: {e}")
            return results

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append", index: bool = False) -> None:
        """
        Inserts a DataFrame into the specified table using SQLAlchemy's to_sql method.
        
        Args:
            df: The DataFrame to insert.
            table_name: The target table name.
            if_exists: How to behave if the table already exists.
                       Options: 'fail', 'replace', 'append'. Default is 'append'.
            index: Whether to write DataFrame's index as a column. Default is False.
        """
        print(f"Inserting data into table {table_name}...")
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            print(f"Data inserted successfully into table {table_name}")
        except Exception as e:
            print(f"Failed to insert data into table {table_name} due to: {e}")

    def create_view(self, query: str) -> None:
        """Creates or replaces a view in the database using SQLAlchemy."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text(query))
                connection.commit()  # Commit DDL changes
            print("View created (or replaced) successfully.")
        except SQLAlchemyError as e:
            print(f"Failed to create view due to: {e}")

    def drop_objects(self, objects: dict) -> None:
        """
        Drops multiple tables and/or views at the same time.
        The objects parameter should be a dictionary with keys "table" and/or "view"
        and values as lists of object names. For example:
        
            objects = {
                "table": ["raw_clean", "data_item"],
                "view": ["cliente_por_produto", "rfv"]
            }
        
        Each drop is executed with CASCADE.
        """
        try:
            with self.engine.connect() as connection:
                for obj_type, names in objects.items():
                    if not names:
                        continue
                    if obj_type.lower() == "table":
                        query = f"DROP TABLE IF EXISTS {', '.join(names)} CASCADE;"
                    elif obj_type.lower() == "view":
                        query = f"DROP VIEW IF EXISTS {', '.join(names)} CASCADE;"
                    else:
                        print(f"Unsupported object type: {obj_type}")
                        continue
                    connection.execute(text(query))
                connection.commit()
            print("Specified tables and views dropped successfully.")
        except Exception as e:
            print(f"Failed to drop objects due to: {e}")
    
    def close_connection(self):
        """Closes the database connection."""
        self.engine.dispose()
        print("Database connection closed.")