"""
AAA
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import duckdb

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
class DBConnection:
    def __init__(self, path: str = "/data/niveldb"):
        self.path = path
        self.connection = self.connect()  

    def connect(self):
        """Creates a connection to a DuckDB database file and tests the connection"""
        try:
            self.connection = duckdb.connect(self.path)

            # Test query; 
            test_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main';
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

    def run(self, query: str = None, params: Tuple = None):
        """Run a query and return a DuckDB relation"""
        if params:
            return self.connection.execute(query, params)
        else:
            return self.connection.sql(query).to_df()        
        
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
        """Insert DataFrame into a table"""
        if if_exists == "replace":
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.register("tmp_df", df)
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        else:  
            self.conn.register("tmp_df", df)
            self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
        print(f"Inserted data into {table_name}")

    def close(self):
        """Close the DuckDB connection"""
        self.connection.close()