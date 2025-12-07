"""
Main pipeline orchestrator for data processing, transformation, and model training;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import argparse
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.pipeline import Pipeline

# Internal imports;
from source_database.data_io import get_data, save_to_database
from source_database.data_cleaning import clean_dataframe
from source_database.data_transformation import fill_gaps, aggregate_data, melt_dataframe
from source_database.outlier_detection import outlier_removal
from source_database.data_imputation import feature_imputation

########################################################################################################################
#                                                                  
# MAIN PIPELINE
#
########################################################################################################################
def main_database(
    save_to_db: bool = False,
    frequency: str = 'h',
    max_fill_steps: int = 96  # 96 steps of 15 minutes = 1 day;
) -> None:
    """
    Execute complete ETL pipeline: data extraction, transformation, and loading;
    
    Orchestrates the full data processing pipeline: retrieves raw station data, applies cleaning,
    gap filling (CubicSpline interpolation), outlier removal (ECOD+PCA consensus), time aggregation,
    feature imputation (IterativeImputer), and reshaping. Can persist all intermediate results to database.
    This pipeline stops at producing clean, analysis-ready data. Model training is handled separately
    in source_backend;
    
    Parameters:
        save_to_db (bool): If True, save all intermediate dataframes to DuckDB (default: False);
        frequency (str): Pandas frequency string for time aggregation (default: 'h').
            Examples: '15min', 'h', 'D', 'W'. See pandas date offset docs;
        max_fill_steps (int): Max 15-minute intervals to interpolate gaps (default: 96 = 24 hours);
    
    Returns:
        None: Function performs side effects (database writes) but returns None;
    """
    # Read stations data;
    df = get_data()

    # Convert values and cut the dataframe to a time range where most data is available;
    df_cleaned = clean_dataframe(df=df)

    # Fill the data gaps;
    df_filled, missing = fill_gaps(df=df_cleaned, max_fill_steps=max_fill_steps)

    # Identify and remove Outliers (dynamic threshold per station);
    df_out = outlier_removal(df=df_filled, threshold_method='iqr')

    # Aggregate the data to the desired frequency;
    df_agg = aggregate_data(df=df_out, frequency=frequency)

    # Feature Imputation - IterativeImputer with RandomForest;
    df_imp, imputer_stats = feature_imputation(df=df_agg, n_estimators=20)

    # Melt the dataframe;
    df_melted = melt_dataframe(df=df_imp)

    # Save the clean, transformed dataframes to the database;
    # Model training (splitting, encoding, training) is handled in source_backend;
    if save_to_db:
        save_to_database(
            df_cleaned=df_cleaned,
            df_filled=df_filled,
            missing=missing,
            df_out=df_out,
            df_agg=df_agg,
            df_imp=df_imp,
            df_melted=df_melted
        )
    
    print("ETL pipeline completed. Clean data ready for model training.")
    return None

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main ETL pipeline for data processing and transformation")
    
    # Main database parameters;
    parser.add_argument('--frequency', type=str, default='h', help='Pandas frequency string for time aggregation (e.g., 15min, h, D, W)')
    parser.add_argument('--max_fill_steps', type=int, default=96, help='Max 15-minute intervals to interpolate gaps (default: 96 = 24 hours)')
    
    # Bool arguments;
    parser.add_argument('--save_to_db', action='store_true', help='Save all intermediate dataframes to DuckDB')
    parser.add_argument('--no_save_to_db', dest='save_to_db', action='store_false', help='Do not save results to database')
    
    # Set default values for booleans;
    parser.set_defaults(
        save_to_db=True,
    )
    
    args = parser.parse_args()
    print(f'Arguments: {args}')
    
    # Execute main database pipeline with parsed arguments;
    main_database(
        save_to_db=args.save_to_db, 
        frequency=args.frequency, 
        max_fill_steps=args.max_fill_steps
    )
    print('All Done!')