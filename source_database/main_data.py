"""
Main pipeline orchestrator for data processing, transformation, and model training;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.pipeline import Pipeline

# Internal imports;
from source_database.data_transformation import (
    get_data,
    clean_dataframe,
    fill_gaps,
    outlier_removal,
    aggregate_data,
    feature_imputation,
    melt_dataframe,
    save_to_database
)
from source_database.data_pipeline import (
    data_division,
    encoding_pipeline
)

########################################################################################################################
#                                                                  
# MAIN PIPELINE
#
########################################################################################################################
def main_database(
    save_to_db: bool = False,
    frequency: str = 'h',
    max_fill_steps: int = 96, # 96 steps of 15 minutes = 1 day;
    target_column: Optional[str] = None
) -> None:
    """
    Execute complete ETL pipeline: data extraction, transformation, and optional model preparation;
    
    Orchestrates the full data processing pipeline: retrieves raw station data, applies cleaning,
    gap filling (CubicSpline interpolation), outlier removal (ECOD+PCA consensus), time aggregation,
    feature imputation (IterativeImputer), and reshaping. Optionally performs train/test split and
    preprocessing if target_column is provided. Can persist all intermediate results to database;
    
    Parameters:
        save_to_db (bool): If True, save all intermediate dataframes to DuckDB (default: False);
        frequency (str): Pandas frequency string for time aggregation (default: 'h').
            Examples: '15min', 'h', 'D', 'W'. See pandas date offset docs;
        max_fill_steps (int): Max 15-minute intervals to interpolate gaps (default: 96 = 24 hours);
        target_column (Optional[str]): Target column name for train/test split and preprocessing.
            If None, skips data division and encoding (default: None);
    
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

    # Divide data;
    X_train, X_test, y_train, y_test = data_division(
        df=df_melted,
        target_column=target_column,
        test_size=0.2,
        random_state=42
    )

    # Preprocess the data;
    pipeline = encoding_pipeline()
    df_preprocessed = pipeline.fit_transform(df_melted)

    # Train the model; #TODO: Implement training_pipeline function -> Have to check is this is step should be done here (Check with MLFlow as well);
    # model = training_pipeline()
    # model.fit(df_preprocessed)

    # Save the dataframes to the database;
    if save_to_db:
        save_to_database(
            df_cleaned=df_cleaned,
            df_filled=df_filled,
            missing=missing,
            df_out=df_out,
            df_agg=df_agg,
            df_imp=df_imp,
            df_melted=df_melted,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            df_preprocessed=df_preprocessed
        )
    return None

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    main_database(
        save_to_db=True, 
        frequency='h', 
        max_fill_steps=96,  # 96 steps * 15min = 24 hours (1 day);
        target_column=None
    )
    
    print("All Done!")