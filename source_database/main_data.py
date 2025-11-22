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
    max_fill_steps: int = 96,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Optional[Pipeline], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """
    Execute the complete data transformation pipeline from raw station data to final processed dataset;
    
    Retrieves data from database, applies cleaning, gap filling, outlier removal, aggregation,
    imputation, and melting transformations. Optionally saves intermediate results to database;

    Parameters:
        save_to_db (bool): Whether to save all intermediate dataframes to the database (default: False);
        frequency (str): Pandas frequency offset string for data aggregation (default: 'h' for hourly).
            Examples: 'min' (minutes), 'h' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...
        max_fill_steps (int): Maximum number of 15-minute steps to interpolate gaps (default: 96);
            Default 96 = fills gaps up to 24 hours (96 * 15min = 1440min);
        target_column (Optional[str]): Target column name for model training. If None, data division is skipped (default: None);

    Returns:
        tuple: A tuple containing:
            - df_cleaned (pd.DataFrame): The cleaned dataframe;
            - df_filled (pd.DataFrame): The gap-filled dataframe;
            - missing (pd.DataFrame): Missing values tracking dataframe;
            - df_out (pd.DataFrame): The outlier-removed dataframe;
            - df_agg (pd.DataFrame): The aggregated dataframe;
            - df_imp (pd.DataFrame): The imputed dataframe;
            - df_melted (pd.DataFrame): The melted/transformed dataframe;
            - imputer_stats (dict): Imputer statistics per station;
            - pipeline (Optional[Pipeline]): Preprocessing pipeline if encoding was applied;
            - df_preprocessed (Optional[pd.DataFrame]): Preprocessed dataframe if encoding was applied;
            - X_train (Optional[pd.DataFrame]): Training features if target_column was provided;
            - X_test (Optional[pd.DataFrame]): Test features if target_column was provided;
            - y_train (Optional[pd.Series]): Training target if target_column was provided;
            - y_test (Optional[pd.Series]): Test target if target_column was provided;
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