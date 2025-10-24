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
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA as sklearn_PCA

from source_database.db_handler import DBConnection
from util import convert_to_float, STATIONS_COLS, AGG_DICT, START_DATE, END_DATE

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def collect_all_stations(save_to_db: bool = False, frequency: str = 'h', max_fill_steps: int = 8):
    """
    Retrieve the data from the stations and return a single dataframe concatenated;

    Parameters:
        save_to_db (bool): Whether to save the dataframe to the database;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'h' for hourly).
            Examples: 'min' (minutes), 'h' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...
        max_fill_steps (int): The maximum number of steps to forward-fill the data;

    Returns:
        df (pd.DataFrame): The concatenated dataframe;
        df_cleaned (pd.DataFrame): The cleaned and aggregated dataframe;
    """
    # Initialize the Connection;
    db = DBConnection()

    # Query data;
    query = {
        'cai_1':        'SELECT * FROM station_cai_1',
        'cai_2':        'SELECT * FROM station_cai_2',
        'gravatai_1':   'SELECT * FROM station_gravatai_1',
        'guaiba_1':     'SELECT * FROM station_guaiba_1',
        'guaiba_2':     'SELECT * FROM station_guaiba_2',
        'jacui_1':      'SELECT * FROM station_jacui_1',
        'sinos_1':      'SELECT * FROM station_sinos_1',
        'sinos_2':      'SELECT * FROM station_sinos_2',
        'taquari_1':    'SELECT * FROM station_taquari_1',
        'taquari_2':    'SELECT * FROM station_taquari_2',
        }
    dataframe = db.run(query=query)

    # Merge guaiba_1 and guaiba_2 into a single dataframe -> This is because the guaiba_2 station was setted as a backup to guaiba_1 during the 2024 floods. Therefore, their data should be considered as a continuous time series;
    guaiba_merged = pd.concat([dataframe['guaiba_1'], dataframe['guaiba_2']])
    guaiba_merged['codigoestacao'] = '87450004'
    
    # Update the dictionary with the merged guaiba and remove guaiba_2;
    dataframe['guaiba_1'] = guaiba_merged
    dataframe.pop('guaiba_2')
    
    # Concatenate all station dataframes;
    df = pd.concat(list(dataframe.values()))

    # Convert values and cut the dataframe to a time range where most data is available;
    df_cleaned = clean_dataframe(df=df)

    # Fill the data gaps;
    df_filled = fill_gaps(df=df_cleaned, max_fill_steps=max_fill_steps)

    # Aggregate the data to the desired frequency;
    df_agg = aggregate_data(df=df_filled, frequency=frequency)

    # Identify and remove Outliers;
    df_out = outlier_removal(df=df_agg)

    # Feature Imputation - IteractiveImputer;
    df_imp = feature_imputation(df=df_out)

    # Melt the dataframe;
    df_melted = melt_dataframe(df=df_imp)

    # Save the dataframes to the database;
    if save_to_db:
        db.write(df=df_cleaned, table_name='data_stations_cleaned', inplace=True)
        db.write(df=df_filled, table_name='data_stations_filled', inplace=True)
        db.write(df=df_agg, table_name='data_stations_aggregated', inplace=True)
        db.write(df=df_out, table_name='data_stations_outlier', inplace=True)
        db.write(df=df_imp, table_name='data_stations_imputed', inplace=True)
        db.write(df=df_melted, table_name='data_stations_melted', inplace=True)
    return df_cleaned, df_filled, df_agg, df_out, df_imp, df_melted


def clean_dataframe(df: pd.DataFrame):
    """
    Convert values and cut the dataframe to a time range where most data is available;

    Parameters:
        df (pd.DataFrame): The dataframe to cut;
    """
    # Convert the value columns to float;
    for col in (set(df.columns) - {'Data_Atualizacao', 'Data_Hora_Medicao', 'codigoestacao'}):
        df[col] = df[col].apply(convert_to_float)

    # Convert the date column to datetime;
    df['Data_Hora_Medicao'] = pd.to_datetime(df['Data_Hora_Medicao'])
    df['Data_Atualizacao'] = pd.to_datetime(df['Data_Atualizacao'])

    # Cut the dataframe to a time range where most data is available;
    df = df[df['Data_Hora_Medicao'] >= START_DATE]
    df = df[df['Data_Hora_Medicao'] <= END_DATE]
    df = df.sort_values('Data_Hora_Medicao').reset_index(drop=True)
    df = df.set_index('Data_Hora_Medicao')

    # Create missing temperature status column;
    df.loc[df['Temperatura_Interna'].notna(), 'Temperatura_Interna_Status'] = float(0.0)
    df.loc[df['Temperatura_Interna'].isna(), 'Temperatura_Interna_Status'] = float(5.0)
    return df


def fill_gaps(df: pd.DataFrame, max_fill_steps: int = 8):
    """
    Fill data gaps using sensor fallback hierarchy and forward-fill interpolation.
    Creates continuous 15-minute timeline from START_DATE to END_DATE for each station,
    then applies forward-fill with quality control status tracking.
    
    Status Codes Quality Control Convention: 0=Normal, 1=Suspicious, 2=Bad, 3=Very Bad, 4=Filled, 5=Missing;

    Parameters:
        df (pd.DataFrame): Raw station data with temporal gaps;
        max_fill_steps (int): Maximum consecutive forward-fill steps at 15-min intervals;
            Default 8 = fills gaps up to 2 hours;
    
    Returns:
        pd.DataFrame: Gap-filled data with continuous timeline and quality status codes;
    """
    # Remove negative values as they don't make sense in this dataset, except for Temperature -> These values will be Imputed after;
    neg_cols = list(set(df.columns.unique()) - set(['Data_Hora_Medicao', 'Data_Atualizacao', 'codigoestacao', 'Temperatura_Interna']))
    for col in neg_cols:
        df.loc[df[col] < 0, col] = np.nan

    # Use sensor data to fill the gaps in the level column and respective status;
    was_nan = df['Cota_Adotada'].isna()
    df['Cota_Adotada'] = df['Cota_Adotada'].fillna(df['Cota_Manual'])
    df['Cota_Adotada'] = df['Cota_Adotada'].fillna(df['Cota_Sensor'])
    
    # Set status to 4 for filled values;
    is_now_filled = was_nan & df['Cota_Adotada'].notna()
    df.loc[is_now_filled, 'Cota_Adotada_Status'] = float(4.0)

    # Reset index to work with Data_Hora_Medicao as a column for merging;
    df = df.reset_index()
    
    # Create continuous timeline at 15-minute intervals for each station before filling;
    df_filled_list = []
    for station_id in df['codigoestacao'].unique():
        df_station = df[df['codigoestacao'] == station_id].copy()
        df_station = df_station.sort_values('Data_Hora_Medicao').reset_index(drop=True)

        # Create complete date range at 15-minute intervals;
        date_range = pd.date_range(
            start=pd.to_datetime(START_DATE),
            end=pd.to_datetime(END_DATE),
            freq='15min' # Needs to use the natural frequency of the source data;
        )
        
        # Create a DataFrame with the complete date range;
        df_complete = pd.DataFrame({'Data_Hora_Medicao': date_range})
        
        # Merge with existing data to create NaN rows for missing timestamps;
        df_station = df_complete.merge(df_station, on='Data_Hora_Medicao', how='left')
        
        # Fill station_id for the newly created rows;
        df_station['codigoestacao'] = df_station['codigoestacao'].fillna(station_id)
        
        # Time-based interpolation of small gaps;
        df_station = df_station.set_index('Data_Hora_Medicao')

        for col in df_station.columns:
            if col not in ['codigoestacao'] and not col.endswith('_Status'):
                # Track which rows were NaN before filling;
                was_nan = df_station[col].isna()
                
                # Fill small gaps with Linear Interpolation (Sufficient for 15 minutes data frequency, as changes shouldn't be that fast);
                df_station[col] = df_station[col].interpolate(method='linear', limit=max_fill_steps).round(1)
                
                # Set status to 4 for filled values if status column exists;
                status_col = col + '_Status'
                if status_col in df_station.columns:
                    is_now_filled = was_nan & df_station[col].notna()
                    df_station.loc[is_now_filled, status_col] = float(4.0)

        # Reset index to convert back to column for concatenation;
        df_station = df_station.reset_index()
        
        nan_percentage = df_station['Cota_Adotada'].isna().sum() / len(df_station) * 100
        print(f"Percentage of Missing Level data in station {station_id}: {nan_percentage:.2f}%")
        df_filled_list.append(df_station)
    
    # Concatenate all stations back together;
    df = pd.concat(df_filled_list, ignore_index=True)

    # Fill the missing _Status columns;
    status_cols = [col for col in df.columns if col.endswith('_Status')]
    for status_col in status_cols:
        info_col = status_col.replace('_Status', '')

        # Fill the missing values on the '_Status' columns with '5' where corresponding 'info' column is None;
        if info_col in df.columns:
            df.loc[df[info_col].isna(), status_col] = float(5.0)
        
        # Fill missing _Status where the _info is not NaN with Normal status;
        if info_col in df.columns:
            df.loc[df[info_col].notna() & df[status_col].isna(), status_col] = float(0.0)

    # Select only the columns that are needed;
    df = df[STATIONS_COLS.keys()]
    df.rename(columns=STATIONS_COLS, inplace=True)
    return df


def aggregate_data(df: pd.DataFrame, frequency: str = 'h'):
    """
    Aggregate the data to the desired frequency;

    Parameters:
        df (pd.DataFrame): The dataframe to aggregate;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'h' for hourly).
            Examples: 'min' (minutes), 'h' (hours), 'D' (days), 'W' (weeks), ...
            Available options: 'https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects'
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...

    Returns:
        df_agg (pd.DataFrame): The aggregated dataframe;
    """
    # Resample the data to the desired frequency to create continuous timeline and fill gaps;
    df_agg = (
        df.groupby('station_id', group_keys=True)
          .apply(lambda g: g.resample(frequency, on='date').agg(AGG_DICT))
          .reset_index()
    )

    # Convert the value columns to float and round to 3 decimal places;
    for col in (set(df_agg.columns) - {'date', 'station_id'}):
        df_agg[col] = df_agg[col].apply(convert_to_float)
        df_agg[col] = df_agg[col].round(3)
    return df_agg


def outlier_removal(df: pd.DataFrame, contamination: float = 0.03):
    """
    Detect outliers using ECOD and PCA methods and visualize results;

    Parameters:
        df (pd.DataFrame): The dataframe to detect outliers from;
    
    Returns:
        df (pd.DataFrame): Original dataframe (outlier removal can be implemented later);
    """
    # Prepare data;
    index_cols = ['date', 'station_id']
    status_cols = [col for col in df.columns if col.endswith('_status')]
    non_feature_cols = index_cols + status_cols
    feature_cols = list(set(df.columns.unique()) - set(non_feature_cols)) #FIXME .unique()?
    
    processed_stations = []
    for station in df['station_id'].unique():
        print(f"\nProcessing station {station}...")
        df_station = df[df['station_id'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()

        # Only use complete rows for outlier detection;
        complete_mask = df_station_features.notna().all(axis=1)
        df_complete = df_station_features[complete_mask]
        
        # ECOD Detection;
        print("[ECOD DETECTOR]")
        ecod_detector = ECOD(contamination=contamination)
        ecod_detector.fit(df_complete)
        ecod_predictions = ecod_detector.predict(df_complete)
        ecod_scores = ecod_detector.decision_scores_
        
        # PCA Detection;
        print("[PCA DETECTOR]")
        pca_detector = PCA(contamination=contamination)
        pca_detector.fit(df_complete)
        pca_predictions = pca_detector.predict(df_complete)
        pca_scores = pca_detector.decision_scores_

        # Combined outliers;
        combined_outliers = (ecod_predictions | pca_predictions).astype(bool)
        
        # Map outliers back to original dataframe;
        complete_indices = df_station_features[complete_mask].index
        outlier_indices = complete_indices[combined_outliers]

        # Replace outliers with NaN;
        for feature in feature_cols:
            original_count = df_station[feature].isna().sum()
            df_station.loc[outlier_indices, feature] = np.nan
            new_count = df_station[feature].isna().sum()
            print(f"  {feature}: {original_count} → {new_count} missing values")
            
            # Update status to track outlier replacement;
            status_col = feature + '_status'
            if status_col in df_station.columns:
                df_station.loc[outlier_indices, status_col] = 6.0  # 6 = Outlier flagged;
        processed_stations.append(df_station)
    df_result = pd.concat(processed_stations, ignore_index=True)
    return df_result


def feature_imputation(df: pd.DataFrame): #TODO: Do a bigger check of the missing values before and after per station;
    """
    Apply multivariate feature imputation using IterativeImputer with ExtraTreesRegressor estimator on the large gaps in
    data that where not filled in the function fill_gaps(). Imputes per station to preserve within-station feature
    correlations. Exclues non-numeric columns from imputation;
    
    Parameters:
        df (pd.DataFrame): Dataframe with longs gaps to impute;
    
    Returns:
        pd.DataFrame: Imputed dataframe;
    """
    #TODO: Geographical Imputation for the Guaíba_1 (87450004) and Guaíba_2(87444000) Stations; Searched for Stations on
    # Rio_Codigo IN ('87200000') and did not found any station that is both Tipo_Estacao_Telemetrica IN ('1') and
    # Tipo_Rede_Classe_Vazao IN ('1') at the same time. There is a station very close that has Vazao, '87450005', but it
    # is not Tipo_Estacao_Telemetrica, and therefore the data can't be collected via the API; Also tried stations
    # '87500020' and '87460120'-> Flow data can't be used;
     
    # Separate index/categorical columns from numeric features;
    index_cols = ['date', 'station_id']
    status_cols = [col for col in df.columns if col.endswith('_status')]
    non_feature_cols = index_cols + status_cols
    feature_cols = list(set(df.columns.unique()) - set(non_feature_cols)) #FIXME .unique()?
    
    # Impute per station to preserve within-station correlations;
    imputed_stations = []
    for station in df['station_id'].unique():
        print(f"\nImputing station {station}...")
        df_station = df[df['station_id'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()
        
        # ExtraTreesRegressor for better non-linear relationships, robustness to outliers and different scales between feats;
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
            random_state=42,
            max_iter=10,
            imputation_order='ascending',
            verbose=1
        )
        
        # Impute numeric feature columns for this station;
        imputed = imputer.fit_transform(df_station_features)
        df_station_imputed = pd.DataFrame(imputed, columns=feature_cols, index=df_station_features.index)
        df_station_result = pd.concat([df_station_non_features, df_station_imputed], axis=1)
        imputed_stations.append(df_station_result)
    df_result = pd.concat(imputed_stations, ignore_index=True)
    df_result = round(df_result, 3)
    return df_result


def melt_dataframe(df: pd.DataFrame):
    """
    Melt the dataframe to long format;

    Parameters:
        df (pd.DataFrame): The dataframe to melt;
    """
    # Get the value columns (excluding date and station_id);
    value_cols = [col for col in df.columns if col not in ['date', 'station_id']]
    
    # Melt the dataframe to long format;
    melted = df.melt(id_vars=['date', 'station_id'], 
                     value_vars=value_cols,
                     var_name='metric', 
                     value_name='value')
    
    # Create new column names combining metric and station_id;
    melted['new_col'] = melted['metric'] + '_' + melted['station_id'].astype(str)
    
    # Check for duplicates in date + station_id + metric combinations;
    duplicate_mask = melted.duplicated(subset=['date', 'station_id', 'metric'], keep=False)
    duplicates_df = melted[duplicate_mask].sort_values(['date', 'station_id', 'metric'])
    
    if len(duplicates_df) > 0:
        print(f"Found {len(duplicates_df)} duplicate records (date + station_id + metric combinations):")
        print(f"Number of unique duplicate combinations: {len(duplicates_df.drop_duplicates(subset=['date', 'station_id', 'metric']))}")
        print("\nFirst 20 duplicate records:")
        print(duplicates_df.head(20))
        print("\nDuplicate summary by combination:")
        duplicate_counts = melted.groupby(['date', 'station_id', 'metric']).size()
        print(duplicate_counts[duplicate_counts > 1].head(10))
    else:
        print("No duplicates found in date + station_id + metric combinations")
    
    # Pivot to wide format using pivot_table to handle duplicates;
    df_pivoted = melted.pivot_table(index='date', columns='new_col', values='value', aggfunc='first')
    
    # Reset index and return pivoted dataframe;
    df_cleaned = df_pivoted.reset_index()
    return df_cleaned

########################################################################################################################
#
# SCRIPT EXECUTION
#
########################################################################################################################
if __name__ == "__main__":
    df_cleaned, df_agg, df_melted = collect_all_stations(save_to_db=True, frequency='h', max_fill_steps=8)
    print("All Done!")