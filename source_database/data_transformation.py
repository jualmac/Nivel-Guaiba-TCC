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
from typing import Tuple, Dict, Any, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from scipy.interpolate import CubicSpline

from source_database.db_handler import DBConnection
from util import convert_to_float, STATION_COLS, AGG_DICT, START_DATE, END_DATE

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def get_data() -> pd.DataFrame:
    """
    Retrieve raw station data from DuckDB database and merge guaiba stations;
    
    Queries all station tables from the database and concatenates them into a single dataframe.
    Merges guaiba_1 and guaiba_2 stations into a continuous time series, as guaiba_2 was used
    as a backup during the 2024 floods;
    
    Returns:
        pd.DataFrame: Concatenated dataframe containing all station data with merged guaiba stations;
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
    return df


def clean_dataframe(
    df: pd.DataFrame, 
    cut: bool = True
) -> pd.DataFrame:
    """
    Convert values and cut the dataframe to a time range where most data is available;

    Parameters:
        df (pd.DataFrame): The dataframe to cut;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy()

    # Cast'_Status' columns as 'int64';
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    for col in status_cols:
        df_cpy[col] = df_cpy[col].astype('Int64')

    # Create missing temperature status column as integer type;
    df_cpy['Temperatura_Interna_Status'] = pd.Series(dtype='Int64')  # Nullable integer type;
    df_cpy.loc[df_cpy['Temperatura_Interna'].notna(), 'Temperatura_Interna_Status'] = 0
    df_cpy.loc[df_cpy['Temperatura_Interna'].isna(), 'Temperatura_Interna_Status'] = 4

    # Convert the value columns to float (excluding status columns);
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    exclude_cols = {'Data_Atualizacao', 'Data_Hora_Medicao', 'codigoestacao'} | set(status_cols)
    for col in (set(df_cpy.columns) - exclude_cols):
        df_cpy[col] = df_cpy[col].apply(convert_to_float)

    # Convert the date column to datetime;
    df_cpy['Data_Hora_Medicao'] = pd.to_datetime(df_cpy['Data_Hora_Medicao'])
    df_cpy['Data_Atualizacao'] = pd.to_datetime(df_cpy['Data_Atualizacao'])

    # Cut the dataframe to a time range where most data is available;
    if cut:
        df_cpy = df_cpy[df_cpy['Data_Hora_Medicao'] >= START_DATE]
        df_cpy = df_cpy[df_cpy['Data_Hora_Medicao'] <= END_DATE]
    df_cpy = df_cpy.sort_values('Data_Hora_Medicao').reset_index(drop=True)
    
    # Use sensor data to fill the gaps in the level column and respective status;
    was_nan = df_cpy['Cota_Adotada'].isna()
    df_cpy['Cota_Adotada'] = df_cpy['Cota_Adotada'].fillna(df_cpy['Cota_Manual'])
    # df_cpy['Cota_Adotada'] = df_cpy['Cota_Adotada'].fillna(df_cpy['Cota_Sensor']) # This is creating many outliers. Better to remove it;
    
    # Set status to 4 for filled values;
    is_now_filled = was_nan & df_cpy['Cota_Adotada'].notna()
    df_cpy.loc[is_now_filled, 'Cota_Adotada_Status'] = 4
    return df_cpy


def fill_gaps(
    df: pd.DataFrame, 
    max_fill_steps: int = 96
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill data gaps using sensor fallback hierarchy and CubicSpline interpolation for short gaps.
    Creates continuous 15-minute timeline from START_DATE to END_DATE for each station,
    then applies CubicSpline interpolation only to gaps within max_fill_steps threshold.
    
    Status Codes Quality Control Convention: 0=Normal, 1=Suspicious, 2=Bad, 3=Very Bad, 4=Filled/Missing, 5=Outlier flagged;

    Parameters:
        df (pd.DataFrame): Raw station data with temporal gaps;
        max_fill_steps (int): Maximum consecutive gap length to interpolate (15-min intervals);
            Default 96 = fills gaps up to 24 hours (1 day);
            Gaps longer than this are left as NaN;
    
    Returns:
        pd.DataFrame: Gap-filled data with continuous timeline and quality status codes;
    
    Notes:
        CubicSpline interpolation captures smooth temporal patterns in hydrological/meteorological data.
        Requires at least 4 valid data points per column; falls back to linear interpolation otherwise.
        Uses bc_type='natural' to prevent unrealistic boundary oscillations.
        Only interpolates gaps within valid data boundaries (no extrapolation at edges).
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    # Define columns classification;
    index_cols = ['Data_Hora_Medicao', 'Data_Atualizacao', 'codigoestacao']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols

    # Create continuous timeline at 15-minute intervals for each station before filling;
    df_filled_list = []
    missing_values = []

    for station_id in df_cpy['codigoestacao'].unique():
        df_station = df_cpy[df_cpy['codigoestacao'] == station_id].copy()
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

        missing_values.append({
            'station_id': station_id, 
            'period': 'before', 
            'missing_percentage': df_station['Cota_Adotada'].isna().sum() / len(df_station) * 100})

        for col in df_station.columns:
            if col not in non_feature_cols:
                # Track which rows were NaN before filling;
                was_nan = df_station[col].isna().copy()
                
                # Get valid (non-NaN) indices and values;
                valid_mask = df_station[col].notna()
                valid_indices = np.where(valid_mask)[0]
                valid_values = df_station[col].iloc[valid_indices].values
                
                # CubicSpline requires at least 4 points;
                if len(valid_indices) >= 4:
                    # Identify contiguous NaN gap blocks and their lengths (Very overengenieered);
                    nan_array = was_nan.values
                    gap_starts = np.where(~nan_array[:-1] & nan_array[1:])[0] + 1
                    gap_ends = np.where(nan_array[:-1] & ~nan_array[1:])[0] + 1
                    
                    # Handle edge cases: gap at start or end;
                    if nan_array[0]:
                        gap_starts = np.concatenate([[0], gap_starts])
                    if nan_array[-1]:
                        gap_ends = np.concatenate([gap_ends, [len(nan_array)]])
                    
                    # Build CubicSpline interpolation with natural boundary conditions;
                    cs = CubicSpline(valid_indices, valid_values, bc_type='natural')
                    
                    # Constrain interpolation range to avoid extrapolation at boundaries;
                    first_valid_idx = valid_indices[0]
                    last_valid_idx = valid_indices[-1]
                    
                    # Process each gap individually;
                    for gap_start, gap_end in zip(gap_starts, gap_ends):
                        gap_length = gap_end - gap_start
                        
                        # Only interpolate if gap is within threshold and within valid data boundaries;
                        if gap_length <= max_fill_steps and gap_start >= first_valid_idx and gap_end <= last_valid_idx:
                            # Generate indices for this gap;
                            gap_indices = np.arange(gap_start, gap_end)
                            
                            # Interpolate using CubicSpline;
                            interpolated_vals = cs(gap_indices)
                            
                            # Clip to reasonable bounds to prevent extreme overshoot. Use 50% margin beyond observed min/max for safety -> Temperature can be negative;
                            value_min = valid_values.min()
                            value_max = valid_values.max()
                            value_range = value_max - value_min
                            lower_bound = value_min - 0.5 * value_range
                            upper_bound = value_max + 0.5 * value_range

                            # Enforce non-negative for non-temperature features to prevent negative interpolated values;
                            if col != 'Temperatura_Interna':
                                lower_bound = max(0, lower_bound)  # Ensure non-negative for hydrological features;
                            interpolated_vals = np.clip(interpolated_vals, lower_bound, upper_bound)
                            
                            # Apply interpolated values to this gap;
                            df_station.iloc[gap_indices, df_station.columns.get_loc(col)] = interpolated_vals
                    
                    # Round after all interpolation;
                    df_station[col] = df_station[col].round(1)
                
                # Set status to 4 for filled values if status column exists;
                status_col = col + '_Status'
                if status_col in df_station.columns:
                    is_now_filled = was_nan & df_station[col].notna()
                    df_station.loc[is_now_filled, status_col] = 4

        # Reset index to convert back to column for concatenation;
        df_station = df_station.reset_index()
        df_filled_list.append(df_station)

        missing_values.append({
            'station_id': station_id, 
            'period': 'after', 
            'missing_percentage': df_station['Cota_Adotada'].isna().sum() / len(df_station) * 100})
    
    # Convert list of records to DataFrame;
    missing_values = pd.DataFrame(missing_values)
    
    # Concatenate all stations back together;
    df_cpy = pd.concat(df_filled_list, ignore_index=True)

    # Remove negative values as they don't make sense in this dataset, except for Temperature -> These values will be Imputed after;   
    non_feature_cols = non_feature_cols + ['Temperatura_Interna']
    neg_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    for col in neg_cols:
        # Track which values were negative before removal;
        was_negative = df_cpy[col] < 0
        df_cpy.loc[was_negative, col] = np.nan
        # Update status to 2 (Bad) for removed negative values;
        status_col = col + '_Status'
        if status_col in df_cpy.columns:
            df_cpy.loc[was_negative, status_col] = 2  # 2 = Bad;

    # Fill the missing _Status columns;
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    for status_col in status_cols:
        info_col = status_col.replace('_Status', '')

        # Fill the missing values on the '_Status' columns with '4' where corresponding 'info' column is None;
        if info_col in df_cpy.columns:
            df_cpy.loc[df_cpy[info_col].isna(), status_col] = 4
        
        # Fill missing _Status where the _info is not NaN with Normal status;
        if info_col in df_cpy.columns:
            df_cpy.loc[df_cpy[info_col].notna() & df_cpy[status_col].isna(), status_col] = 0

    # Select only the columns that are needed;
    df_cpy = df_cpy[STATION_COLS]
    return df_cpy, missing_values


def aggregate_data(
    df: pd.DataFrame, 
    frequency: str = 'h'
) -> pd.DataFrame:
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
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    # Resample the data to the desired frequency to create continuous timeline and fill gaps;
    df_agg = (
        df_cpy.groupby('codigoestacao', group_keys=True)
          .apply(lambda g: g.resample(frequency, on='Data_Hora_Medicao').agg(AGG_DICT))
          .reset_index()
    )

    # Convert the value columns to float and round to 3 decimal places (excluding status columns);
    status_cols = [col for col in df_agg.columns if col.endswith('_Status')]
    exclude_cols = {'Data_Hora_Medicao', 'codigoestacao'} | set(status_cols)
    for col in (set(df_agg.columns) - exclude_cols):
        df_agg[col] = df_agg[col].apply(convert_to_float)
        df_agg[col] = df_agg[col].round(3)
    return df_agg


def outlier_removal(
    df: pd.DataFrame, 
    threshold_method: str = 'iqr'
) -> pd.DataFrame:
    """
    Detect outliers using ECOD and PCA methods with automatic threshold detection per station.
    Uses statistical thresholds (IQR or percentile) to dynamically determine outliers for each
    station independently, avoiding the need to pre-specify contamination rates.

    Parameters:
        df (pd.DataFrame): The dataframe to detect outliers from;
        threshold_method (str): Method for automatic threshold detection.
            Options:
                - 'iqr' (Interquartile Range): Outliers beyond Q3 + 1.5*IQR (default);
                - 'percentile': Top 5% of decision scores flagged as outliers;
    
    Returns:
        df (pd.DataFrame): Dataframe with outliers replaced by NaN and status codes updated;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    # Prepare data;
    index_cols = ['Data_Hora_Medicao', 'codigoestacao']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols
    feature_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    
    processed_stations = []
    for station in df_cpy['codigoestacao'].unique():
        print(f"\nProcessing station {station}...")
        df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()

        # Identify fully missing columns for this station and exclude them from outlier detection;
        fully_missing = df_station_features.isna().all()
        available_feature_cols = [col for col in feature_cols if not fully_missing[col]]
        fully_missing_cols = [col for col in feature_cols if fully_missing[col]]
        
        if fully_missing_cols:
            print(f"  Skipping fully missing features for outlier detection: {fully_missing_cols}")
        
        # If no features available, skip outlier detection for this station;
        if not available_feature_cols:
            print(f"  Warning: No available features for station {station}. Skipping outlier detection.")
            processed_stations.append(df_station)
            continue
        
        # Only use complete rows for outlier detection (considering only available features);
        df_station_available = df_station_features[available_feature_cols].copy()
        complete_mask = df_station_available.notna().all(axis=1)
        df_complete = df_station_available[complete_mask]
        
        # If no complete rows, skip outlier detection;
        if len(df_complete) == 0:
            print(f"  Warning: No complete rows for station {station}. Skipping outlier detection.")
            processed_stations.append(df_station)
            continue
        
        print(f"  Using {len(available_feature_cols)} features and {len(df_complete)} complete rows for outlier detection")
        
        # ECOD Detection;
        print("[ECOD DETECTOR]")
        # Fit model with minimal contamination to extract decision scores (Actual threshold is determined dynamically via IQR/percentile method below);
        ecod_detector = ECOD(contamination=0.001)
        ecod_detector.fit(df_complete)
        ecod_scores = ecod_detector.decision_scores_
        
        # PCA Detection;
        print("[PCA DETECTOR]")
        # Fit model with minimal contamination to extract decision scores (Actual threshold is determined dynamically via IQR/percentile method below);
        pca_detector = PCA(contamination=0.001)
        pca_detector.fit(df_complete)
        pca_scores = pca_detector.decision_scores_
        
        # Apply dynamic threshold detection based on decision scores;
        if threshold_method == 'iqr':
            # Tukey Fence (IQR method): outliers are beyond Q3 + 1.5*IQR;
            ecod_q1, ecod_q3 = np.percentile(ecod_scores, [25, 75])
            ecod_iqr = ecod_q3 - ecod_q1
            ecod_threshold = ecod_q3 + 1.5 * ecod_iqr
            ecod_predictions = (ecod_scores > ecod_threshold).astype(int)
            
            pca_q1, pca_q3 = np.percentile(pca_scores, [25, 75])
            pca_iqr = pca_q3 - pca_q1
            pca_threshold = pca_q3 + 1.5 * pca_iqr
            pca_predictions = (pca_scores > pca_threshold).astype(int)
            
            print(f"  ECOD: threshold={ecod_threshold:.4f}, outliers={ecod_predictions.sum()}/{len(ecod_predictions)} ({100*ecod_predictions.sum()/len(ecod_predictions):.2f}%)")
            print(f"  PCA: threshold={pca_threshold:.4f}, outliers={pca_predictions.sum()}/{len(pca_predictions)} ({100*pca_predictions.sum()/len(pca_predictions):.2f}%)")
        
        # Fixed ammount of outliers; 
        elif threshold_method == 'percentile':
            # Percentile method: use 95th percentile as threshold (top 5% are outliers);
            ecod_threshold = np.percentile(ecod_scores, 95)
            ecod_predictions = (ecod_scores > ecod_threshold).astype(int)
            
            pca_threshold = np.percentile(pca_scores, 95)
            pca_predictions = (pca_scores > pca_threshold).astype(int)
            
            print(f"  ECOD: threshold={ecod_threshold:.4f} (95th percentile), outliers={ecod_predictions.sum()}/{len(ecod_predictions)} ({100*ecod_predictions.sum()/len(ecod_predictions):.2f}%)")
            print(f"  PCA: threshold={pca_threshold:.4f} (95th percentile), outliers={pca_predictions.sum()}/{len(pca_predictions)} ({100*pca_predictions.sum()/len(pca_predictions):.2f}%)")
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}. Use 'iqr' or 'percentile'")

        # Combined outliers (intersection of both detectors - only flag if both agree -> Imply consensus);
        combined_outliers = (ecod_predictions & pca_predictions).astype(bool)
        print(f"  Consensus: {combined_outliers.sum()}/{len(combined_outliers)} outliers ({100*combined_outliers.sum()/len(combined_outliers):.2f}%)")
        
        # Map outliers back to original dataframe (only for available features);
        complete_indices = df_station_available[complete_mask].index
        outlier_indices = complete_indices[combined_outliers]

        # Replace outliers with NaN (only for available features);
        for feature in available_feature_cols:
            original_count = df_station[feature].isna().sum()
            df_station.loc[outlier_indices, feature] = np.nan
            new_count = df_station[feature].isna().sum()
            print(f"  {feature}: {original_count} → {new_count} missing values")
            
            # Update status to track outlier replacement;
            status_col = feature + '_Status'
            if status_col in df_station.columns:
                df_station.loc[outlier_indices, status_col] = 5  # 5 = Outlier flagged;
        
        processed_stations.append(df_station)
    df_result = pd.concat(processed_stations, ignore_index=True)
    return df_result


def feature_imputation(
    df: pd.DataFrame, 
    n_estimators: int = 50
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply multivariate feature imputation using IterativeImputer with RandomForestRegressor estimator on the large gaps in
    data that where not filled in the function fill_gaps(). Imputes per station to preserve within-station feature
    correlations. Excludes non-numeric columns from imputation;
    
    Parameters:
        df (pd.DataFrame): Dataframe with long gaps to impute;
        n_estimators (int): Number of trees in RandomForest (default: 50);
    
    Returns:
        pd.DataFrame: Imputed dataframe;
        dict: Imputer statistics per station (initial means, feature order, imputation rounds);
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    #TODO: Geographical Imputation for the Guaíba_1 (87450004) and Guaíba_2(87444000) Stations; Searched for Stations on
    # Rio_Codigo IN ('87200000') and did not found any station that is both Tipo_Estacao_Telemetrica IN ('1') and
    # Tipo_Rede_Classe_Vazao IN ('1') at the same time. There is a station very close that has Vazao, '87450005', but it
    # is not Tipo_Estacao_Telemetrica, and therefore the data can't be collected via the API; Also tried stations
    # '87500020' and '87460120'-> Flow data can't be used;
     
    # Separate index/categorical columns from numeric features;
    index_cols = ['Data_Hora_Medicao', 'codigoestacao']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols
    feature_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    
    # Impute per station to preserve within-station correlations;
    imputed_stations = []
    imputer_stats = {}
    
    for station in df_cpy['codigoestacao'].unique():
        print(f"\nImputing station {station}...")
        df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()
        
        # Identify fully missing columns for this station and exclude from imputation (Imputing 100% missing data would create synthetic values with no basis in reality);
        fully_missing = df_station_features.isna().all()
        cols_to_impute = [col for col in feature_cols if not fully_missing[col]]
        fully_missing_cols = [col for col in feature_cols if fully_missing[col]]
        
        if fully_missing_cols:
            print(f"  Skipping fully missing features (cannot impute): {fully_missing_cols}")
        
        if cols_to_impute:
            # RandomForestRegressor with deeper trees for better non-linear relationships (No max_depth restriction allows deep splits; min_samples_leaf=1 for fine-grained predictions);
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=12,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                random_state=42,
                max_iter=10,
                imputation_order='roman',
                verbose=1
            )
            
            # Impute only columns with partial data;
            imputed = imputer.fit_transform(df_station_features[cols_to_impute])
            df_station_imputed = pd.DataFrame(imputed, columns=cols_to_impute, index=df_station_features.index)
            
            # Store imputer statistics for diagnostics;
            imputer_stats[station] = {
                'initial_means': dict(zip(cols_to_impute, imputer.initial_imputer_.statistics_)),
                'imputation_sequence': imputer.imputation_sequence_,
                'n_iter': imputer.n_iter_,
                'cols_imputed': cols_to_impute,
                'cols_skipped': fully_missing_cols
            }
            
            # Keep fully missing columns as NaN (they will be excluded later in melt);
            for col in fully_missing_cols:
                df_station_imputed[col] = df_station_features[col]
        else:
            print(f"  Warning: No features to impute for station {station}")
            df_station_imputed = df_station_features
            imputer_stats[station] = {'error': 'No features to impute'}
        
        df_station_result = pd.concat([df_station_non_features, df_station_imputed], axis=1)
        imputed_stations.append(df_station_result)
    df_result = pd.concat(imputed_stations, ignore_index=True)
    
    # Convert status columns to integers to ensure they remain categorical;
    status_cols = [col for col in df_result.columns if col.endswith('_Status')]
    for status_col in status_cols:
        df_result[status_col] = df_result[status_col].astype('Int64')  # Nullable integer type;
    
    # Round only numeric feature columns;
    feature_cols = [col for col in df_result.columns if not col.endswith('_Status') and col not in ['Data_Hora_Medicao', 'codigoestacao']]
    for col in feature_cols:
        df_result[col] = df_result[col].round(3)
    return df_result, imputer_stats


def melt_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform dataframe from long format (rows per station) to wide format (columns per station-metric).
    Automatically excludes station-metric combinations where all values are missing.

    Parameters:
        df (pd.DataFrame): The dataframe to transform;
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy()
    
    # Get the value columns (excluding date and codigoestacao);
    value_cols = [col for col in df_cpy.columns if col not in ['Data_Hora_Medicao', 'codigoestacao']]
    
    # Melt the dataframe to long format first;
    melted = df_cpy.melt(id_vars=['Data_Hora_Medicao', 'codigoestacao'], 
                     value_vars=value_cols,
                     var_name='metric', 
                     value_name='value')
    
    # Create new column names combining metric and codigoestacao;
    melted['new_col'] = melted['metric'] + '_' + melted['codigoestacao'].astype(str)
    
    # Identify and exclude station-metric combinations that are 100% missing;
    # This automatically removes Vazao_Adotada_87450004 and similar problematic columns;
    missing_by_combination = melted.groupby('new_col')['value'].apply(lambda x: x.isna().all())
    fully_missing_combinations = missing_by_combination[missing_by_combination].index.tolist()
    
    if fully_missing_combinations:
        print(f"\nExcluding {len(fully_missing_combinations)} fully missing station-metric combinations:")
        for combo in fully_missing_combinations:
            print(f"  - {combo}")
        melted = melted[~melted['new_col'].isin(fully_missing_combinations)]
    
    # Check for duplicates in Data_Hora_Medicao + codigoestacao + metric combinations;
    duplicate_mask = melted.duplicated(subset=['Data_Hora_Medicao', 'codigoestacao', 'metric'], keep=False)
    duplicates_df = melted[duplicate_mask].sort_values(['Data_Hora_Medicao', 'codigoestacao', 'metric'])
    
    if len(duplicates_df) > 0:
        print(f"Found {len(duplicates_df)} duplicate records (Data_Hora_Medicao + codigoestacao + metric combinations):")
        print(f"Number of unique duplicate combinations: {len(duplicates_df.drop_duplicates(subset=['Data_Hora_Medicao', 'codigoestacao', 'metric']))}")
        print("\nFirst 20 duplicate records:")
        print(duplicates_df.head(20))
        print("\nDuplicate summary by combination:")
        duplicate_counts = melted.groupby(['Data_Hora_Medicao', 'codigoestacao', 'metric']).size()
        print(duplicate_counts[duplicate_counts > 1].head(10))
    else:
        print("No duplicates found in Data_Hora_Medicao + codigoestacao + metric combinations")
    
    # Pivot to wide format using pivot_table to handle duplicates;
    df_pivoted = melted.pivot_table(index='Data_Hora_Medicao', columns='new_col', values='value', aggfunc='first')
    
    # Reset index and return pivoted dataframe;
    df_pivoted = df_pivoted.reset_index()
    return df_pivoted

    
def save_to_database(
    df_cleaned: Optional[pd.DataFrame] = None,
    df_filled: Optional[pd.DataFrame] = None,
    missing: Optional[pd.DataFrame] = None,
    df_out: Optional[pd.DataFrame] = None,
    df_agg: Optional[pd.DataFrame] = None,
    df_imp: Optional[pd.DataFrame] = None,
    df_melted: Optional[pd.DataFrame] = None,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
    df_preprocessed: Optional[pd.DataFrame] = None
) -> None:
    """
    Save all processed dataframes to DuckDB database;
    
    Writes each dataframe to its corresponding table in the database. All writes are performed
    inplace, replacing any existing data in the target tables. Each dataframe is saved only if provided;
    
    Parameters:
        df_cleaned (Optional[pd.DataFrame]): Cleaned dataframe to save (default: None);
        df_filled (Optional[pd.DataFrame]): Gap-filled dataframe to save (default: None);
        missing (Optional[pd.DataFrame]): Missing values tracking dataframe to save (default: None);
        df_out (Optional[pd.DataFrame]): Outlier-removed dataframe to save (default: None);
        df_agg (Optional[pd.DataFrame]): Aggregated dataframe to save (default: None);
        df_imp (Optional[pd.DataFrame]): Imputed dataframe to save (default: None);
        df_melted (Optional[pd.DataFrame]): Melted/transformed dataframe to save (default: None);
        X_train (Optional[pd.DataFrame]): Training features to save (default: None);
        X_test (Optional[pd.DataFrame]): Test features to save (default: None);
        y_train (Optional[pd.Series]): Training target to save (default: None);
        y_test (Optional[pd.Series]): Test target to save (default: None);
        df_preprocessed (Optional[pd.DataFrame]): Preprocessed dataframe to save (default: None);
    """
    # Initialize the Connection;
    db = DBConnection()

    # Save main pipeline dataframes only if provided;
    if df_cleaned is not None:
        db.write(df=df_cleaned, table_name='data_stations_cleaned', inplace=True)
    if df_filled is not None:
        db.write(df=df_filled, table_name='data_stations_filled', inplace=True)
    if missing is not None:
        db.write(df=missing, table_name='data_stations_missing', inplace=True)
    if df_out is not None:
        db.write(df=df_out, table_name='data_stations_outlier', inplace=True)
    if df_agg is not None:
        db.write(df=df_agg, table_name='data_stations_aggregated', inplace=True)
    if df_imp is not None:
        db.write(df=df_imp, table_name='data_stations_imputed', inplace=True)
    if df_melted is not None:
        db.write(df=df_melted, table_name='data_stations_melted', inplace=True)
    if X_train is not None:
        db.write(df=X_train, table_name='data_train_features', inplace=True)
    if X_test is not None:
        db.write(df=X_test, table_name='data_test_features', inplace=True)
    if y_train is not None:
        # Convert Series to DataFrame for database write;
        y_train_df = pd.DataFrame(y_train)
        db.write(df=y_train_df, table_name='data_train_target', inplace=True)
    if y_test is not None:
        # Convert Series to DataFrame for database write;
        y_test_df = pd.DataFrame(y_test)
        db.write(df=y_test_df, table_name='data_test_target', inplace=True)
    if df_preprocessed is not None:
        db.write(df=df_preprocessed, table_name='data_preprocessed', inplace=True)