"""
Feature imputation operations;

Handles missing value imputation using station-wise CubicSpline interpolation;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# IMPUTATION FUNCTIONS
#
########################################################################################################################
def feature_imputation(
    df: pd.DataFrame,
    max_gap_steps: int = 96
) -> pd.DataFrame:
    """
    Impute missing values using CubicSpline interpolation per station;
    
    Applies station-wise temporal interpolation with natural boundary conditions, mirroring the
    fill_gaps approach. Uses only observed (valid) values to build the spline and fills gaps left
    by outlier_detection. Skips columns with fewer than 4 valid points or entirely missing values.
    Excludes status/meta columns from interpolation;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with missing values to impute;
        max_gap_steps (int): Maximum consecutive rows to interpolate per gap (default: 96);
    
    Returns:
        pd.DataFrame: Imputed dataframe with missing values filled;
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
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols + meta_cols
    feature_cols = list(set(df_cpy.columns.unique()) - set(non_feature_cols))
    
    # Impute per station to preserve within-station correlations;
    imputed_stations = []
    
    for station in df_cpy['codigoestacao'].unique():
        logger.info("Interpolating station %s with CubicSpline...", station)
        df_station = df_cpy[df_cpy['codigoestacao'] == station].copy()
        df_station = df_station.sort_values('Data_Hora_Medicao').reset_index(drop=True)
        df_station_non_features = df_station[non_feature_cols].copy()
        df_station_features = df_station[feature_cols].copy()
        
        # Identify fully missing columns for this station and exclude from interpolation (Interpolating 100% missing data would create synthetic values with no basis in reality);
        fully_missing = df_station_features.isna().all()
        cols_to_impute = [col for col in feature_cols if not fully_missing[col]]
        fully_missing_cols = [col for col in feature_cols if fully_missing[col]]
        
        if fully_missing_cols:
            logger.warning("Skipping fully missing features (cannot interpolate): %s", fully_missing_cols)
        
        if cols_to_impute:
            for col in cols_to_impute:
                # Track NaN positions before interpolation (for status updates);
                was_nan = df_station_features[col].isna()
                
                # Gather valid points for spline fit; only non-NaN values participate;
                valid_mask = df_station_features[col].notna()
                valid_indices = np.where(valid_mask)[0]
                valid_values = df_station_features[col].iloc[valid_indices].values
                
                # CubicSpline needs at least 4 points to operate reliably;
                if len(valid_indices) < 4:
                    continue
                
                # Identify contiguous NaN gap blocks and their lengths;
                nan_array = was_nan.values
                gap_starts = np.where(~nan_array[:-1] & nan_array[1:])[0] + 1
                gap_ends = np.where(nan_array[:-1] & ~nan_array[1:])[0] + 1
                
                # Handle edge gaps; ensures start/end gaps are captured;
                if nan_array[0]:
                    gap_starts = np.concatenate([[0], gap_starts])
                if nan_array[-1]:
                    gap_ends = np.concatenate([gap_ends, [len(nan_array)]])
                
                # Build spline with natural boundary conditions using only valid observations;
                cs = CubicSpline(valid_indices, valid_values, bc_type='natural')
                first_valid_idx = valid_indices[0]
                last_valid_idx = valid_indices[-1]
                
                filled_points = 0
                for gap_start, gap_end in zip(gap_starts, gap_ends):
                    gap_length = gap_end - gap_start
                    
                    # Only interpolate gaps bounded by observed data and within max_gap_steps;
                    if gap_length > 0 and gap_length <= max_gap_steps and gap_start >= first_valid_idx and gap_end <= last_valid_idx:
                        gap_indices = np.arange(gap_start, gap_end)
                        interpolated_vals = cs(gap_indices)
                        
                        # Clip to ±50% beyond observed range; keep hydrological features non-negative;
                        value_min = valid_values.min()
                        value_max = valid_values.max()
                        value_range = value_max - value_min
                        lower_bound = value_min - 0.5 * value_range
                        upper_bound = value_max + 0.5 * value_range
                        if col != 'Temperatura_Interna':
                            lower_bound = max(0, lower_bound)
                        interpolated_vals = np.clip(interpolated_vals, lower_bound, upper_bound)
                        
                        # Apply interpolated values; keep order by using positional index;
                        df_station_features.iloc[gap_indices, df_station_features.columns.get_loc(col)] = interpolated_vals
                        filled_points += gap_length
                
                if filled_points > 0:
                    logger.info("Station %s - %s: filled %s points via CubicSpline", station, col, filled_points)
                
                # Mark status columns with code 4 (Filled/Missing) for interpolated values;
                status_col = col + '_Status'
                if status_col in df_station_non_features.columns:
                    is_now_filled = was_nan & df_station_features[col].notna()
                    df_station_non_features.loc[is_now_filled, status_col] = 4
            
        else:
            logger.warning("No features to interpolate for station %s", station)

        # Final safeguard: fill remaining NaN with station-wise mean to avoid pipeline breaks;
        for col in feature_cols:
            col_mean = df_station_features[col].mean(skipna=True)
            if pd.isna(col_mean):
                continue  # Skip if still all NaN;
            still_nan = df_station_features[col].isna()
            if still_nan.any():
                df_station_features.loc[still_nan, col] = col_mean
                logger.info("Station %s - %s: filled %s points with column mean", station, col, int(still_nan.sum()))
                status_col = col + '_Status'
                if status_col in df_station_non_features.columns:
                    df_station_non_features.loc[still_nan, status_col] = 4
        
        df_station_result = pd.concat([df_station_non_features, df_station_features], axis=1)
        imputed_stations.append(df_station_result)
        logger.info("Station %s interpolation done.", station)
    df_result = pd.concat(imputed_stations, ignore_index=True)
    
    # Convert status columns to integers to ensure they remain categorical;
    status_cols = [col for col in df_result.columns if col.endswith('_Status')]
    for status_col in status_cols:
        df_result[status_col] = df_result[status_col].astype('Int64')  # Nullable integer type;
    
    # Round only numeric feature columns;
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    feature_cols = [col for col in df_result.columns if not col.endswith('_Status') and col not in ['Data_Hora_Medicao', 'codigoestacao'] + meta_cols]
    for col in feature_cols:
        df_result[col] = df_result[col].round(3)
    return df_result