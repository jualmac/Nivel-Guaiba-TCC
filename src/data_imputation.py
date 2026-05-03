"""
Feature imputation operations;

Handles missing value imputation using CubicSpline interpolation for wide-format data;
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
    Impute missing values using CubicSpline interpolation on wide-format data;
    
    Applies temporal interpolation with natural boundary conditions. Uses only observed
    (valid) values to build the spline. Skips columns with fewer than 4 valid points
    or entirely missing values. Does NOT use global mean imputation to prevent data leakage.
    Excludes categorical/status columns from interpolation;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with missing values to impute;
        max_gap_steps (int): Maximum consecutive rows to interpolate per gap (default: 96);
    
    Returns:
        pd.DataFrame: Imputed dataframe with missing values filled;
    """
    df_result = df.copy() 
     
    # Determine columns to interpolate (numeric features)
    status_cols = [col for col in df_result.columns if '_Status' in col]
    exclude_prefixes = ('Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo', 'Data_Hora_Medicao', 'date')
    feature_cols = [
        col for col in df_result.columns 
        if col not in status_cols and not col.startswith(exclude_prefixes) and col != 'target' and col != 'model_name' and col != 'split'
    ]
    
    for col in feature_cols:
        # Check if fully missing
        if df_result[col].isna().all():
            logger.warning("Skipping fully missing feature (cannot interpolate): %s", col)
            continue
            
        was_nan = df_result[col].isna()
        
        valid_mask = df_result[col].notna()
        valid_indices = np.where(valid_mask)[0]
        valid_values = df_result[col].iloc[valid_indices].values
        
        if len(valid_indices) < 4:
            continue
        
        nan_array = was_nan.values
        gap_starts = np.where(~nan_array[:-1] & nan_array[1:])[0] + 1
        gap_ends = np.where(nan_array[:-1] & ~nan_array[1:])[0] + 1
        
        if nan_array[0]:
            gap_starts = np.concatenate([[0], gap_starts])
        if nan_array[-1]:
            gap_ends = np.concatenate([gap_ends, [len(nan_array)]])
        
        cs = CubicSpline(valid_indices, valid_values, bc_type='natural')
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]
        
        filled_points = 0
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            gap_length = gap_end - gap_start
            
            if 0 < gap_length <= max_gap_steps and gap_start >= first_valid_idx and gap_end <= last_valid_idx:
                gap_indices = np.arange(gap_start, gap_end)
                interpolated_vals = cs(gap_indices)
                
                value_min = valid_values.min()
                value_max = valid_values.max()
                value_range = value_max - value_min
                lower_bound = value_min - 0.5 * value_range
                upper_bound = value_max + 0.5 * value_range
                if 'Temperatura' not in col:
                    lower_bound = max(0, lower_bound)
                interpolated_vals = np.clip(interpolated_vals, lower_bound, upper_bound)
                
                df_result.iloc[gap_indices, df_result.columns.get_loc(col)] = interpolated_vals
                filled_points += gap_length
        
        if filled_points > 0:
            logger.info("Column %s: filled %s points via CubicSpline", col, filled_points)
        
        # Mark status columns with code 4 (Filled/Missing) for interpolated values;
        parts = col.split('_')
        if len(parts) >= 2:
            station_code = parts[-1]
            base_metric = "_".join(parts[:-1])
            status_col = f"{base_metric}_Status_{station_code}"
            if status_col in df_result.columns:
                is_now_filled = was_nan & df_result[col].notna()
                df_result.loc[is_now_filled, status_col] = 4

    # Round only numeric feature columns
    for col in feature_cols:
        try:
            df_result[col] = pd.to_numeric(df_result[col]).round(3)
        except:
            pass
        
    return df_result