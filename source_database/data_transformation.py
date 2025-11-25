#TODO: Add categorical features from the stations column. Could be done right before melting the dataframe. Probably just a column with that data, like River Name, LAT, LON, Area_Drenagem (BEM IMPORTANTE, É quanta área acaba escoando para a bacia) -> Encode based on the cardinality if needed;

"""
Time series transformation operations;

Handles temporal operations including gap filling with interpolation and time series aggregation;
"""

########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.interpolate import CubicSpline

from util import convert_to_float, STATION_COLS, AGG_DICT, START_DATE, END_DATE
from db_handler import DBConnection

########################################################################################################################
#                                                                  
# TRANSFORMATION FUNCTIONS
#
########################################################################################################################
def fill_gaps(
    df: pd.DataFrame, 
    max_fill_steps: int = 96
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill temporal gaps using CubicSpline interpolation with bounded constraints;
    
    Creates continuous 15-minute timeline per station from START_DATE to END_DATE. Interpolates gaps
    up to max_fill_steps using CubicSpline with natural boundary conditions. Clips interpolated values
    to ±50% of observed range (non-negative for hydrological features). Updates status codes to 4 for
    filled values. Returns missing value statistics before/after filling;
    
    Status Codes: 0=Normal, 1=Suspicious, 2=Bad, 3=Very Bad, 4=Filled/Missing, 5=Outlier flagged;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with temporal gaps and station codes;
        max_fill_steps (int): Maximum consecutive 15-minute intervals to interpolate (default: 96 = 24h);
            Gaps exceeding this threshold remain as NaN;
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
            - pd.DataFrame: Gap-filled dataframe with continuous 15-minute timeline;
            - pd.DataFrame: Missing value statistics (station_id, period, missing_percentage);
    
    """
    # Copy dataframe to not propagate changes;
    df_cpy = df.copy() 

    # Define columns classification;
    index_cols = ['Data_Hora_Medicao', 'Data_Atualizacao', 'codigoestacao']
    meta_cols = ['Altitude', 'Area_Drenagem', 'Latitude', 'Longitude', 'Rio_Codigo']
    status_cols = [col for col in df_cpy.columns if col.endswith('_Status')]
    non_feature_cols = index_cols + status_cols + meta_cols

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

    # Add station metadata;
    stations = df_cpy['codigoestacao'].dropna().unique()
    if len(stations) > 0:
        print("Loading station metadata from database...")
        db = DBConnection()
        placeholders = ', '.join(['?'] * len(stations))
        query = f"""
            SELECT 
                codigoestacao,
                Altitude,
                Area_Drenagem,
                Latitude,
                Longitude,
                Rio_Codigo
            FROM stations
            WHERE codigoestacao IN ({placeholders})
        """
        station_meta = db.run(query, params=tuple(stations)).get('result', pd.DataFrame())
        if not station_meta.empty:
            df_cpy = df_cpy.merge(station_meta, how='left', on='codigoestacao')

    # Select only the columns that are needed;
    df_cpy = df_cpy[STATION_COLS]
    return df_cpy, missing_values


def aggregate_data(
    df: pd.DataFrame, 
    frequency: str = 'h'
) -> pd.DataFrame:
    """
    Resample time series data to specified frequency using station-specific aggregation;
    
    Groups by station code and resamples Data_Hora_Medicao to target frequency. Applies AGG_DICT
    aggregation rules (mean for values, max for status codes). Converts values to float and rounds
    to 3 decimal places;
    
    Parameters:
        df (pd.DataFrame): Input dataframe with Data_Hora_Medicao and codigoestacao columns;
        frequency (str): Pandas frequency string for resampling (default: 'h').
            Examples: '15min', 'h', 'D', 'W'. See pandas date offset documentation;
    
    Returns:
        pd.DataFrame: Aggregated dataframe resampled to specified frequency;
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


def melt_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Reshape dataframe from long to wide format with station-metric column naming;
    
    Melts dataframe to long format, creates station-metric column names (e.g., 'Cota_Adotada_87450004'),
    then pivots to wide format. Automatically excludes columns where all values are missing.
    Handles duplicate timestamps using pivot_table with aggfunc='first';
    
    Parameters:
        df (pd.DataFrame): Long-format dataframe with Data_Hora_Medicao, codigoestacao, and value columns;
    
    Returns:
        pd.DataFrame: Wide-format dataframe with Data_Hora_Medicao as index and station-metric columns;
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