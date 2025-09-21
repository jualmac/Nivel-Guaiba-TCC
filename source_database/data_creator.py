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
from source_database.db_handler import DBConnection
from util import convert_to_float, STATIONS_COLS, START_DATE, END_DATE

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def collect_all_stations(save_to_db: bool = False, frequency: str = 'H'):
    """
    Retrieve the data from the stations and return a single dataframe concatenated;

    Parameters:
        save_to_db (bool): Whether to save the dataframe to the database;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'H' for hourly).
            Examples: 'min' (minutes), 'H' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...

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
        'cai_3':        'SELECT * FROM station_cai_3',
        'gravatai_1':   'SELECT * FROM station_gravatai_1',
        'guaiba_1':     'SELECT * FROM station_guaiba_1', 
        'guaiba_2':     'SELECT * FROM station_guaiba_2',
        'jacui_1':      'SELECT * FROM station_jacui_1',
        'jacui_2':      'SELECT * FROM station_jacui_2',
        'sinos_1':      'SELECT * FROM station_sinos_1', 
        'sinos_2':      'SELECT * FROM station_sinos_2',
        'sinos_3':      'SELECT * FROM station_sinos_3',
        'taquari_1':    'SELECT * FROM station_taquari_1', 
        'taquari_2':    'SELECT * FROM station_taquari_2',
        'taquari_3':    'SELECT * FROM station_taquari_3',

        }
    dataframe = db.run(query=query)

    # Open query into single dfs;
    cai_1       =   dataframe.get('cai_1')
    cai_2       =   dataframe.get('cai_2')
    cai_3       =   dataframe.get('cai_3')
    gravatai_1  =   dataframe.get('gravatai_1')
    guaiba_1    =   dataframe.get('guaiba_1')
    guaiba_2    =   dataframe.get('guaiba_2')
    jacui_1     =   dataframe.get('jacui_1')
    jacui_2     =   dataframe.get('jacui_2')
    sinos_1     =   dataframe.get('sinos_1')
    sinos_2     =   dataframe.get('sinos_2')
    sinos_3     =   dataframe.get('sinos_3')
    taquari_1   =   dataframe.get('taquari_1')
    taquari_2   =   dataframe.get('taquari_2')
    taquari_3   =   dataframe.get('taquari_3')

    # Merge guaiba_1 and guaiba_2 into a single dataframe;
    guaiba_1 = pd.concat([guaiba_1, guaiba_2])
    guaiba_1['codigoestacao'] = '87450004'

    # Create a dictionary mapping station names to dataframes for easier identification;
    stations_dataframes = {
        'cai_1':        cai_1,
        'cai_2':        cai_2,
        'cai_3':        cai_3,
        'gravatai_1':   gravatai_1, 
        'guaiba_1':     guaiba_1,
        'jacui_1':      jacui_1,
        'jacui_2':      jacui_2,
        'sinos_1':      sinos_1,
        'sinos_2':      sinos_2,
        'sinos_3':      sinos_3,
        'taquari_1':    taquari_1,
        'taquari_2':    taquari_2,
        'taquari_3':    taquari_3,
    }

    # Check the station frequency;
    frequency_results, gaps_df = analyze_station_frequencies(frequency=frequency)
    
    # Concatenate the dataframes;
    df = pd.concat(list(stations_dataframes.values()))

    # Select only the columns that are needed;
    df = df[STATIONS_COLS.keys()]
    df.rename(columns=STATIONS_COLS, inplace=True)

    # Clean the data using specified frequency;
    df_cleaned = clean_data(df=df, frequency=frequency)

    # Save the dataframes to the database;
    if save_to_db:
        db.write(df=df, table_name='data_stations', inplace=True)
        db.write(df=df_cleaned, table_name='data_stations_cleaned', inplace=True)
        db.write(df=frequency_results, table_name='data_stations_frequency', inplace=True)
        db.write(df=gaps_df, table_name='data_stations_gaps', inplace=True)
    return df, df_cleaned

def clean_data(df: pd.DataFrame, frequency: str = 'H', max_ffill_steps: int = 8):
    """
    Clean the concatenated data from the different stations. Cuts the dataframe to a time range where most data is
    available. Groups the data by specified frequency intervals and melts the dataframe to long format -> Better for the Machine
    Learning Models;

    Parameters:
        df (pd.DataFrame): The dataframe to clean;
        frequency (str): Pandas frequency offset string for data aggregation (default: 'H' for hourly).
            Examples: 'min' (minutes), 'H' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...
        max_ffill_steps (int): The maximum number of steps to forward-fill the data;

    Returns:
        df_cleaned (pd.DataFrame): The cleaned dataframe;
    """

    #TODO: Fix the data gaps -> interpolate() -> SPECIFC FUNCTION;
    #TODO: Fix outlier values;
    #TODO: Fix the missing level data -> Usar Cota Manual/Cota Sensor;

    # Cut the dataframe to a time range where most data is available;
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= START_DATE]
    df = df[df['date'] <= END_DATE]

    # Convert the value columns to float;
    for col in (set(df.columns) - {'date', 'station_id'}):
        df[col] = df[col].apply(convert_to_float)

    # Data Aggregation using specified frequency;
    agg_dict = {'level': 'mean',
                'level_status': 'median',
                'rainfall': 'mean',
                'rainfall_status': 'median',
                'rainfall_accumulated': 'mean',
                'rainfall_accumulated_status': 'median',
                'flow': 'mean',
                'flow_status': 'median',
                'temperature': 'mean'
                }

    # Resample the data to the desired frequency to create continuous timeline and fill gaps;
    df_cleaned = (
        df.groupby('station_id', group_keys=True)
          .apply(lambda g: g.resample(frequency, on='date').agg(agg_dict))
          .reset_index()
    )

    # After df_cleaned is created;
    df_cleaned['level'] = df_cleaned.groupby('station_id')['level'].ffill(limit=max_ffill_steps)  # Forward-fill short gaps
    df_cleaned['is_missing'] = df_cleaned['level'].isna().astype(int)  # Gap indicator feature

    # Get the value columns (excluding date and station_id);
    value_cols = [col for col in df.columns if col not in ['date', 'station_id']]
    
    #TODO: Separete this on a specific function;
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

#TODO: Fix this shit: Negative values;
def analyze_station_frequencies(frequency: str = '15min'):
    """
    Analyze the time frequency patterns for each station to verify if they maintain
    consistent intervals throughout their historical datasets based on the given frequency.
    Analysis is limited to the date range defined by START_DATE and END_DATE.
    
    Parameters:
        frequency (str): Pandas frequency offset string to analyze (default: '15min' for 15-minute intervals).
            Examples: 'min' (minutes), 'H' (hours), 'D' (days), 'W' (weeks), ...
            Available options: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
            Combinations are also possible, e.g. '15min', '30min', '1H20m', ...
    
    Returns:
        frequency_analysis (dict): Analysis results for each station including:
            - total_records: number of records
            - date_range: start and end dates (filtered by START_DATE/END_DATE)
            - expected_intervals: expected number of intervals for the given frequency
            - actual_intervals: actual number of records
            - frequency_stats: statistics about time gaps
            - missing_intervals: percentage of missing data
            - common_intervals: most common time intervals found
            - gap_details: list of detailed information about each large gap
        gaps_df (pd.DataFrame): Detailed information about all large gaps including:
            - station: station name
            - timestamp_before_gap: timestamp right before the gap
            - timestamp_after_gap: timestamp right after the gap
            - gap_duration_minutes: duration of gap in minutes
            - gap_duration_hours: duration of gap in hours
            - expected_records_in_gap: number of records that should exist in gap based on frequency
    """
    print("="*80)
    print("ANALYZING TIME FREQUENCY PATTERNS FOR ALL STATIONS")
    print(f"Date range limited to: {START_DATE} to {END_DATE}")
    print(f"Analyzing frequency: {frequency}")
    print("="*80)
    
    # Convert frequency to minutes for calculations;
    freq_offset = pd.tseries.frequencies.to_offset(frequency)
    freq_minutes = freq_offset.delta.total_seconds() / 60
    
    # Define what constitutes a "large gap" (more than 1.5x the expected frequency);
    large_gap_threshold = freq_minutes * 1.5
    
    # Initialize the Connection
    db = DBConnection()

    # Query data (using same queries as collect_all_stations)
    query = {
        'cai_1':        'SELECT Data_Hora_Medicao FROM station_cai_1 ORDER BY Data_Hora_Medicao',
        'cai_2':        'SELECT Data_Hora_Medicao FROM station_cai_2 ORDER BY Data_Hora_Medicao',
        'cai_3':        'SELECT Data_Hora_Medicao FROM station_cai_3 ORDER BY Data_Hora_Medicao',
        'gravatai_1':   'SELECT Data_Hora_Medicao FROM station_gravatai_1 ORDER BY Data_Hora_Medicao',
        'guaiba_1':     'SELECT Data_Hora_Medicao FROM station_guaiba_1 ORDER BY Data_Hora_Medicao', 
        'guaiba_2':     'SELECT Data_Hora_Medicao FROM station_guaiba_2 ORDER BY Data_Hora_Medicao',
        'jacui_1':      'SELECT Data_Hora_Medicao FROM station_jacui_1 ORDER BY Data_Hora_Medicao',
        'jacui_2':      'SELECT Data_Hora_Medicao FROM station_jacui_2 ORDER BY Data_Hora_Medicao',
        'sinos_1':      'SELECT Data_Hora_Medicao FROM station_sinos_1 ORDER BY Data_Hora_Medicao', 
        'sinos_2':      'SELECT Data_Hora_Medicao FROM station_sinos_2 ORDER BY Data_Hora_Medicao',
        'sinos_3':      'SELECT Data_Hora_Medicao FROM station_sinos_3 ORDER BY Data_Hora_Medicao',
        'taquari_1':    'SELECT Data_Hora_Medicao FROM station_taquari_1 ORDER BY Data_Hora_Medicao', 
        'taquari_2':    'SELECT Data_Hora_Medicao FROM station_taquari_2 ORDER BY Data_Hora_Medicao',
        'taquari_3':    'SELECT Data_Hora_Medicao FROM station_taquari_3 ORDER BY Data_Hora_Medicao',
    }
    
    dataframes = db.run(query=query)
    
    frequency_analysis = {}
    
    for station_name, df in dataframes.items():
        if df is None or df.empty:
            print(f"\n{station_name}: No data available")
            frequency_analysis[station_name] = {
                'error': 'No data available',
                'total_records': 0
            }
            continue
            
        print(f"\n{'-'*60}")
        print(f"ANALYZING STATION: {station_name.upper()}")
        print(f"{'-'*60}")
        
        # Convert to datetime
        df['Data_Hora_Medicao'] = pd.to_datetime(df['Data_Hora_Medicao'])
        df = df.dropna(subset=['Data_Hora_Medicao'])  # Remove any null timestamps
        original_records = len(df)
        
        # Filter by START_DATE and END_DATE range
        df = df[df['Data_Hora_Medicao'] >= START_DATE]
        df = df[df['Data_Hora_Medicao'] <= END_DATE]
        df = df.sort_values('Data_Hora_Medicao').reset_index(drop=True)
        
        filtered_records = len(df)
        excluded_records = original_records - filtered_records
        
        total_records = len(df)
        
        if total_records < 2:
            print(f"Insufficient data (only {total_records} records)")
            frequency_analysis[station_name] = {
                'error': 'Insufficient data',
                'total_records': total_records
            }
            continue
            
        # Calculate time differences between consecutive records
        df['time_diff'] = df['Data_Hora_Medicao'].diff()
        
        # Remove the first NaT value from diff calculation
        time_diffs = df['time_diff'].dropna()
        
        # Convert to minutes for easier analysis
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60
        
        # Basic statistics
        start_date = df['Data_Hora_Medicao'].min()
        end_date = df['Data_Hora_Medicao'].max()
        total_duration = end_date - start_date
        
        # Expected number of intervals for the given frequency
        expected_intervals = int(total_duration.total_seconds() / (freq_minutes * 60)) + 1
        
        # Find most common intervals (rounded to nearest minute)
        interval_counts = time_diffs_minutes.round().value_counts().head(10)
        
        # Calculate percentage of exactly expected frequency intervals
        exactly_freq = (time_diffs_minutes.round() == freq_minutes).sum()
        percent_freq = (exactly_freq / len(time_diffs_minutes)) * 100
        
        # Find gaps larger than expected frequency
        large_gaps = time_diffs_minutes[time_diffs_minutes > large_gap_threshold]
        
        # Create detailed gap analysis - identify timestamps before and after each gap
        gap_details = []
        large_gap_mask = time_diffs_minutes > large_gap_threshold
        
        if large_gap_mask.any():
            large_gap_indices = time_diffs_minutes[large_gap_mask].index
            
            for gap_idx in large_gap_indices:
                gap_duration_minutes = time_diffs_minutes.loc[gap_idx]  # Use .loc instead of .iloc
                gap_duration_hours = gap_duration_minutes / 60
                
                # Get timestamps before and after the gap
                # gap_idx corresponds to the row after the gap in the original dataframe
                timestamp_before = df.loc[gap_idx - 1, 'Data_Hora_Medicao']
                timestamp_after = df.loc[gap_idx, 'Data_Hora_Medicao']
                
                gap_info = {
                    'station': station_name,
                    'timestamp_before_gap': timestamp_before.strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp_after_gap': timestamp_after.strftime('%Y-%m-%d %H:%M:%S'),
                    'gap_duration_minutes': round(gap_duration_minutes, 2),
                    'gap_duration_hours': round(gap_duration_hours, 2),
                    'expected_records_in_gap': int(gap_duration_minutes / freq_minutes) - 1  # Subtract 1 for the actual record
                }
                gap_details.append(gap_info)
        
        # Statistics
        missing_percentage = ((expected_intervals - total_records) / expected_intervals) * 100
        
        # Store analysis results
        analysis_result = {
            'total_records': total_records,
            'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_days': total_duration.days,
            'frequency_analyzed': frequency,
            'frequency_minutes': freq_minutes,
            'expected_intervals': expected_intervals,
            'missing_intervals': expected_intervals - total_records,
            'missing_percentage': round(missing_percentage, 2),
            'exactly_freq_intervals': exactly_freq,
            'percent_exactly_freq': round(percent_freq, 2),
            'large_gaps_count': len(large_gaps),
            'largest_gap_hours': round(large_gaps.max() / 60, 2) if len(large_gaps) > 0 else 0,
            'common_intervals_minutes': interval_counts.to_dict(),
            'avg_interval_minutes': round(time_diffs_minutes.mean(), 2),
            'median_interval_minutes': round(time_diffs_minutes.median(), 2),
            'gap_details': gap_details,
        }
        
        frequency_analysis[station_name] = analysis_result
        
        # Print detailed results
        print(f"Total records (filtered): {total_records:,}")
        if excluded_records > 0:
            print(f"Records excluded by date filter: {excluded_records:,}")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_duration.days} days)")
        print(f"Expected {frequency} intervals: {expected_intervals:,}")
        print(f"Missing intervals: {expected_intervals - total_records:,} ({missing_percentage:.1f}%)")
        print(f"Exactly {frequency} intervals: {exactly_freq:,} ({percent_freq:.1f}%)")
        print(f"Large gaps (>{large_gap_threshold:.1f}min): {len(large_gaps):,}")
        if len(large_gaps) > 0:
            print(f"Largest gap: {large_gaps.max() / 60:.1f} hours")
        print(f"Average interval: {time_diffs_minutes.mean():.1f} minutes")
        print(f"Most common intervals (minutes):")
        for interval, count in interval_counts.head(5).items():
            percentage = (count / len(time_diffs_minutes)) * 100
            print(f"{interval:.0f} min: {count:,} times ({percentage:.1f}%)")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}")
    
    valid_stations = {k: v for k, v in frequency_analysis.items() if 'error' not in v}
    
    if valid_stations:
        avg_missing = np.mean([v['missing_percentage'] for v in valid_stations.values()])
        avg_freq_compliance = np.mean([v['percent_exactly_freq'] for v in valid_stations.values()])
        
        print(f"Average missing data across stations: {avg_missing:.1f}%")
        print(f"Average {frequency} compliance: {avg_freq_compliance:.1f}%")
        
        # Stations with best/worst compliance
        compliance_sorted = sorted(valid_stations.items(), key=lambda x: x[1]['percent_exactly_freq'], reverse=True)
        print(f"Best compliance: {compliance_sorted[0][0]} ({compliance_sorted[0][1]['percent_exactly_freq']:.1f}%)")
        print(f"Worst compliance: {compliance_sorted[-1][0]} ({compliance_sorted[-1][1]['percent_exactly_freq']:.1f}%)")
    
    # Compile all gap details into a single dataframe for easier analysis
    all_gaps = []
    for station_name, analysis in valid_stations.items():
        if 'gap_details' in analysis and analysis['gap_details']:
            all_gaps.extend(analysis['gap_details'])
    
    # Create gap details dataframe
    if all_gaps:
        gaps_df = pd.DataFrame(all_gaps)
        gaps_df = gaps_df.sort_values(['station', 'timestamp_before_gap'])
        print(f"\nTotal large gaps found across all stations: {len(gaps_df)}")
        print(f"Stations with gaps: {gaps_df['station'].nunique()}")
        print(f"Average gap duration: {gaps_df['gap_duration_hours'].mean():.1f} hours")
        print(f"Largest gap found: {gaps_df['gap_duration_hours'].max():.1f} hours")
    else:
        gaps_df = pd.DataFrame()  # Empty dataframe if no gaps found
        print("\nNo large gaps found across any stations")
    return frequency_analysis, gaps_df

# def regularize_station(df: pd.DataFrame, 
#                     freq: str = '15min', 
#                     value_cols: tuple = ('level','rainfall','rainfall_accumulated','temperature'),
#                     max_ffill_steps: int = 8
#                     ) -> pd.DataFrame:
    
#     #TODO: Implement the function;
    
#     # df columns: ['unique_id','ds', <value_cols>]; assumes a single station in df;
#     df = df.sort_values('ds').drop_duplicates('ds');  # enforce order and unique timestamps;
#     idx = pd.date_range(df['ds'].min().floor(freq), df['ds'].max().ceil(freq), freq=freq);
#     out = pd.DataFrame({'ds': idx}).merge(df, on='ds', how='left');

#     # Gap features;
#     out['is_missing_row'] = out[value_cols].isna().all(axis=1).astype('int8');  # full-row miss flag;

#     # Forward-fill within a cap to avoid smearing across long outages;
#     for c in value_cols:
#         out[f'{c}_was_ffill'] = out[c].isna().astype('int8');
#         out[c] = out[c].ffill(limit=max_ffill_steps);  # only short gaps will be filled;
        
#     # Optional: time-based interpolation for continuous variables (not for pulses/counters);
#     for c in ('level','temperature'):
#         out[c] = out[c].interpolate(method='time', limit_area='inside');  # safer for medium gaps;

#     # For event-like variables:
#     # - rainfall (incremental per step): avoid filling with zeros unless you are sure "missing = no rain";
#     # - rainfall_accumulated (cumulative): reconstruct per-step rainfall via diff if needed, handle resets;
#     # Leave remaining NaNs for tree models (they can handle NaN) and SARIMA/LSTM segments may need drop/impute;
#     return out;

if __name__ == "__main__":
    df, df_cleaned = collect_all_stations(save_to_db=False, frequency='H')
    print("All Done!")