"""
This code is used to pull all the informations from a single station considering the limits the API has to pull data on
specific predetermined chunks of data, determined on the variable 'available_ranges'. This function divides the required
data range on appropriate chunks, make the requests for each chunk and concats everything. On the main call of the
function, multiple stations should be defined by defining multiple executions of the 'get_station_data' data. This will
garantee the reproductability of the data colletion;
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import os
import json
import requests
import pandas as pd
from source_api.api_auth import get_auth
from db_handler import DBConnection
from util import START_DATE, END_DATE

########################################################################################################################
#
# FUNCTION
#
########################################################################################################################
def get_station_data(station_code: str, 
                    start_date: str, 
                    end_date: str, 
                    inplace: bool = True, 
                    date_filter_type: str ="DATA_LEITURA", 
                    table_name: str = "gasometro",
                    ):
    """
    Gets detailed telemetric series data for a specific station from HidroWeb API
    
    Parameters:
    - station_code: Station code (e.g., "87450004")
    - start_date: Start date (format: yyyy-MM-dd)
    - end_date: End date (format: yyyy-MM-dd)
    - date_filter_type: Type of date filter (default: "DATA_LEITURA")
    - table_name: Name of the table to save in database (default: "gasometro")
    """
    
    print(f"\n[api_single_station.py] {'='*60}")
    print(f"[api_single_station.py] STARTING DATA COLLECTION FOR STATION {station_code}")
    print(f"[api_single_station.py] Date range: {start_date} to {end_date}")
    print(f"[api_single_station.py] Table name: {table_name}")
    print(f"[api_single_station.py] {'='*60}\n")
    
    # Get proper HidroWeb Token;
    token = get_auth()
    
    if not token:
        print("[api_single_station.py] Failed to get authentication token")
        return
    
    # Convert dates to datetime objects
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Validate date range
    if start_dt > end_dt:
        print("[api_single_station.py] Error: start_date cannot be after end_date")
        return
    
    # Calculate total days
    total_days = (end_dt - start_dt).days + 1
    
    print(f"[api_single_station.py] Total date range: {total_days} days")
    print(f"[api_single_station.py] Will be split into chunks of maximum 30 days each")
    print(f"[api_single_station.py] Estimated number of API calls: {(total_days + 29) // 30}")
    print()
    
    # Base URL;
    url = "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaDetalhada/v1"
    
    # Headers with the authorization token;
    headers = {
        "accept": "*/*",
        "Authorization": f"Bearer {token}"
    }
    
    # Initialize empty list to store all dataframes
    all_dataframes = []
    
    # Available range values from the API
    available_ranges = [
        "MINUTO_5", "MINUTO_10", "MINUTO_15", "MINUTO_30",
        "HORA_1", "HORA_2", "HORA_3", "HORA_4", "HORA_5", "HORA_6", "HORA_7", "HORA_8", "HORA_9", "HORA_10", "HORA_11", "HORA_12",
        "HORA_13", "HORA_14", "HORA_15", "HORA_16", "HORA_17", "HORA_18", "HORA_19", "HORA_20", "HORA_21", "HORA_22", "HORA_23", "HORA_24",
        "DIAS_2", "DIAS_7", "DIAS_14", "DIAS_21", "DIAS_30"
    ]
    
    # Split the date range into chunks using available range values
    current_date = start_dt
    chunk_number = 1
    estimated_chunks = (total_days + 29) // 30  # Rough estimate
    auth_retry_count = 0
    max_auth_retries = 5
    
    print(f"[api_single_station.py] Starting API calls... (Estimated chunks: {estimated_chunks})")
    print("[api_single_station.py] " + "-" * 40)
    
    while current_date <= end_dt:
        # Calculate remaining days
        remaining_days = (end_dt - current_date).days + 1
        
        # Choose the best available range for remaining days
        if remaining_days >= 30:
            range_days = "DIAS_30"
            chunk_days = 30
        elif remaining_days >= 21:
            range_days = "DIAS_21"
            chunk_days = 21
        elif remaining_days >= 14:
            range_days = "DIAS_14"
            chunk_days = 14
        elif remaining_days >= 7:
            range_days = "DIAS_7"
            chunk_days = 7
        elif remaining_days >= 2:
            range_days = "DIAS_2"
            chunk_days = 2
        else:
            # For 1 day, we'll use DIAS_2 but only process 1 day
            range_days = "DIAS_2"
            chunk_days = 1
        
        # Calculate the end date for this chunk
        chunk_end_date = min(current_date + timedelta(days=chunk_days-1), end_dt)
        actual_chunk_days = (chunk_end_date - current_date).days + 1
        
        # Format dates for API
        search_date = current_date.strftime("%Y-%m-%d")
        
        # Query parameters for this chunk;
        params = {
            "Código da Estação": station_code,
            "Tipo Filtro Data": date_filter_type,
            "Data de Busca (yyyy-MM-dd)": search_date,
            "Range Intervalo de busca": range_days
        }
        
        # Create request for this chunk;
        print(f"[api_single_station.py] Chunk {chunk_number}: Getting data from {search_date} using {range_days} (actual days: {actual_chunk_days})...")
        response = requests.get(url, headers=headers, params=params)
        
        # Request Response for this chunk;
        if response.status_code == 200:
            station_data = response.json()
            print(f'[api_single_station.py] ✓ Chunk {chunk_number} data collected!')
            
            # Turn station json into a proper dataframe;
            if 'items' in station_data and station_data['items']:
                df = pd.DataFrame(station_data['items'])
                all_dataframes.append(df)
                print(f'[api_single_station.py]   → Chunk {chunk_number}: {len(df)} records collected')
            else:
                print(f'[api_single_station.py]   → Chunk {chunk_number}: No data items found in the response')

            # Reset auth retry counter on success;
            auth_retry_count = 0
            
            # Move to next chunk only if status_code == 200;
            current_date = chunk_end_date + timedelta(days=1)
            chunk_number += 1

        # Get authentication again if the token has expired;
        elif response.status_code == 401:
            auth_retry_count += 1
            
            if auth_retry_count > max_auth_retries:
                print(f'[api_single_station.py] ✗ Max authentication retries ({max_auth_retries}) reached. API may be down.')
                print(f'[api_single_station.py]   → Stopping data collection at chunk {chunk_number}')
                break
            
            print(f'[api_single_station.py] ⚠ Authentication expired. Retrying... (Attempt {auth_retry_count}/{max_auth_retries})')
            token = get_auth()
            headers["Authorization"] = f"Bearer {token}"

        else:
            print(f'[api_single_station.py] ✗ Chunk {chunk_number}: Request failed with status code: {response.status_code}')
            print(f'[api_single_station.py]   → Response text: {response.text}')
            # Move to next chunk on other errors to avoid infinite loop;
            current_date = chunk_end_date + timedelta(days=1)
            chunk_number += 1
    
    # Combine all dataframes and save to database
    print("\n[api_single_station.py] " + "-" * 40)
    print("[api_single_station.py] PROCESSING COMPLETE")
    print("[api_single_station.py] " + "-" * 40)
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"[api_single_station.py] ✓ Total records collected: {len(combined_df)}")
        
        print(f"[api_single_station.py] → Saving data to database table '{table_name}'...")
        db_handler = DBConnection()
        db_handler.write(combined_df, table_name, inplace=inplace)
        print(f"[api_single_station.py] ✓ All data successfully saved to table '{table_name}'")
    else:
        print("[api_single_station.py] ✗ No data was collected from any chunk")
    
    print(f"\n[api_single_station.py] {'='*60}")
    print(f"[api_single_station.py] SCRIPT EXECUTION COMPLETED")
    print(f"[api_single_station.py] {'='*60}\n")

# Example usage
if __name__ == "__main__":
    #================== Estações Guaíba ==================
    get_station_data(station_code="87450004", start_date="2014-07-01", end_date="2024-05-03", table_name="station_guaiba_1")
    get_station_data(station_code="87444000", start_date="2024-05-03", end_date=END_DATE, table_name="station_guaiba_2")
    
    #================== Estações Gravataí ==================
    get_station_data(station_code="87399000", start_date="2018-07-01", end_date=END_DATE, table_name="station_gravatai_1")
    
    #================== Estações Sinos ==================
    get_station_data(station_code="87382000", start_date="2018-07-01", end_date=END_DATE, table_name="station_sinos_1")
    get_station_data(station_code="87380000", start_date="2013-12-01", end_date=END_DATE, table_name="station_sinos_2")

    #================== Estações Taquari ==================
    get_station_data(station_code="86510000", start_date="2017-10-01", end_date=END_DATE, table_name="station_taquari_1")
    get_station_data(station_code="86720000", start_date="2008-12-01", end_date=END_DATE, table_name="station_taquari_2")

    #================== Estações Caí ==================
    get_station_data(station_code="87150000", start_date="2010-01-01", end_date=END_DATE, table_name="station_cai_1")
    get_station_data(station_code="87170000", start_date="2018-01-01", end_date=END_DATE, table_name="station_cai_2")

    #================== Estações Jacuí ==================
    get_station_data(station_code="85900000", start_date="2017-10-01", end_date=END_DATE, table_name="station_jacui_1")
    print('[api_single_station.py] All Done!')