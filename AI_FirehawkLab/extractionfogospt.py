import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import sys

# Base API URL
BASE_URL = "https://api.fogos.pt/v2/incidents/search"

# --- SMART DATE DETECTION LOGIC ---
last_date_recorded = None
# Default start date if no file is found 
start_date_dt = datetime(2019, 1, 1) 

try:
    print("-> Checking dataset_final_clean.csv for the last recorded date...")
    # Read only DHINICIO column to be fast
    df_check = pd.read_csv('dataset_final_clean.csv', usecols=['DHINICIO'])
    df_check['DHINICIO'] = pd.to_datetime(df_check['DHINICIO'], errors='coerce')
    
    # Drop rows with invalid dates (NaT)
    df_valid = df_check.dropna(subset=['DHINICIO'])
    
    if not df_valid.empty:
        last_date_recorded = df_valid['DHINICIO'].max()
        # Add 1 minute to start fetching strictly after the last record
        start_date_dt = last_date_recorded + timedelta(minutes=1)
        print(f"-> Last date found: {last_date_recorded}")
        print(f"-> Starting extraction from: {start_date_dt}")
    else:
        print("-> File found but contains no valid dates. Using default (2023).")

except FileNotFoundError:
    print("-> dataset_final_clean1.csv not found. Starting from scratch (2023).")
except Exception as e:
    print(f"-> Error reading last date ({e}). Using default (2023).")

# Determine years to fetch dynamically based on the start date found
current_year = datetime.now().year
start_year = start_date_dt.year
# Range needs to go up to current_year + 1 to include current_year
YEARS_TO_FETCH = list(range(start_year, current_year + 1))

# Limit of incidents to return.
LIMIT = 1000000 
all_incidents = []

for year in YEARS_TO_FETCH:
    # Logic to define the exact start date for the API parameter
    if year == start_year:
        # If it's the starting year, use the specific date we calculated
        date_param_start = start_date_dt.strftime("%Y-%m-%d")
    else:
        # For subsequent years, start from Jan 1st
        date_param_start = f"{year}-01-01"
        
    end_date = f"{year}-12-31"
    # If it's the current year, the end date is today
    if year == current_year:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading fire incident data for period: {date_param_start} to {end_date}...")

    # Request parameters
    params = {
        "after": date_param_start,
        "before": end_date,
        "limit": LIMIT
    }

    try:
        # Make GET request to API
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("success") and "data" in data:
            incidents_raw = data["data"]
            count_added = 0
            
            for incident in incidents_raw:
                # --- GRANULAR FILTERING (Minute precision) ---
                # API filters by day, but we need to filter by minute/second to avoid duplicates
                try:
                    # Construct datetime from incident data (format usually: "dd-mm-YYYY" and "HH:MM")
                    incident_date_str = f"{incident.get('date')} {incident.get('time')}"
                    incident_dt = datetime.strptime(incident_date_str, "%d-%m-%Y %H:%M")
                    
                    # If we have a cutoff date and this incident is older or equal, SKIP IT
                    if last_date_recorded and incident_dt <= last_date_recorded:
                        continue
                except (ValueError, TypeError):
                    # If date parsing fails, we keep the incident to be safe (or log error)
                    pass

                all_incidents.append(incident)
                count_added += 1
                
            print(f"-> Downloaded {len(incidents_raw)} raw, kept {count_added} new incidents for {year}.")
            
        else:
            print(f"-> Error in API response for {year}:", data.get("error", "No success flag."))

    except Exception as e:
        print(f"-> An unexpected error occurred for {year}: {e}")

print(f"\nTotal NEW incidents downloaded: {len(all_incidents)}")

if all_incidents:
    # Continue with data processing
    incidents = all_incidents

    if incidents:
        # Normalize data to pandas DataFrame
        df = pd.json_normalize(incidents)

        # Select and rename relevant columns
        
        df = df[[
            'id',
            'date',
            'hour',       
            'location',
            'district',
            'concelho',
            'freguesia',
            'natureza',
            'status',
            'man',
            'terrain',
            'aerial',
            'coords',
            'lat',
            'lng',
            'created.sec',
            'updated.sec'
        ]]
        
        
        df.columns = [
            'ID_Incidente',
            'Data',
            'Hora',
            'Localizacao',
            'Distrito',
            'Concelho',
            'Freguesia',
            'Natureza',
            'Estado',
            'Operacionais_Man',
            'Meios_Terrestres',
            'Meios_Aereos',
            'Tem_Coordenadas',
            'Latitude',
            'Longitude',
            'Criado_Timestamp',
            'Atualizado_Timestamp'
        ]

        # Convert timestamp columns to readable date format
        df['Data_Criacao'] = pd.to_datetime(df['Criado_Timestamp'], unit='s')
        df['Data_Atualizacao'] = pd.to_datetime(df['Atualizado_Timestamp'], unit='s')
        df.drop(columns=['Criado_Timestamp', 'Atualizado_Timestamp'], inplace=True)

        # Save DataFrame to CSV file (append mode)
        csv_filename = f"fogos_pt_historicaldata.csv"
        
        # Check if file exists and load existing data
        try:
            existing_df = pd.read_csv(csv_filename, encoding='utf-8-sig')
            print(f"-> Found existing file with {len(existing_df)} records. Merging with new data...")
            
            # Combine old and new data
            df = pd.concat([existing_df, df], ignore_index=True)
            
            # Remove duplicates based on ID_Incidente (keep most recent)
            df = df.drop_duplicates(subset=['ID_Incidente'], keep='last')
            
            # Sort by Data and Hora (most recent first)
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.sort_values(['Data', 'Hora'], ascending=[False, False]).reset_index(drop=True)
            
            print(f"-> After merge and deduplication: {len(df)} total records")
        except FileNotFoundError:
            print(f"-> No existing file found. Creating new file...")
        
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"Data from {YEARS_TO_FETCH[0]}_to_{end_date} successfully saved to: {csv_filename}")
    else:
        print("The API did not return incidents for the specified period.")
# Install 'requests' and 'pandas' libraries if not installed 

