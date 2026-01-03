import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import os

# List of years
# --- LÓGICA DE DETEÇÃO DE DATA ---
last_date_recorded = None
start_year_scan = 2019 # Default

try:
    print("-> A verificar dataset_final_clean.csv...")
    if os.path.exists('dataset_final_clean.csv'):
        # READS JUST DHINICIO
        df_check = pd.read_csv('dataset_final_clean.csv', usecols=['DHINICIO'])
        df_check['DHINICIO'] = pd.to_datetime(df_check['DHINICIO'], errors='coerce')
        df_validas = df_check.dropna(subset=['DHINICIO'])
        
        if not df_validas.empty:
            last_date_recorded = df_validas['DHINICIO'].max()
            start_year_scan = last_date_recorded.year
            print(f"-> Last recorded date: {last_date_recorded}")
            print(f"-> Starting extraction from year: {start_year_scan}")
        else:
            print("-> No valid dates. Full extraction (2019+).")
    else:
        print("-> File does not exist. Full extraction (2019+).")

except Exception as e:
    print(f"-> Error checking file ({e}). Full extraction (2019+).")

# Define years to be fetched
current_year = datetime.now().year
anos = list(range(start_year_scan, current_year + 1))
todos_dados = []

print(f"--- STARTING EXTRACTION FOR YEARS: {anos} ---")

for ano in anos:
    url = f"https://fogos.icnf.pt/localizador/webserviceocorrencias.asp?ano={ano}" #Searching for annual blocks to surpass API safety limits but also timeout error
    print(f"-> Downloading year {ano}...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        contagem_ano = 0
        
        for child in root:
            dados_incendio = {}
            
            # 1. THE KEY (NCCO)
            ncco = child.find('NCCO')
            if ncco is not None:
                dados_incendio['NCCO'] = ncco.text
            else:
                # If no NCCO, tries ID, if no ID, skips
                continue

            # --- Time/Present records Filter ---
            # Verifies if fire already exists in our records based on DHINICIO
            dhinicio_tag = child.find('DHINICIO')
            if dhinicio_tag is not None and dhinicio_tag.text:
                try:
                    # ICNF Format: DD-MM-YYYY HH:MM:SS
                    dhinicio_dt = datetime.strptime(dhinicio_tag.text, '%d-%m-%Y %H:%M:%S')
                    
                    # if we already have register and the date shown is older or equal, skip
                    if last_date_recorded and dhinicio_dt <= last_date_recorded:
                        continue 
                except ValueError:
                    pass # if date is rarely malformed, we proceed to extract it anyway


            # 2. DEFINE WHAT WE WANT TO EXTRACT
            # Mapping: "Name we want in CSV" : "Name of TAG in XML"
            mapa_colunas = {
                'DHINICIO': 'DHINICIO',
                'DHFIM': 'DHFIM',
                'DISTRITO': 'DISTRITO',
                'LAT': 'LAT',
                'LON': 'LON',
                'FWI': 'FWI',
                'DMC': 'DMC',
                'DC': 'DC',
                'ISI': 'ISI',
                'BUI': 'BUI',
                'FFMC': 'FFMC',
                'ALTITUDEMEDIA': 'ALTITUDEMEDIA',
                'DECLIVEMEDIO': 'DECLIVEMEDIO',
                'TEMPERATURA': 'TEMPERATURA',
                'HUMIDADERELATIVA': 'HUMIDADERELATIVA',
                'VENTOINTENSIDADE': 'VENTOINTENSIDADE',
                'Area_Ardida_ha': 'AREATOTAL'

            }
            
            # extraction
            for nome_csv, tag_xml in mapa_colunas.items():
                elemento = child.find(tag_xml)
                if elemento is not None and elemento.text:
                    dados_incendio[nome_csv] = elemento.text
                else:
                    dados_incendio[nome_csv] = None # Creates the column even if empty
            
            todos_dados.append(dados_incendio)
            contagem_ano += 1
            
        print(f"   Success: {contagem_ano} lines extracted from {ano}.")
        
    except Exception as e:
        print(f"   Error in year {ano}: {e}")

# --- CREATE FINAL DATAFRAME ---
df = pd.DataFrame(todos_dados)

# --- CLEANING ---
print("\nProcessing data...")

# 1. CRITICAL DATE CONVERSION (Smart Filter)
# Convert dates immediately. If DHFIM fails or is missing, it becomes NaT (Not a Time)
if 'DHFIM' in df.columns:
    df['DHFIM'] = pd.to_datetime(df['DHFIM'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    
    # LOGIC: If there is no end date, it does not belong in the historical dataset. DISCARD!
    # This removes "In Progress" fires from this file (they will be handled by the Active Pipeline script)
    initial_len = len(df)
    df = df.dropna(subset=['DHFIM'])
    print(f"-> DHFIM Filter: {initial_len - len(df)} active/incomplete fires ignored.")
else:
    # If the DHFIM column is missing entirely, the dataset is empty or corrupted
    df = pd.DataFrame() 

# Proceed only if there is data left
if not df.empty:
    if 'DHINICIO' in df.columns:
        df['DHINICIO'] = pd.to_datetime(df['DHINICIO'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    # List of numeric columns to convert
    cols_numericas = [
        'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FFMC', 
        'ALTITUDEMEDIA', 'DECLIVEMEDIO', 'LAT', 'LON', 
        'Area_Ardida_ha',
        'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'VPD_kPa'
    ]

    for col in cols_numericas:
        # Extra safety check: only converts if column exists
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If the column does not exist in the XML, create it with 0
            df[col] = 0
            
# Fill failures with mean
df = df.fillna(df.mean(numeric_only=True))

# Convert dates TO datetime format
df['DHINICIO'] = pd.to_datetime(df['DHINICIO'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
df['DHFIM'] = pd.to_datetime(df['DHFIM'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

#Calculates VPD (Vapor Pressure Deficit) from Temperature and Relative Humidity
    # Function to calculate Saturation Vapor Pressure (SVP) in kPa
def calc_svp(T_celsius):
    # Returns result in kPa (kilopascals)
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

    # Calculate VPD in kPa
T = df['TEMPERATURA']
RH = df['HUMIDADERELATIVA']

    # Calculate Saturation Vapor Pressure (SVP) for the given temperature
SVP = calc_svp(T)

    # Calculate Vapor Pressure Deficit (VPD = SVP - AVP)
df['VPD_kPa'] = SVP * (1 - RH / 100)



# --- MERGE AND SAVE---
# Check if file exists and load existing data

csv_filename = 'dataset_fires_features.csv'

if not df.empty:
    try:
        if os.path.exists(csv_filename):
            print(f"-> Loading current register: {csv_filename}")
            df_old = pd.read_csv(csv_filename)
            
            print("-> Merging new data...")
            df_final = pd.concat([df_old, df], ignore_index=True)
            
            # Remove duplicates in key (NCCO), keeping most recent (last)
            df_final = df_final.drop_duplicates(subset=['NCCO'], keep='last')
        else:
            print("-> History file not found. Creating new file.")
            df_final = df
            
        df_final.to_csv(csv_filename, index=False)
        print(f"\nCOMPLETED! Database is UPDATED. Total registers: {len(df_final)}")
        
    except Exception as e:
        print(f"-> ERROR IN MERGE/SAVE: {e}")
        
else:
    print("-> No new data extracted. The original file remains unchanged.")
