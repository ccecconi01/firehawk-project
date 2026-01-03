import pandas as pd
import numpy as np

print("--- STEP 1: CLEANING AND MERGE ---")

# 1. LOAD DATA
df_icnf = pd.read_csv('dataset_fires_features.csv', dtype={'NCCO': str})
df_fogos = pd.read_csv('fogos_pt_historicaldata.csv', dtype={'ID_Incidente': str})

print(f"-> ICNF Original: {len(df_icnf)} rows")
print(f"-> Fogos.pt Original: {len(df_fogos)} rows")


# 2. REMOVE DUPLICATES 
# We keep the first record found and delete ID repetitions
df_icnf = df_icnf.drop_duplicates(subset=['NCCO'], keep='first')
df_fogos = df_fogos.drop_duplicates(subset=['ID_Incidente'], keep='first')

# 3. MERGE (The Intersection)
# Join NCCO (ICNF) with ID_Incidente (Fogos.pt)
df_final = pd.merge(df_icnf, df_fogos, left_on='NCCO', right_on='ID_Incidente', how='inner')

print(f"-> Combined Fires (Merge): {len(df_final)}")

# 3.5 CONVERT DATETIME COLUMNS
# Convert DHINICIO and DHFIM to datetime objects (they come as strings from CSV)
df_final['DHINICIO'] = pd.to_datetime(df_final['DHINICIO'], errors='coerce')
df_final['DHFIM'] = pd.to_datetime(df_final['DHFIM'], errors='coerce')

# Create Mes (month) feature from DHINICIO
df_final['Mes'] = df_final['DHINICIO'].dt.month

# 4. SANITIZATION (CRITICAL ORDER) --- 
# A. Recover the HOUR for each row that is missing it
missing_hora = df_final['Hora'].isna() | (df_final['Hora'].astype(str).str.strip() == '')

if missing_hora.sum() > 0:
    print(f"-> Recovering 'Hora' for {missing_hora.sum()} rows...")
    
    if 'Hora_x' in df_final.columns:
        mask = missing_hora & df_final['Hora_x'].notna()
        df_final.loc[mask, 'Hora'] = df_final.loc[mask, 'Hora_x']
        print(f"   - Recovered {mask.sum()} rows from Hora_x")
    
    if 'Hora_y' in df_final.columns:
        missing_hora = df_final['Hora'].isna() | (df_final['Hora'].astype(str).str.strip() == '')
        mask = missing_hora & df_final['Hora_y'].notna()
        df_final.loc[mask, 'Hora'] = df_final.loc[mask, 'Hora_y']
        print(f"   - Recovered {mask.sum()} rows from Hora_y")
    
    if 'DHINICIO' in df_final.columns:
        missing_hora = df_final['Hora'].isna() | (df_final['Hora'].astype(str).str.strip() == '')
        mask = missing_hora & df_final['DHINICIO'].notna()
        if mask.sum() > 0:
            df_final.loc[mask, 'Hora'] = pd.to_datetime(
                df_final.loc[mask, 'DHINICIO'], 
                format='%d-%m-%Y %H:%M:%S', 
                errors='coerce'
            ).dt.hour
            print(f"   - Recovered {mask.sum()} rows from DHINICIO")

# B. Create 'DHINICIO' for each row that is missing it
missing_dhinicio = df_final['DHINICIO'].isna() | (df_final['DHINICIO'].astype(str).str.strip() == '')

if missing_dhinicio.sum() > 0 and 'Data' in df_final.columns and 'Hora' in df_final.columns:
    print(f"-> Creating 'DHINICIO' for {missing_dhinicio.sum()} rows from 'Data' and 'Hora'...")
    mask = missing_dhinicio & df_final['Data'].notna() & df_final['Hora'].notna()
    df_final.loc[mask, 'DHINICIO'] = pd.to_datetime(
        df_final.loc[mask, 'Data'].astype(str) + ' ' + 
        df_final.loc[mask, 'Hora'].astype(str),  # Hora is already in time format (HH:MM)
        errors='coerce'
    )
    print(f"   - Created {mask.sum()} rows")

#Create feature "Duracao_Horas" from already verified DHINICIO and DHFIM
df_final['Duracao_Horas'] = (df_final['DHFIM'] - df_final['DHINICIO']).dt.total_seconds() / 3600 


# C. Fix Negative Altitude (Satellite)
if 'ALTITUDEMEDIA' in df_final.columns:
    df_final['ALTITUDEMEDIA'] = df_final['ALTITUDEMEDIA'].clip(lower=0)


# D REMOVE MISSING ID AND DATETIME
# Remove rows with missing ID, DHINICIO, or DHFIM
df_final = df_final.dropna(subset=['NCCO', 'ID_Incidente', 'DHINICIO', 'DHFIM'])
df_final = df_final[df_final['NCCO'].astype(str).str.strip() != '']  # Remove empty strings
df_final = df_final[df_final['ID_Incidente'].astype(str).str.strip() != '']  
df_final = df_final[df_final['DHINICIO'].astype(str).str.strip() != '']
df_final = df_final[df_final['DHFIM'].astype(str).str.strip() != ''] 

print(f"-> After removing missing ID/DHINICIO/DHFIM: {len(df_final)} rows")

# E. Remove Redundant or Empty Columns
# We remove 'Latitude'/'Longitude' (written in full) to use 'LAT'/'LON' (from CSV)
cols_to_drop = ['Latitude', 'Longitude', 'Data', 'Hora_x', 'Hora_y',
                 'Distrito','Localizacao', 'ID_Incidente', 'Tem_Coordenadas']
df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], errors='ignore')

print("-> Data sanitized: Hour recovered, altitudes corrected and duplicates removed.")

# 5. QUALITY FILTER (Remove NaNs)
# Columns that MUST have data.
# If weather, coordinates or number of firefighters are missing, the row is garbage.
essential_columns = [
    'LAT', 'LON', 'Area_Ardida_ha', 
    'FWI', 'FFMC', 'DECLIVEMEDIO', 'ALTITUDEMEDIA', # From ICNF
    'TEMPERATURA', 'VENTOINTENSIDADE', 'VPD_kPa', # From ICNF
    'Operacionais_Man', 'Meios_Terrestres' # From Fogos.pt 
]

# Force numeric conversion (errors become NaN)
for col in essential_columns:
    # Check if column exists before converting (safety)
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# THE BIG FILTER: Delete rows that have ANY missing values in essentials
df_final = df_final.dropna(subset=[c for c in essential_columns if c in df_final.columns])

# Extra Filter: Remove false positives (Area 0 or 0 Operatives)
df_final = df_final[df_final['Area_Ardida_ha'] > 0]
if 'Operacionais_Man' in df_final.columns:
    df_final = df_final[df_final['Operacionais_Man'] > 0]

# 6. Final alter to FFMC: CHange FFMC values to FM (%) as it was comproved to be a better metric for evaluating wildfires in the mediterranean climate
#df_final['FM'] = (147.2 * (101.0 - df_final['FFMC'])) / (59.5 + df_final['FFMC']) - GENERAL FORMULA FFMC ->FM
df_final['FM'] = (df_final['FFMC'] / 28.5) ** (1 / 0.281) # Formula presented in recent studies, adapted to Portuguese vegetation, specifically the Pinus pinaster species present in 30% of forest territory.
# It is assumed that other species widely present in national territory (Eucalyptus globulus and Quercus Suber) have similar behavior relative to Pinus pinaster regarding the FM.
df_final = df_final.drop(columns=['FFMC'])


# 6.2 Remover durações negativas ou nulas (Erros de registo)
df_final = df_final[df_final['Duracao_Horas'] > 0]

# 6.3 Remover Área Ardida <= 0 (Não nos interessa para treino de magnitude)
if 'Area_Ardida_ha' in df_final.columns:
    df_final = df_final[df_final['Area_Ardida_ha'] > 0]

# 6.4 (Opcional) Filtrar por Operacionais > 0 se quiseres prever apenas incêndios que tiveram combate
if 'Operacionais_Man' in df_final.columns:
    df_final = df_final[df_final['Operacionais_Man'] > 0]

# 7. SELEÇÃO FINAL DE COLUNAS
# Garantir que temos apenas o que interessa para o dataset final
colunas_finais = [
    # IDs
    'NCCO',  
    # Temporal
    'DHINICIO', 'DHFIM', 'Mes', 'Hora', 'Duracao_Horas', 'Estado',
    # Localização
    'DISTRITO', 'LAT', 'LON', 'ALTITUDEMEDIA', 'DECLIVEMEDIO','Concelho', 'Freguesia', 'Natureza',
    # Meteo / FWI
    'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FM', 
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'VPD_kPa',
    # Targets / Resultados Reais
    'Area_Ardida_ha', 'Operacionais_Man', 'Meios_Terrestres', 'Meios_Aereos'
]

# Filtramos apenas as colunas que realmente existem no dataframe (para evitar erros)
cols_existentes = [c for c in colunas_finais if c in df_final.columns]
df_final = df_final[cols_existentes]




print(f"-> FINAL DATASET (Cleaned and ready): {len(df_final)} rows")


# 8 SORT BY DATETIME (Most recent first)
if 'DHINICIO' in df_final.columns:
    df_final = df_final.sort_values('DHINICIO', ascending=False).reset_index(drop=True)
    print(f"-> Dataset sorted by DHINICIO (most recent first)")

# 9. SAVE
df_final.to_csv('dataset_final_clean.csv', index=False)
print("Success! The file 'dataset_final_clean.csv' has been created.")