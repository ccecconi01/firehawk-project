import requests
import pandas as pd
import joblib
import numpy as np
import datetime
from datetime import timedelta
import math
import sys
import os

# --- CONFIGURATION ---
MODEL_FILE = 'model_resources_lite.pkl'
FEATURES_FILE = 'model_features_list.pkl'
OUTPUT_FILE = 'dashboard_predictions.csv'
TOP_N_RECENT = 20

# ==========================================
# 1. SCIENTIFIC FWI CALCULATION ENGINE
# ==========================================
def calculate_fwi_codes(temp, rh, wind_kph, rain_mm, month):
    # Standard start-up values
    prev_ffmc = 85.0
    prev_dmc = 20.0
    prev_dc = 100.0
    
    # FFMC
    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)
    if rain_mm > 0.5:
        rf = rain_mm - 0.5
        if mo > 150:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf)) \
                 + 0.0015 * (mo - 150.0)**2 * math.sqrt(rf)
        if mr > 250: mr = 250
        mo = mr

    ed = 0.942 * (rh**0.679) + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    ew = 0.618 * (rh**0.753) + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))

    if mo > ed:
        kw = 0.424 * (1.0 - (rh / 100.0)**1.7) + 0.0694 * math.sqrt(wind_kph) * (1.0 - (rh / 100.0)**8)
        kw = kw * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) / 10.0**kw
    elif mo < ew:
        if ew < ed: ew = ed
        kw = 0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7) + 0.0694 * math.sqrt(wind_kph) * (1.0 - ((100.0 - rh) / 100.0)**8)
        kw = kw * 0.581 * math.exp(0.0365 * temp)
        m = ew - (ew - mo) / 10.0**kw
    else:
        m = mo

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    ffmc = max(0, min(101, ffmc))

    # DMC
    el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    t_eff = max(-1.1, temp)
    rk = 1.894 * (t_eff + 1.1) * (100.0 - rh) * el[month-1] * 0.0001
    
    if rain_mm > 1.5:
        re = 0.92 * rain_mm - 1.27
        mo_dmc = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        if prev_dmc <= 33:
            b = 100.0 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc <= 65:
            b = 14.0 - 1.3 * math.log(prev_dmc)
        else:
            b = 6.2 * math.log(prev_dmc) - 17.2
        mr_dmc = mo_dmc + 1000.0 * re / (48.77 + b * re)
        pr_dmc = 43.43 * (5.6348 - math.log(max(0.1, mr_dmc - 20.0)))
        dmc = max(0, pr_dmc + rk)
    else:
        dmc = max(0, prev_dmc + rk)

    # DC
    fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    vp = 0.36 * (temp + 2.8) + fl[month-1]
    vp = max(0, vp)
    
    if rain_mm > 2.8:
        rd = 0.83 * rain_mm - 1.27
        Qo = 800.0 * math.exp(-prev_dc / 400.0)
        Qr = Qo + 3.937 * rd
        Dr = 400.0 * math.log(800.0 / Qr)
        dc = max(0, Dr + 0.5 * vp)
    else:
        dc = max(0, prev_dc + 0.5 * vp)

    # ISI
    f_wind = math.exp(0.05039 * wind_kph)
    f_ffmc = 91.9 * math.exp(-0.1386 * m) * (1.0 + m**5.31 / (4.93 * 10**7))
    isi = 0.208 * f_wind * f_ffmc

    # BUI
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc)**1.7)
    bui = max(0, bui)

    # FWI
    if bui <= 80:
        f_bui = 0.626 * bui**0.809 + 2.0
    else:
        f_bui = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    
    fwi_val = 0.208 * isi * f_bui
    
    if fwi_val > 1:
        fwi_final = math.exp(2.72 * (0.434 * math.log(fwi_val))**0.647)
    else:
        fwi_final = fwi_val

    return {
        'FFMC': round(ffmc, 1), 'DMC': round(dmc, 1), 'DC': round(dc, 1),
        'ISI': round(isi, 1), 'BUI': round(bui, 1), 'FWI': round(fwi_final, 1)
    }

# ==========================================
# 2. WEATHER & ELEVATION APIs
# ==========================================

def get_historical_weather(lat, lon, date_obj):
    """Fetches past weather from Open-Meteo Archive (for the specific fire date)."""
    date_str = date_obj.strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date_str, "end_date": date_str,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
        "timezone": "auto"
    }
    
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        idx = 12 
        
        temp = data['hourly']['temperature_2m'][idx]
        hum = data['hourly']['relative_humidity_2m'][idx]
        wind = data['hourly']['wind_speed_10m'][idx]
        rain_sum = sum(data['hourly']['rain'][:13])

        return {
            'TEMPERATURA': float(temp), 'HUMIDADERELATIVA': float(hum),
            'VENTOINTENSIDADE': float(wind), 'CHUVA_24H': float(rain_sum),
            'SUCCESS': True
        }
    except Exception as e:
        return {'SUCCESS': False}

def get_real_time_weather(lat, lon):
    """Fetches current weather from Open-Meteo Forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain"
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()['current']
        return {
            'TEMPERATURA': float(data['temperature_2m']),
            'HUMIDADERELATIVA': float(data['relative_humidity_2m']),
            'VENTOINTENSIDADE': float(data['wind_speed_10m']),
            'CHUVA_24H': float(data['rain']),
            'SUCCESS': True
        }
    except Exception as e:
        return {'SUCCESS': False}

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        r = requests.get(url, timeout=5).json()
        return r.get('elevation', [200])[0]
    except:
        return 200.0

# ==========================================
# 3. HELPER: FETCH RECENT HISTORY (V2 API)
# ==========================================

def fetch_recent_history_v2(days_back):
    """
    Query v2/incidents/search from [Now - days_back] to [Now].
    """
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    url = "https://api.fogos.pt/v2/incidents/search"
    
    params = {
        "after": start_date.strftime("%Y-%m-%d"),
        "before": end_date.strftime("%Y-%m-%d"),
        "limit": 500 # Get a large batch
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success") and "data" in data:
                return data["data"]
    except Exception as e:
        print(f"   -> API V2 Error: {e}")
        
    return []

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_pipeline():
    print("--- STARTING PIPELINE (HYBRID & ADAPTIVE) ---")

    # A. Load Model
    try:
        model = joblib.load(MODEL_FILE)
        model_features = joblib.load(FEATURES_FILE)
        print("-> Model 'Lite' loaded successfully.")
    except FileNotFoundError:
        print(f"CRITICAL: Run 'train_lite_model.py' first.")
        sys.exit(1)

    # B. Build Fire List (Adaptive Expansion)
    # ----------------------------------------------------------------
    collected_fires = []
    seen_ids = set() 
    
    # 1. First, check Active Fires (new/fires)
    print("-> 1. Checking Active Fires (new/fires)...")
    try:
        resp = requests.get("https://api.fogos.pt/new/fires", timeout=10)
        if resp.status_code == 200:
            active_data = resp.json().get('data', [])
            for f in active_data:
                fid = f.get('id')
                if fid:
                    f['is_active_api'] = True
                    collected_fires.append(f)
                    seen_ids.add(fid)
            print(f"   Found {len(active_data)} active fires.")
    except Exception as e:
        print(f"   Error checking active fires: {e}")

    # 2. Adaptive Backfill (Loop until we have TOP_N_RECENT)
    days_window = 3 # Start checking last 3 days
    max_days = 90   # Max limit to stop infinite loop
    
    while len(collected_fires) < TOP_N_RECENT and days_window <= max_days:
        needed = TOP_N_RECENT - len(collected_fires)
        print(f"-> 2. Expanding Search: Checking last {days_window} days (Need {needed} more)...")
        
        history_data = fetch_recent_history_v2(days_back=days_window)
        
        # Add NEW unique fires found in this window
        added_in_this_pass = 0
        for f in history_data:
            fid = f.get('id')
            if fid not in seen_ids:
                f['is_active_api'] = False
                collected_fires.append(f)
                seen_ids.add(fid)
                added_in_this_pass += 1
        
        print(f"   Found {added_in_this_pass} new historic fires.")
        
        if len(collected_fires) >= TOP_N_RECENT:
            break
            
        # Increase window size for next iteration (e.g., +7 days)
        days_window += 7

    # C. Date Parsing & Sorting
    # -------------------------------
    valid_fires = []
    for f in collected_fires:
        time_str = f.get('hour') or f.get('time') or "00:00"
        date_str = f.get('date') 
        
        try:
            full_dt_str = f"{date_str} {time_str}"
            dt_obj = datetime.datetime.strptime(full_dt_str, "%d-%m-%Y %H:%M")
        except:
            dt_obj = datetime.datetime.now()
            
        f['_dt_obj'] = dt_obj
        valid_fires.append(f)

    # Sort descending
    valid_fires.sort(key=lambda x: x['_dt_obj'], reverse=True)
    target_fires = valid_fires[:TOP_N_RECENT]

    if not target_fires:
        print("-> No fires found even after expanding search.")
        return

    print(f"-> Processing Top {len(target_fires)} incidents...")
    processed_rows = []

    # D. Enrichment Loop
    # -------------------------
    for i, fire in enumerate(target_fires):
        status = fire.get('status', 'Unknown')
        print(f"   [{i+1}/{len(target_fires)}] Date: {fire.get('date')} {fire.get('time') or fire.get('hour')} | Loc: {fire.get('location', fire.get('concelho'))}")
        
        try:
            lat = float(fire['lat'])
            lon = float(fire['lng'])
        except:
            continue 

        # Weather Strategy
        is_active_status = any(s in status for s in ['Curso', 'Despacho', 'Ativo', 'Vigilância'])
        use_realtime = fire.get('is_active_api', False) or is_active_status

        if use_realtime:
            w_data = get_real_time_weather(lat, lon)
        else:
            w_data = get_historical_weather(lat, lon, fire['_dt_obj'])

        temp = w_data.get('TEMPERATURA', 20.0)
        rh = w_data.get('HUMIDADERELATIVA', 50.0)
        wind = w_data.get('VENTOINTENSIDADE', 10.0)
        rain = w_data.get('CHUVA_24H', 0.0)

        fwi_idx = calculate_fwi_codes(temp, rh, wind, rain, fire['_dt_obj'].month)
        
        altitude = get_elevation(lat, lon)
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        ea = es * (rh / 100.0)
        vpd = es - ea
        
        declive = 12.0
        fm = (fwi_idx['FFMC'] / 28.5) ** (1 / 0.281)
        natureza = fire.get('natureza', 'Mato')
        if natureza: natureza = natureza.strip().title()

        row_model = {
            'LAT': lat, 'LON': lon,
            'Mes': fire['_dt_obj'].month,
            'Hora': fire['_dt_obj'].hour,
            'TEMPERATURA': temp,
            'HUMIDADERELATIVA': rh,
            'VENTOINTENSIDADE': wind,
            'FWI': fwi_idx['FWI'],
            'DMC': fwi_idx['DMC'],
            'DC': fwi_idx['DC'],
            'ISI': fwi_idx['ISI'],
            'BUI': fwi_idx['BUI'],
            'FM': fm,
            'VPD_kPa': vpd,
            'ALTITUDEMEDIA': altitude,
            'DECLIVEMEDIO': declive,
            'FWI_Wind_Interaction': fwi_idx['FWI'] * wind,
            'FM_Slope_Interaction': fm * declive
        }

        df_single = pd.DataFrame([row_model])
        for feature in model_features:
            if feature.startswith('Natureza_'):
                cat_name = feature.replace('Natureza_', '')
                df_single[feature] = 1 if natureza == cat_name else 0

        X_pred = pd.DataFrame()
        for col in model_features:
            X_pred[col] = df_single[col] if col in df_single.columns else 0.0
        X_pred = X_pred[model_features]

        preds = model.predict(X_pred)
        preds = np.maximum(preds, 0).round(0).astype(int)

        real_aerial = fire.get('aerial') or fire.get('air') or 0
        real_man = fire.get('man') or 0
        real_terrain = fire.get('terrain') or 0

        final_row = {
            'id': fire['id'],
            'data': fire.get('date'),
            'hora': fire.get('time') or fire.get('hour'),
            'status': status,
            'local': fire.get('location', fire.get('concelho', '')),
            'lat': lat, 'lon': lon,
            'temp': round(temp, 1),
            'humidade': round(rh, 0),
            'vento': round(wind, 1),
            'fwi': fwi_idx['FWI'],
            'Prev_Homens': preds[0][0],
            'Prev_Terrestres': preds[0][1],
            'Prev_Aereos': preds[0][2],
            'Real_Homens': real_man,
            'Real_Terrestres': real_terrain,
            'Real_Aereos': real_aerial
        }
        processed_rows.append(final_row)

    if processed_rows:
        df_out = pd.DataFrame(processed_rows)
        df_out['temp_sort_dt'] = pd.to_datetime(
            df_out['data'] + ' ' + df_out['hora'], 
            format='%d-%m-%Y %H:%M', 
            errors='coerce'
        )
        df_out = df_out.sort_values(by='temp_sort_dt', ascending=False)
        df_out = df_out.drop(columns=['temp_sort_dt'])
        
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"-> SUCCESS: {len(df_out)} incidents saved to '{OUTPUT_FILE}'.")
        print("-> Data Sample (Sorted):")
        print(df_out[['data', 'hora', 'local', 'Prev_Homens']].head())
    else:
        print("-> No data processed.")

if __name__ == "__main__":
    run_pipeline()