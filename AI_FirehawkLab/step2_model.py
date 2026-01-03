import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("--- FINAL TRAINING: RANDOM FOREST (CORRECT SANITIZATION & GEOGRAPHY) ---")

# 1. LOAD DATA
print("Loading data...")
try:
    df = pd.read_csv('dataset_final_clean.csv')
except FileNotFoundError:
    print("CRITICAL ERROR: Could not find 'dataset_final_clean1.csv'. Check the folder.")
    exit()

#2. SANITIZATION TAKEN OUT AND ADDED TO FILTERCLEANMERGE.PY

# 2.5 CONVERT HORA FROM TIME STRING TO NUMERIC FOR ML
if 'Hora' in df.columns:
    # Extract hour from time string format "HH:MM"
    df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M', errors='coerce').dt.hour
    print(f"-> Converted Hora to numeric (0-23)")



# 2.6 ONE-HOT ENCODING FOR NATUREZA (fire type)
# Reference: Forestry research shows different fire types require different resource allocation strategies
# - Mato (shrubland): Rapid spread, lower intensity, fewer personnel needed
# - Agrícola (agricultural): Lower spread rate, containable with fewer resources
# - Povoamento Florestal (forest): Highest intensity, most resources required
if 'Natureza' in df.columns:
    natureza_encoded = pd.get_dummies(df['Natureza'], prefix='Natureza', drop_first=False)
    df = pd.concat([df, natureza_encoded], axis=1)
    print(f"-> One-hot encoded Natureza: {list(natureza_encoded.columns)}")

# 2.7 INTERACTION FEATURES (Academic basis below)
# Reference: Rothermel (1972) and subsequent fire behavior models show that resource needs scale multiplicatively
# with both fuel availability (FM/FFMC) and environmental drivers (wind, temperature)

# FWI * Wind interaction: High fire danger + high wind = exponential spread rate
# This is the basis of the Canadian Fire Weather Index System (Van Wagner & Pickett, 1985)
df['FWI_Wind_Interaction'] = df['FWI'] * df['VENTOINTENSIDADE']
print(f"-> Created FWI * Wind interaction (fire danger × wind speed)")

# FM * Declivity interaction: Fuel moisture × slope affects fire intensity
# Reference: Cruz & Alexander (2010) - Assessing improvements in the Canadian forest fire weather index
df['FM_Slope_Interaction'] = df['FM'] * df['DECLIVEMEDIO']
print(f"-> Created FM * Slope interaction (fuel state × terrain steepness)")



# 3. DEFINE FEATURES
# Based on your real list: We use LAT and LON, not Latitude/Longitude
features = [
    'LAT', 'LON',             # Geography  
    'Mes', 'Hora',            # Temporal
    'Duracao_Horas', 'Area_Ardida_ha', # Fire
    'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FM', # Physics (FWI)
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'VPD_kPa', # Weather
    'DECLIVEMEDIO', 'ALTITUDEMEDIA', # Topography
    'FWI_Wind_Interaction', 'FM_Slope_Interaction',
]


natureza_cols = [col for col in df.columns if col.startswith('Natureza_')]
features.extend(natureza_cols)

targets = ['Operacionais_Man', 'Meios_Terrestres', 'Meios_Aereos']

# Safety Validation
missing = [c for c in features if c not in df.columns]
if missing:
    print(f"!!! WARNING !!! Missing critical columns: {missing}")
else:
    print(f"-> All {len(features)} features are ready.")

X = df[features]
y = df[targets]
ids = df['NCCO']

# 4. LOG TRANSFORM
y_log = np.log1p(y)

# 5. SPLIT
X_train, X_test, y_train_log, y_test_log, id_train, id_test = train_test_split(
    X, y_log, ids, test_size=0.2, random_state=42
)

# 6. TRAIN RANDOM FOREST
print("Training Random Forest (Robust Configuration)...")
rf_reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,         
    min_samples_split=10,
    min_samples_leaf=4,   
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(rf_reg)
model.fit(X_train, y_train_log)

# 7. PREDICTION
y_pred_log = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test_log)

y_pred_real = np.maximum(y_pred_real, 0)
y_test_real = np.maximum(y_test_real, 0)

# 8. METRICS
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("-" * 40)
print(f"FINAL RESULT (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("-" * 40)

# 9. EXPORT
resultados = pd.DataFrame({'ID_Incendio': id_test})
for i, col in enumerate(targets):
    resultados[f'Real_{col}'] = y_test_real[col].values.round(0).astype(int)
    resultados[f'Previsto_{col}'] = y_pred_real[:, i].round(0).astype(int)
    resultados[f'Erro_{col}'] = resultados[f'Previsto_{col}'] - resultados[f'Real_{col}']

resultados.to_csv('resultados_rf1_final.csv', index=False)
print("-> 'resultados_rf1_final.csv' saved.")

# 10. GRAPHS (Visual Correction)
print("Generating graphs...")

if hasattr(model.estimators_[0], 'feature_importances_'):
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    # SEABORN CORRECTION: hue assigned to index and legend=False
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, legend=False, palette='viridis')
    plt.title('Importance Variable (RF + LAT/LON + FWI)')
    plt.tight_layout()
    plt.savefig('gra1_importance_final.png')

if len(targets) > 0:
    target_plot = targets[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_real[target_plot], y_pred_real[:, 0], alpha=0.4, color='royalblue', edgecolor='k', s=50)
    max_val = max(y_test_real[target_plot].max(), y_pred_real[:, 0].max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
    plt.xlabel(f'Real ({target_plot})')
    plt.ylabel(f'Predicted ({target_plot})')
    plt.title(f'Accuracy: {target_plot}')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('gra1_precision_final.png')

print("Completed!")