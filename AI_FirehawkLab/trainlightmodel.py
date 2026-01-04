import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
import joblib



#print("--- FINAL TRAINING: RANDOM FOREST (CORRECT SANITIZATION & GEOGRAPHY) ---")

# 1. LOAD DATA
print("Loading data...")
try:
    df = pd.read_csv('dataset_final_clean.csv')
except FileNotFoundError:
    print("CRITICAL ERROR: Could not find 'dataset_final_clean.csv'. Check the folder.")
    exit()

#2. SANITIZATION TAKEN OUT AND ADDED TO FILTERCLEANMERGE.PY

# 2.3 OPTIONAL: CLEAN COLUMN NAMES (remove spaces)
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

# 2.4 CLEAN NATUREZA COLUMN (Remove leading/trailing whitespace, standardize case)
# This handles typos like 'Mato ', ' Agrícola', 'MATO' etc.
if 'Natureza' in df.columns:
    df['Natureza'] = df['Natureza'].str.strip()  # Remove spaces before/after
    df['Natureza'] = df['Natureza'].str.title()  # Standardize to Title Case


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

# 2.6B DISTRITO ONE-HOT (operational footprint proxy)
'''if 'DISTRITO' in df.columns:
    df['DISTRITO'] = df['DISTRITO'].astype(str).str.strip().str.title()
    distrito_encoded = pd.get_dummies(df['DISTRITO'], prefix='Distrito', drop_first=False)
    df = pd.concat([df, distrito_encoded], axis=1)
    print(f"-> One-hot encoded Distrito: {len(distrito_encoded.columns)} categories")'''

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

# Additional interactions commonly used in operational models
'''df['Temp_Wind_Interaction'] = df['TEMPERATURA'] * df['VENTOINTENSIDADE']
df['FWI_VPD_Interaction'] = df['FWI'] * df['VPD_kPa']
print("-> Created Temp*Wind and FWI*VPD interactions")'''

# 2.8 CYCLICAL TIME FEATURES + CALENDAR
if 'DHINICIO' in df.columns:
    dt = pd.to_datetime(df['DHINICIO'], errors='coerce')
else:
    dt = pd.NaT

# Hour/Mes already present; build cyclical encodings (robust for tree ensembles too)
'''if 'Hora' in df.columns:
    df['Hora_sin'] = np.sin(2 * np.pi * df['Hora'] / 24.0)
    df['Hora_cos'] = np.cos(2 * np.pi * df['Hora'] / 24.0)
if 'Mes' in df.columns:
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12.0)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12.0)'''

# Day-of-week and weekend flags if DHINICIO present
'''if isinstance(dt, pd.Series):
    df['DiaSemana'] = dt.dt.dayofweek
    df['FimDeSemana'] = df['DiaSemana'].isin([5, 6]).astype(int)
    print("-> Added calendar features: DiaSemana, FimDeSemana, cyclical encodings")'''

# 3. DEFINE FEATURES

features = [
    'LAT', 'LON',             # Geography  
    'Mes', 'Hora',            # Temporal
    #'Hora_sin', 'Hora_cos', 'Mes_sin', 'Mes_cos', 'DiaSemana', 'FimDeSemana',
    'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FM', # Physics (FWI System)
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'VPD_kPa', # Weather
    'DECLIVEMEDIO', 'ALTITUDEMEDIA', # Topography
    'FWI_Wind_Interaction', 'FM_Slope_Interaction'#, 'Temp_Wind_Interaction', 'FWI_VPD_Interaction' # Interactions
]

# Add one-hot encoded Natureza columns
natureza_cols = [col for col in df.columns if col.startswith('Natureza_')]
features.extend(natureza_cols)

# Add Distrito one-hot columns if present
#distrito_cols = [col for col in df.columns if col.startswith('Distrito_')]
#features.extend(distrito_cols)

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

# 4-5. SPLIT (keep targets on original scale; we will transform inside the model)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42
)

# 6. TRAIN RANDOM FOREST (restored)
print("Training Random Forest (CV-based parameter sweep)...")

candidate_params = [
    {"n_estimators": 300, "max_depth": 14, "min_samples_split": 10, "min_samples_leaf": 4},
    {"n_estimators": 500, "max_depth": 18, "min_samples_split": 8, "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": None, "min_samples_split": 6, "min_samples_leaf": 3},
    {"n_estimators": 250, "max_depth": 12, "min_samples_split": 12, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 1},
]

best_model = None
best_cv_score = -np.inf
best_params = None

for i, params in enumerate(candidate_params, start=1):
    print(f"-> Sweep {i}/{len(candidate_params)}: {params}")
    base_rf = RandomForestRegressor(
        **params,
        random_state=42,
        n_jobs=-1
    )
    '''ttr = TransformedTargetRegressor(
        regressor=base_rf,
        func=np.log1p,
        inverse_func=np.expm1
    )'''
    model_candidate = MultiOutputRegressor(base_rf)
    
    # Use 5-fold CV on training set 
    cv_scores = cross_val_score(
        model_candidate, 
        X_train, 
        y_train, 
        cv=5, 
        scoring='r2',
        n_jobs=-1
    )
    mean_cv_score = cv_scores.mean()
    print(f"   CV R2: {mean_cv_score:.4f} (±{cv_scores.std():.4f})")

    if mean_cv_score > best_cv_score:
        best_cv_score = mean_cv_score
        best_params = params
        # Don't save model_candidate here - retrain on full train set after sweep

print(f"\nBest RF params from CV: {best_params} (CV R2={best_cv_score:.4f})")

# Retrain best model on full training set
print("-> Retraining best model on full training set...")
rf_reg_best = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputRegressor(
    TransformedTargetRegressor(
        regressor=rf_reg_best,
        func=np.log1p,
        inverse_func=np.expm1
    )
)
model.fit(X_train, y_train)

# 7. PREDICTION (already on original scale due to TransformedTargetRegressor)
y_pred_real = model.predict(X_test)
y_test_real = y_test.values

y_pred_real = np.maximum(y_pred_real, 0)
y_test_real = np.maximum(y_test_real, 0)

# DEBUG: Check raw prediction distribution
print("\n[DEBUG] Prediction Statistics:")
print(f"y_pred_real range: [{y_pred_real.min():.2f}, {y_pred_real.max():.2f}]")
print(f"y_test_real range: [{y_test_real.min():.2f}, {y_test_real.max():.2f}]")
print(f"y_pred_real mean: {y_pred_real.mean():.2f}, y_test_real mean: {y_test_real.mean():.2f}")
print(f"y_pred_real std: {y_pred_real.std():.2f}, y_test_real std: {y_test_real.std():.2f}")

# 8. METRICS
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("-" * 40)
print(f"FINAL RESULT (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("-" * 40)

# Per-target diagnostics
for i, col in enumerate(targets):
    r2_t = r2_score(y_test_real[:, i], y_pred_real[:, i])
    mae_t = mean_absolute_error(y_test_real[:, i], y_pred_real[:, i])
    print(f"{col}: R2={r2_t:.4f} | MAE={mae_t:.2f}")
    print(f"  -> Pred range: [{y_pred_real[:, i].min():.2f}, {y_pred_real[:, i].max():.2f}]")

# 7. SAVE MODEL AND FEATURES (CRUCIAL!)
joblib.dump(model, 'model_resources_lite.pkl')

#SAVE FEATURE LIST
joblib.dump(features, 'model_features_list.pkl')

print("\n-> SUCCESS!")
print("-> Model saved: model_resources_lite.pkl")
print("-> Feature list saved: model_features_list.pkl (DO NOT DELETE THIS FILE)")
# 9. EXPORT
'''
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
'''
print("Completed!")