import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

print("--- FINAL TRAINING: XGBOOST + VISUAL CLEANUP ---")

# 1. LOAD DATA
print("Loading data...")
df = pd.read_csv('dataset_final_limpo.csv')

# --- 2. SANITIZATION (Name and value fixes) ---
if 'Hora' not in df.columns:
    if 'Hora_x' in df.columns: df = df.rename(columns={'Hora_x': 'Hora'})
    elif 'Hora_y' in df.columns: df = df.rename(columns={'Hora_y': 'Hora'})

if 'ALTITUDEMEDIA' in df.columns:
    df['ALTITUDEMEDIA'] = df['ALTITUDEMEDIA'].clip(lower=0)

# 3. DEFINE FEATURES (WITHOUT LAT/LON)
# XGBoost will focus on fire severity (FWI, Area) and not position
features = [
    'Duracao_Horas', 'Area_Ardida_ha', 
    'FWI', 'ISI', 
    'TEMPERATURA', 'VENTOINTENSIDADE', 
    'DECLIVEMEDIO', 'ALTITUDEMEDIA',
    'Mes', 'Hora'
]

targets = ['Operacionais_Man', 'Meios_Terrestres', 'Meios_Aereos']

# Ensure columns exist
features = [col for col in features if col in df.columns]
y_cols = [col for col in targets if col in df.columns]

print(f"-> Features used: {features}")

X = df[features]
y = df[y_cols]
ids = df['NCCO']

# 4. LOG TRANSFORM (Essential)
y_log = np.log1p(y)

# 5. SPLIT
X_train, X_test, y_train_log, y_test_log, id_train, id_test = train_test_split(
    X, y_log, ids, test_size=0.2, random_state=42
)

# 6. TRAIN XGBOOST
print("Training XGBoost (V12 Engine)...")
xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1
)

model = MultiOutputRegressor(xg_reg)
model.fit(X_train, y_train_log)

# 7. PREDICTION AND POST-PROCESSING
y_pred_log = model.predict(X_test)

# --- REVERSE LOG AND CLEAN "GARBAGE" ---
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test_log)

# THE MAGIC CORRECTION:
# 1. Ensure everything is zero or higher (removes -0.000004)
y_pred_real = np.maximum(y_pred_real, 0)
y_test_real = np.maximum(y_test_real, 0)

# 8. METRICS
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("-" * 40)
print(f"XGBOOST RESULT (R2 Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("-" * 40)

# 9. EXPORT CLEAN CSV (INTEGERS)
resultados = pd.DataFrame({'ID_Incendio': id_test})

for i, col in enumerate(y_cols):
    # .round(0).astype(int) forces the number to be integer (5 not 5.0)
    resultados[f'Real_{col}'] = y_test_real[col].values.round(0).astype(int)
    resultados[f'Previsto_{col}'] = y_pred_real[:, i].round(0).astype(int)
    resultados[f'Erro_{col}'] = results = resultados[f'Previsto_{col}'] - resultados[f'Real_{col}']

resultados.to_csv('resultados_xgboost_limpo.csv', index=False)
print("-> 'resultados_xgboost_limpo.csv' created (no floats and no negatives).")

# 10. GRAPHS
print("Generating graphs...")

# Importance Graph
if len(model.estimators_) > 0:
    importances = model.estimators_[0].feature_importances_
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, legend=False, palette='magma')
    plt.title('What does XGBoost value?')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('grafico_importancia_xgboost.png')

# Scatter
if len(y_cols) > 0:
    first_target = y_cols[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_real[first_target], y_pred_real[:, 0], alpha=0.3, color='purple')
    max_val = max(y_test_real[first_target].max(), y_pred_real[:, 0].max())
    plt.plot([0, max_val], [0, max_val], 'k--', lw=2)
    plt.xlabel(f'Real ({first_target})')
    plt.ylabel(f'Predicted ({first_target})')
    plt.title(f'XGBoost: Real vs Predicted (Log Scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('grafico_precisao_xgboost.png')