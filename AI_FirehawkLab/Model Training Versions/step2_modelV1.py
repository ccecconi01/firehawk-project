import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fixes graph freezing error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("--- STEP 2: MODEL TRAINING ---")

# 1. LOAD DATA
df = pd.read_csv('dataset_final_limpo.csv')

# --- DATA SANITIZATION (eliminate unwanted columns and fix negative altitude data) ---
# 1. Fix Negative Altitude (Transform to 0 - the negative value is an error from satellite measurement)
if 'ALTITUDEMEDIA' in df.columns:
    df['ALTITUDEMEDIA'] = df['ALTITUDEMEDIA'].clip(lower=0)

# 2. Remove Redundant Columns (Optional, but recommended for cleanup)
cols_to_drop = ['Latitude', 'Longitude', 'Data', 'Hora_x', 'Hora_y', 'Distrito', 'Concelho', 'Freguesia']
# Only removes if they exist
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

print("Data sanitized: Altitudes corrected and duplicate columns removed.")

# 3. DEFINE X and y
# Features (What the model uses to predict)
features = [
    'Duracao_Horas', 'Area_Ardida_ha', 
    'FWI', 'ISI', 
    'TEMPERATURA', 'VENTOINTENSIDADE', 
    'DECLIVEMEDIO', 'ALTITUDEMEDIA',
    'Mes', 'Hora', 
    'LAT', 'LON'
]

# Targets (What we want to predict)
targets = ['Operacionais_Man', 'Meios_Terrestres', 'Meios_Aereos']

# Ensure we only use columns that really exist in CSV
features = [col for col in features if col in df.columns]
y_cols = [col for col in targets if col in df.columns]

X = df[features]
y = df[y_cols]

# Save IDs for reference in final Excel
ids = df['NCCO']

# 3. SPLIT TRAINING (80%) / TEST (20%)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} fires | Testing: {len(X_test)} fires")

# 4. TRAIN (Random Forest)
print("Training model... (please wait)")
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# 5. PREDICTION AND METRICS
y_pred = model.predict(X_test)

# Calculate R2 (Global Accuracy)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("-" * 30)
print(f"FINAL RESULT (R2 Score): {r2:.4f}")
print(f"Average Error (Resources): {mae:.1f}")
print("-" * 30)

# 6. EXPORT TEST CSV (Real vs Predicted Comparison)
# Creates a DataFrame for you to see line by line
resultados = pd.DataFrame({'ID_Incendio': id_test})

# Adds real and predicted columns dynamically
for i, col in enumerate(y_cols):
    resultados[f'Real_{col}'] = y_test[col].values
    resultados[f'Previsto_{col}'] = y_pred[:, i].round(0) # Round
    resultados[f'Erro_{col}'] = resultados[f'Previsto_{col}'] - resultados[f'Real_{col}']

resultados.to_csv('resultados_teste_detalhado.csv', index=False)
print("-> 'resultados_teste_detalhado.csv' created successfully.")

# 7. GENERATE GRAPHS
print("Generating graphs...")

# Importance Graph (Based on 1st Target - Operatives)
if len(model.estimators_) > 0:
    importances = model.estimators_[0].feature_importances_
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, legend=False, palette='viridis')
    plt.title('What defines the number of Operatives?')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('grafico_importancia.png')
    print("-> 'grafico_importancia.png' saved.")

# Scatter Plot (Real vs Predicted for 1st target)
if len(y_cols) > 0:
    first_target = y_cols[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(resultados[f'Real_{first_target}'], resultados[f'Previsto_{first_target}'], alpha=0.5)
    plt.plot([y[first_target].min(), y[first_target].max()], [y[first_target].min(), y[first_target].max()], 'r--')
    plt.xlabel(f'Real ({first_target})')
    plt.ylabel(f'Predicted ({first_target})')
    plt.title(f'Accuracy: {first_target}')
    plt.grid(True)
    plt.savefig('grafico_precisao.png')
    print("-> 'grafico_precisao.png' saved.")