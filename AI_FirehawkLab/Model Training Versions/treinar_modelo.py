import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- 1. LOAD DATA ---
print("1. Loading files...")
# The file you just created
df_features = pd.read_csv('dataset_incendios_features.csv', dtype={'NCCO': str})

# Your operatives file (CONFIRM THE FILE NAME HERE)
df_ops = pd.read_csv('dados_operacionais.csv', dtype={'id': str})

# --- 2. MERGE (THE JUNCTION) ---
print("2. Joining tables...")
# Join by fire ID
df_final = pd.merge(df_features, df_ops, left_on='NCCO', right_on='id', how='inner')
print(f"   -> Fires found in both tables: {len(df_final)}")

# --- 3. PREPARATION ---
# Define Targets (What we want to predict)
targets = ['man', 'terrain', 'aerial']
df_final[targets] = df_final[targets].fillna(0)

# Define Features (What the model will use to learn)
# Note that I added simple weather and FWI
features_cols = [
    'Duracao_Horas', 
    'Area_Ardida_ha',   # The "analytical" cheat
    'FWI', 'ISI',       # Canadian Indices (Danger)
    'TEMPERATURA', 'VENTOINTENSIDADE', # Simple Weather
    'DECLIVEMEDIO',     # Topography
    'Mes', 'Hora',      # Time
    'LAT', 'LON'        # Location
]

# Final cleanup of NaNs in inputs
X = df_final[features_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df_final[targets]

# --- 4. TRAINING ---
print("3. Training Random Forest (this may take 1 minute)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# --- 5. EVALUATION ---
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("-" * 40)
print(f"FINAL RESULTS:")
print(f"R2 Score (Overall Accuracy): {r2:.4f} (Ideal > 0.6)")
print(f"Mean Absolute Error: {mae:.2f} resource difference")
print("-" * 40)

# --- 6. GRAPHS FOR THE REPORT ---
print("4. Generating graphs...")

# GRAPH 1: Feature Importance (What defines the number of firefighters?)
# We take the first estimator (Operatives 'man')
importances = model.estimators_[0].feature_importances_
feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
plt.title('What most influences the number of Operatives?')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('grafico_importancia.png') # Save to disk
plt.show()

# GRAPH 2: Real vs Predicted (For Operatives)
plt.figure(figsize=(8, 8))
plt.scatter(y_test['man'], y_pred[:, 0], alpha=0.5, color='blue')
plt.plot([y.min().min(), y.max().max()], [y.min().min(), y.max().max()], 'r--', lw=2) # Perfect line
plt.xlabel('Real (Number of Operatives)')
plt.ylabel('Predicted by Model')
plt.title('Accuracy: Real vs Predicted')
plt.grid(True)
plt.savefig('grafico_precisao.png')
plt.show()