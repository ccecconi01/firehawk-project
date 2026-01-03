import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Evita travamentos de interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("--- PASSO 2: TREINO FINAL (Log Transform + Sanitização) ---")

# 1. CARREGAR DADOS
print("A carregar dados...")
df = pd.read_csv('dataset_final_limpo.csv')

# --- 2. SANITIZAÇÃO E PREPARAÇÃO ---

# A. Correção Crítica da Hora (Antes de apagar colunas!)
# Se 'Hora' não existir mas houver 'Hora_x' (do ICNF), renomeamos para 'Hora'
if 'Hora' not in df.columns:
    if 'Hora_x' in df.columns:
        print("-> A recuperar coluna 'Hora' (de Hora_x)...")
        df = df.rename(columns={'Hora_x': 'Hora'})
    elif 'Hora_y' in df.columns:
        df = df.rename(columns={'Hora_y': 'Hora'})

# B. Corrigir Altitude Negativa (O erro de satélite que notaste)
if 'ALTITUDEMEDIA' in df.columns:
    df['ALTITUDEMEDIA'] = df['ALTITUDEMEDIA'].clip(lower=0)

# C. Remover Colunas Redundantes
# (Removemos Hora_y, Latitude do fogos.pt, etc., mantendo apenas as oficiais do ICNF)
cols_to_drop = ['Latitude', 'Longitude', 'Data', 'Hora_x', 'Hora_y', 'Distrito', 'Concelho', 'Freguesia']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

print("-> Dados sanitizados: Altitudes corrigidas e colunas duplicadas removidas.")

# --- 3. DEFINIR FEATURES E TARGETS ---
features = [
    'Duracao_Horas', 'Area_Ardida_ha', 
    'FWI', 'ISI', 
    'TEMPERATURA', 'VENTOINTENSIDADE', 
    'DECLIVEMEDIO', 'ALTITUDEMEDIA',
    'Mes', 'Hora', 
    'LAT', 'LON'
]

targets = ['Operacionais_Man', 'Meios_Terrestres', 'Meios_Aereos']

# Garantir que só usamos colunas que realmente existem após a limpeza
features = [col for col in features if col in df.columns]
y_cols = [col for col in targets if col in df.columns]

print(f"-> Features usadas: {features}")

X = df[features]
y = df[y_cols]
ids = df['NCCO']

# --- 4. TRANSFORMAÇÃO LOGARÍTMICA (O segredo para o R2 positivo) ---
# Aplicamos log(y+1) para "esmagar" a diferença entre fogos pequenos e gigantes
print("-> A aplicar transformação Logarítmica nos targets...")
y_log = np.log1p(y)

# 5. SPLIT TREINO (80%) / TESTE (20%)
X_train, X_test, y_train_log, y_test_log, id_train, id_test = train_test_split(
    X, y_log, ids, test_size=0.2, random_state=42
)

print(f"-> Treino: {len(X_train)} | Teste: {len(X_test)} incêndios")

# 6. TREINAR (Random Forest)
print("A treinar modelo (escala logarítmica)...")
# n_jobs=-1 usa todos os processadores para ser mais rápido
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
model.fit(X_train, y_train_log)

# 7. PREVISÃO E PÓS-PROCESSAMENTO
y_pred_log = model.predict(X_test)

# --- REVERTER A TRANSFORMAÇÃO (Voltar para números reais) ---
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test_log)

# Garantir que não há valores negativos (impossível ter -1 bombeiros)
y_pred_real = np.maximum(y_pred_real, 0)

# 8. MÉTRICAS FINAIS
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print("-" * 40)
print(f"RESULTADO FINAL (R2 Score): {r2:.4f}")
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"RMSE (Erro Quadrático): {rmse:.2f}")
print("-" * 40)

# 9. EXPORTAR CSV DE RESULTADOS
resultados = pd.DataFrame({'ID_Incendio': id_test})

for i, col in enumerate(y_cols):
    resultados[f'Real_{col}'] = y_test_real[col].values
    resultados[f'Previsto_{col}'] = y_pred_real[:, i].round(0)
    resultados[f'Erro_{col}'] = resultados[f'Previsto_{col}'] - resultados[f'Real_{col}']

resultados.to_csv('resultados_finais_log.csv', index=False)
print("-> 'resultados_finais_log.csv' criado com sucesso.")

# 10. GERAR GRÁFICOS
print("A gerar gráficos...")

# Gráfico A: Importância das Features
if len(model.estimators_) > 0:
    importances = model.estimators_[0].feature_importances_
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, legend=False, palette='viridis')
    plt.title('O que define o nº de Operacionais?')
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    plt.savefig('grafico_importancia.png')
    print("-> 'grafico_importancia.png' salvo.")

# Gráfico B: Dispersão Log-Log (Melhor para ver a precisão em várias escalas)
if len(y_cols) > 0:
    first_target = y_cols[0] # Operacionais
    plt.figure(figsize=(8, 8))
    
    # Plotar dados
    plt.scatter(y_test_real[first_target], y_pred_real[:, 0], alpha=0.3)
    
    # Linha de referência perfeita (Vermelha tracejada)
    max_val = max(y_test_real[first_target].max(), y_pred_real[:, 0].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel(f'Real ({first_target})')
    plt.ylabel(f'Previsto ({first_target})')
    plt.title(f'Precisão: {first_target} (Escala Log)')
    
    # Usar escala log nos eixos para facilitar a visualização
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig('grafico_precisao_log.png')
    print("-> 'grafico_precisao_log.png' salvo.")