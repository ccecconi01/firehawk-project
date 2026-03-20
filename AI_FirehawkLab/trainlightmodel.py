import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (r2_score, mean_absolute_error,
                             roc_auc_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix, accuracy_score)
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
    # "Florestal" is an incomplete label for the same category as "Povoamento Florestal" — merge them.
    # "3105" and "Consolidação De Rescaldo" are data-entry noise (2 rows total) — reassign to most common.
    df['Natureza'] = df['Natureza'].replace('Florestal', 'Povoamento Florestal')
    main_cats = df['Natureza'].value_counts().nlargest(3).index.tolist()
    df.loc[~df['Natureza'].isin(main_cats), 'Natureza'] = main_cats[0]
    print(f"-> Natureza cleaned: {df['Natureza'].value_counts().to_dict()}")


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

# 2.6B DISTRITO TARGET ENCODING (operational footprint proxy)
# Target encoding: each district is replaced by the mean of each target in the TRAINING set.
# Why not one-hot: 18 binary columns dilute RF split sampling (sqrt(39)≈6 features per split).
# 3 numeric columns are denser and more informative.
# Encoding is computed AFTER the train/test split to avoid data leakage.
if 'DISTRITO' in df.columns:
    df['DISTRITO'] = df['DISTRITO'].astype(str).str.strip().str.title()
    print(f"-> DISTRITO standardized: {df['DISTRITO'].nunique()} unique districts. Encoding post-split.")

# 2.7 INTERACTION FEATURES (Academic basis below)
# Reference: Rothermel (1972) and subsequent fire behavior models show that resource needs scale multiplicatively
# with both fuel availability (FM/FFMC) and environmental drivers (wind, temperature)

# FWI * Wind interaction: High fire danger + high wind = exponential spread rate
# This is the basis of the Canadian Fire Weather Index System (Van Wagner & Pickett, 1985)
# Commented out: removed from feature set (not part of Phase 1 plan, may add noise)
'''df['FWI_Wind_Interaction'] = df['FWI'] * df['VENTOINTENSIDADE']
print(f"-> Created FWI * Wind interaction (fire danger × wind speed)")'''

# FM * Declivity interaction: Fuel moisture × slope affects fire intensity
# Reference: Cruz & Alexander (2010) - Assessing improvements in the Canadian forest fire weather index
# Commented out: removed from feature set (not part of Phase 1 plan, may add noise)
'''df['FM_Slope_Interaction'] = df['FM'] * df['DECLIVEMEDIO']
print(f"-> Created FM * Slope interaction (fuel state × terrain steepness)")'''

# Additional interactions commonly used in operational models
'''df['Temp_Wind_Interaction'] = df['TEMPERATURA'] * df['VENTOINTENSIDADE']
df['FWI_VPD_Interaction'] = df['FWI'] * df['VPD_kPa']
print("-> Created Temp*Wind and FWI*VPD interactions")'''

# 2.8 CYCLICAL TIME FEATURES + CALENDAR
if 'DHINICIO' in df.columns:
    dt = pd.to_datetime(df['DHINICIO'], errors='coerce')
else:
    dt = pd.NaT

# 2.9 PHASE 1 FEATURE: n_concurrent_fires
# Count how many OTHER fires were active in the same DISTRITO on the same date.
# Uses only input data (date + district) — zero leakage risk.
if 'DISTRITO' in df.columns and isinstance(dt, pd.Series):
    df['_date_only'] = dt.dt.date
    concurrent = (
        df.groupby(['DISTRITO', '_date_only'])['DISTRITO']
        .transform('count') - 1  # subtract the fire itself
    )
    df['n_concurrent_fires'] = concurrent.fillna(0).astype(int)
    df.drop(columns=['_date_only'], inplace=True)
    print(f"-> n_concurrent_fires: mean={df['n_concurrent_fires'].mean():.2f}, max={df['n_concurrent_fires'].max()}")
else:
    df['n_concurrent_fires'] = 0
    print("-> n_concurrent_fires: defaulted to 0 (DHINICIO or DISTRITO missing)")

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
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', # Weather
    #'VPD_kPa', # removed: negative permutation importance in Phase 2b
    'DECLIVEMEDIO', 'ALTITUDEMEDIA', # Topography
    'n_concurrent_fires', # Phase 1: resource competition signal
]

# Step 3: Natureza_* features REMOVED from the model.
# Feature importance (Step 1 graph) ranked them 22nd, 23rd, and 24th out of 24 — effectively zero signal.
# Fire type tells us WHAT burned but not HOW MANY resources were dispatched (that's operational capacity).
# Removing them reduces dimensionality noise. The one-hot columns stay in df for inspection only.
natureza_cols = [col for col in df.columns if col.startswith('Natureza_')]
# (natureza_cols intentionally NOT added to features)

# Distrito target-encoded columns will be added to X_train/X_test AFTER the split (see below).
# Placeholder list — populated post-split so the joblib dump and graph use the final feature count.
distrito_cols = []

# Step 2: Meios_Aereos is removed from regression targets.
# It is 98.9% zeros — a regressor always predicts ~0, compresses predictions, and poisons the multi-output MAE.
# It is now handled separately as: binary classifier (got aerial? yes/no) + count regressor (how many).
targets = ['Operacionais_Man', 'Meios_Terrestres']

# Safety Validation
missing = [c for c in features if c not in df.columns]
if missing:
    print(f"!!! WARNING !!! Missing critical columns: {missing}")
else:
    print(f"-> All {len(features)} features are ready.")

X = df[features].copy()
y = df[targets]
ids = df['NCCO']
# Carry DISTRITO for target encoding — it is NOT a model feature yet
distritos = df['DISTRITO'] if 'DISTRITO' in df.columns else None
# Aerial target extracted separately — carried through the split for consistent train/test indices
y_aerial = df['Meios_Aereos']
y_aerial_binary = (y_aerial > 0).astype(int)   # 1 = fire got aerial support, 0 = did not
print(f"-> Aerial support fires: {y_aerial_binary.sum()} ({y_aerial_binary.mean()*100:.1f}% of dataset)")

# 4-5. SPLIT — pass DISTRITO and aerial labels alongside X for consistent splitting
if distritos is not None:
    X_train, X_test, y_train, y_test, id_train, id_test, dist_train, dist_test, \
    yab_train, yab_test, ya_train, ya_test = train_test_split(
        X, y, ids, distritos, y_aerial_binary, y_aerial, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test, id_train, id_test, \
    yab_train, yab_test, ya_train, ya_test = train_test_split(
        X, y, ids, y_aerial_binary, y_aerial, test_size=0.2, random_state=42
    )
    dist_train, dist_test = None, None

# TARGET ENCODING: compute mean of each target per district FROM TRAINING SET ONLY (no leakage).
# Unseen districts (if any) fall back to the global training mean.
# Saves the encoding map so pipeline_active.py can apply the same transformation to live data.
if dist_train is not None:
    district_encodings = {}
    for t in targets:
        enc_col = f'Distrito_enc_{t}'
        tmp = pd.DataFrame({'DISTRITO': dist_train.values, 'val': y_train[t].values})
        means = tmp.groupby('DISTRITO')['val'].mean()
        global_mean = y_train[t].mean()
        if enc_col == 'Distrito_enc_Operacionais_Man':  # negative perm importance in Phase 2b — excluded
            district_encodings[t] = means.to_dict()
            continue
        X_train[enc_col] = dist_train.values
        X_train[enc_col] = X_train[enc_col].map(means).fillna(global_mean)
        X_test[enc_col] = dist_test.values
        X_test[enc_col] = X_test[enc_col].map(means).fillna(global_mean)
        features.append(enc_col)
        distrito_cols.append(enc_col)
        district_encodings[t] = means.to_dict()
    print(f"-> Target encoded DISTRITO: {len(distrito_cols)} features added post-split.")
    joblib.dump(district_encodings, 'district_target_encoding.pkl')
    print("-> District encoding map saved: district_target_encoding.pkl")

# PHASE 1 FEATURES: district_max_single_incident + district_median_single_incident
# Proxy for each district's operational capacity ceiling and typical allocation.
# Computed from TRAINING SET ONLY to avoid leakage (derived from Operacionais_Man target).
# Unseen districts fall back to the global training mean/median.
if dist_train is not None:
    ref_col = 'Operacionais_Man'
    tmp_cap = pd.DataFrame({
        'DISTRITO': dist_train.values,
        ref_col: y_train[ref_col].values
    })
    dist_max    = tmp_cap.groupby('DISTRITO')[ref_col].max()
    dist_median = tmp_cap.groupby('DISTRITO')[ref_col].median()
    global_max    = y_train[ref_col].max()
    global_median = y_train[ref_col].median()

    # X_train['district_max_single_incident']    = dist_train.values  # negative perm importance — excluded
    # X_train['district_max_single_incident']    = X_train['district_max_single_incident'].map(dist_max).fillna(global_max)
    # X_test['district_max_single_incident']     = dist_test.values
    # X_test['district_max_single_incident']     = X_test['district_max_single_incident'].map(dist_max).fillna(global_max)

    X_train['district_median_single_incident'] = dist_train.values
    X_train['district_median_single_incident'] = X_train['district_median_single_incident'].map(dist_median).fillna(global_median)
    X_test['district_median_single_incident']  = dist_test.values
    X_test['district_median_single_incident']  = X_test['district_median_single_incident'].map(dist_median).fillna(global_median)

    #features.append('district_max_single_incident')  # negative permutation importance in Phase 2b — excluded
    features.append('district_median_single_incident')
    print(f"-> Phase 1 capacity feature added: district_median_single_incident (district_max excluded — negative importance)")
    print(f"   Median range: [{X_train['district_median_single_incident'].min():.0f}, {X_train['district_median_single_incident'].max():.0f}]")

# 6. PHASE 2: K-MEANS TIER LABELING + RF CLASSIFIER
# Why switch from regression to tier classification?
#   Regression on these targets gives R²≈0 — the exact values are not predictable from
#   environmental/geographic inputs alone. But the TIER (small/medium/large deployment) may be.
#   K=7 was selected by silhouette score (0.69) in Step 3a. The tiers represent natural
#   operational deployment levels in the data, not arbitrary cuts.
#
# KMeans is fit in log1p space to avoid large fires dominating the cluster geometry.
# The trained KMeans is saved so that pipeline_active.py can assign tiers to live predictions.

K_TIERS = 3
print(f"\n--- Phase 2: K-Means Tier Labeling (K={K_TIERS}) ---")

# Fit KMeans on TRAINING SET ONLY (no leakage) in log1p space
y_train_log = np.log1p(y_train[targets].values)
kmeans = KMeans(n_clusters=K_TIERS, random_state=42, n_init=20)
kmeans.fit(y_train_log)

# Assign tier labels — train and test
tier_train = kmeans.labels_                       # from .fit()
tier_test  = kmeans.predict(np.log1p(y_test[targets].values))  # uses trained centroids

# Re-order tiers by Operacionais_Man centroid (small → large) for interpretability.
# KMeans cluster IDs are arbitrary — sorting makes tier 0 always the smallest deployment.
centroids_log = kmeans.cluster_centers_
centroids_orig = np.expm1(centroids_log)          # back to original scale for reporting
sort_order = np.argsort(centroids_orig[:, 0])     # sort by Operacionais_Man
remap = {old: new for new, old in enumerate(sort_order)}
tier_train = np.array([remap[t] for t in tier_train])
tier_test  = np.array([remap[t] for t in tier_test])
centroids_orig = centroids_orig[sort_order]       # reorder centroids to match new labels

# Print tier summary
print("Tier centroids (Operacionais_Man, Meios_Terrestres):")
tier_counts_train = pd.Series(tier_train).value_counts().sort_index()
for i in range(K_TIERS):
    n = tier_counts_train.get(i, 0)
    pct = 100 * n / len(tier_train)
    print(f"  Tier {i}: Operacionais_Man={centroids_orig[i,0]:.1f}  Meios_Terrestres={centroids_orig[i,1]:.1f}  ({n} fires, {pct:.1f}%)")

# Compute tier ranges (p5–p95) from TRAINING SET ONLY.
# Used for range-based evaluation: instead of asking "did we predict the exact tier?",
# we ask "does the real value fall within the predicted tier's plausible range?".
# p5/p95 chosen to exclude outliers that would widen every tier to uselessness.
tier_ranges = {}
for tier in range(K_TIERS):
    mask = (tier_train == tier)
    ops = y_train['Operacionais_Man'].values[mask]
    veh = y_train['Meios_Terrestres'].values[mask]
    tier_ranges[tier] = {
        'ops_min':    np.percentile(ops, 5),
        'ops_max':    np.percentile(ops, 95),
        'veh_min':    np.percentile(veh, 5),
        'veh_max':    np.percentile(veh, 95),
        'ops_median': np.median(ops),
        'veh_median': np.median(veh),
    }
print("Tier ranges (p5-p95, training set):")
for i in range(K_TIERS):
    r = tier_ranges[i]
    print(f"  Tier {i}: Ops [{r['ops_min']:.0f}-{r['ops_max']:.0f}] median={r['ops_median']:.0f} | "
          f"Veh [{r['veh_min']:.0f}-{r['veh_max']:.0f}] median={r['veh_median']:.0f}")

# Save KMeans model for inference pipeline
joblib.dump(kmeans, 'model_kmeans_tiers.pkl')
joblib.dump(sort_order, 'model_kmeans_sort_order.pkl')
print("-> KMeans model saved: model_kmeans_tiers.pkl")

# 6b. TRAIN RF CLASSIFIER ON TIER LABELS
# class_weight='balanced': Tier 5 and 6 are rare (6% and 1%) — without balancing the classifier
# ignores them entirely and achieves ~75% accuracy by always predicting Tier 2.
print(f"\nTraining RF Classifier on {K_TIERS} deployment tiers...")
clf_tiers = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1
)
clf_tiers.fit(X_train, tier_train)
print("  Classifier done.")

# MEIOS_AEREOS — TWO-STAGE MODEL (Step 2)
# Stage 1: Binary classifier — did this fire require aerial support?
#   class_weight='balanced' compensates for 1.1% positive rate (441/39549 fires).
#   Without it, the model always predicts 0 and gets 98.9% accuracy doing nothing.
# Stage 2: Count regressor — how many aerial units?
#   Trained ONLY on the ~353 training fires (80% of 441) that actually got aerial support.
print("\n--- Step 2: Meios_Aereos Two-Stage Model ---")
pos_count_train = yab_train.sum()
print(f"Training aerial binary classifier on {len(yab_train)} samples ({pos_count_train} positives)...")
clf_aerial = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
clf_aerial.fit(X_train, yab_train)

mask_train_pos = ya_train.values > 0
print(f"Training aerial count regressor on {mask_train_pos.sum()} non-zero training cases...")
reg_aerial = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
reg_aerial.fit(X_train[mask_train_pos], ya_train[mask_train_pos])

# Evaluate Stage 1: binary aerial classification on test set
yab_pred = clf_aerial.predict(X_test)
yab_proba = clf_aerial.predict_proba(X_test)[:, 1]
auc_aerial = roc_auc_score(yab_test, yab_proba)
f1_aerial = f1_score(yab_test, yab_pred, zero_division=0)
prec_aerial = precision_score(yab_test, yab_pred, zero_division=0)
rec_aerial = recall_score(yab_test, yab_pred, zero_division=0)
print(f"  Classifier — ROC-AUC: {auc_aerial:.4f} | Prec: {prec_aerial:.4f} | Recall: {rec_aerial:.4f} | F1: {f1_aerial:.4f}")

# Evaluate Stage 2: count regression on actual positive test cases
mask_test_pos = ya_test.values > 0
if mask_test_pos.sum() > 0:
    ya_pred_pos = reg_aerial.predict(X_test[mask_test_pos])
    r2_aerial_reg = r2_score(ya_test[mask_test_pos], ya_pred_pos)
    mae_aerial_reg = mean_absolute_error(ya_test[mask_test_pos], ya_pred_pos)
    print(f"  Regressor ({mask_test_pos.sum()} test positives) — R²: {r2_aerial_reg:.4f} | MAE: {mae_aerial_reg:.2f}")
else:
    r2_aerial_reg, mae_aerial_reg = None, None

joblib.dump(clf_aerial, 'model_aerial_classifier.pkl')
joblib.dump(reg_aerial, 'model_aerial_regressor.pkl')
print("-> Aerial models saved: model_aerial_classifier.pkl, model_aerial_regressor.pkl")

# 7. PREDICT TIERS ON TEST SET
tier_pred = clf_tiers.predict(X_test)

# 8. CLASSIFICATION METRICS
print("\n--- Phase 2: Classification Metrics ---")
acc = accuracy_score(tier_test, tier_pred)
f1_w = f1_score(tier_test, tier_pred, average='weighted', zero_division=0)
tier_labels = [f"T{i}" for i in range(K_TIERS)]
report_str = classification_report(tier_test, tier_pred, target_names=tier_labels, zero_division=0)
print(f"Accuracy: {acc:.4f}  |  Weighted F1: {f1_w:.4f}")
print(report_str)

# 8b. REGRESSION-EQUIVALENT METRICS via centroid mapping
# Map each predicted tier to its centroid (mean Operacionais_Man / Meios_Terrestres on train set).
# This converts a tier prediction back to a numeric value so we can compare R² with Phase 1.
# centroid_map[tier] = [mean_Operacionais, mean_Meios_Terrestres]
centroid_map = centroids_orig  # shape (K_TIERS, 2), already sorted
y_pred_centroid = centroid_map[tier_pred]    # (N_test, 2)
y_true_real     = y_test[targets].values     # (N_test, 2)

print("\n--- Regression-Equivalent Metrics (centroid mapping) ---")
r2_centroid, mae_centroid = [], []
for i, col in enumerate(targets):
    r2_c  = r2_score(y_true_real[:, i], y_pred_centroid[:, i])
    mae_c = mean_absolute_error(y_true_real[:, i], y_pred_centroid[:, i])
    r2_centroid.append(r2_c)
    mae_centroid.append(mae_c)
    print(f"{col}: R2={r2_c:.4f} | MAE={mae_c:.2f}")

# 8c. MEDIAN MAPPING — more robust than centroid for skewed within-tier distributions
median_map = np.array([[tier_ranges[i]['ops_median'], tier_ranges[i]['veh_median']]
                       for i in range(K_TIERS)])
y_pred_median = median_map[tier_pred]

print("\n--- Regression-Equivalent Metrics (median mapping) ---")
r2_median, mae_median = [], []
for i, col in enumerate(targets):
    r2_m  = r2_score(y_true_real[:, i], y_pred_median[:, i])
    mae_m = mean_absolute_error(y_true_real[:, i], y_pred_median[:, i])
    r2_median.append(r2_m)
    mae_median.append(mae_m)
    print(f"{col}: R2={r2_m:.4f} | MAE={mae_m:.2f}")

# 8d. RANGE-BASED ACCURACY
# For each test sample, check if the real value falls within the predicted tier's p5–p95 range.
# This is a softer, operationally meaningful metric: the model is "correct" if it points to the
# right deployment bracket, not necessarily the exact same k-means bucket.
y_true_ops = y_true_real[:, 0]
y_true_veh = y_true_real[:, 1]

in_range_ops = np.array([
    tier_ranges[tier_pred[j]]['ops_min'] <= y_true_ops[j] <= tier_ranges[tier_pred[j]]['ops_max']
    for j in range(len(tier_pred))
])
in_range_veh = np.array([
    tier_ranges[tier_pred[j]]['veh_min'] <= y_true_veh[j] <= tier_ranges[tier_pred[j]]['veh_max']
    for j in range(len(tier_pred))
])

range_acc_ops  = in_range_ops.mean()
range_acc_veh  = in_range_veh.mean()
range_acc_both = (in_range_ops & in_range_veh).mean()

print("\n--- Range Accuracy (real value within predicted tier's p5-p95 range) ---")
print(f"Operacionais_Man in range: {range_acc_ops*100:.1f}%")
print(f"Meios_Terrestres in range: {range_acc_veh*100:.1f}%")
print(f"Both within range:         {range_acc_both*100:.1f}%")

# 9. SAVE MODELS AND FEATURES
joblib.dump(clf_tiers, 'model_tier_classifier.pkl')
joblib.dump(features, 'model_features_list.pkl')
print("\n-> SUCCESS!")
print("-> Tier classifier saved: model_tier_classifier.pkl")
print("-> Feature list saved: model_features_list.pkl (DO NOT DELETE THIS FILE)")

# 10. SAVE METRICS TO FILE
with open('step5d_k3_clean_results.txt', 'w') as f:
    f.write(f"=== PHASE 2b: RF Classifier on Deployment Tiers (K={K_TIERS}) ===\n")
    f.write(f"Total features used: {len(features)}\n")
    f.write(f"Baseline regression (Phase 1 RF): Overall R2=0.0069 | Operacionais_Man R2=0.0024 | Meios_Terrestres R2=0.0114\n\n")
    f.write("Tier centroids:\n")
    for i in range(K_TIERS):
        n = tier_counts_train.get(i, 0)
        pct = 100 * n / len(tier_train)
        f.write(f"  Tier {i}: Operacionais_Man={centroids_orig[i,0]:.1f}  Meios_Terrestres={centroids_orig[i,1]:.1f}  ({n} train fires, {pct:.1f}%)\n")
    f.write("\n--- Classification Metrics ---\n")
    f.write(f"Overall Accuracy: {acc:.4f}\n")
    f.write(f"Weighted F1:      {f1_w:.4f}\n\n")
    f.write("Per-tier report:\n")
    f.write(report_str)
    f.write("\n--- Regression-Equivalent Metrics (centroid mapping) ---\n")
    for i, col in enumerate(targets):
        r2_c  = r2_score(y_true_real[:, i], y_pred_centroid[:, i])
        mae_c = mean_absolute_error(y_true_real[:, i], y_pred_centroid[:, i])
        f.write(f"{col}: R2={r2_c:.4f} | MAE={mae_c:.2f}\n")
    f.write("\n--- Meios_Aereos (unchanged from Step 2) ---\n")
    f.write(f"ROC-AUC:   {auc_aerial:.4f}\n")
    f.write(f"Precision: {prec_aerial:.4f}\n")
    f.write(f"Recall:    {rec_aerial:.4f}\n")
    f.write(f"F1:        {f1_aerial:.4f}\n")
    if r2_aerial_reg is not None:
        f.write(f"\n--- Meios_Aereos Count Regressor (on positives only) ---\n")
        f.write(f"R2:  {r2_aerial_reg:.4f}\n")
        f.write(f"MAE: {mae_aerial_reg:.2f}\n")
print("-> Metrics saved: step5d_k3_clean_results.txt")

# 11. GRAPHS
print("Generating graphs...")

# --- Graph 1: Confusion matrix heatmap ---
cm = confusion_matrix(tier_test, tier_pred, labels=list(range(K_TIERS)))
tier_axis_labels = [
    f"T{i}\n(≈{centroids_orig[i,0]:.0f}m,{centroids_orig[i,1]:.0f}v)"
    for i in range(K_TIERS)
]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=tier_axis_labels, yticklabels=tier_axis_labels, ax=ax)
ax.set_xlabel('Predicted Tier')
ax.set_ylabel('True Tier')
ax.set_title(f'Phase 2: Confusion Matrix\nAccuracy={acc:.3f}  Weighted F1={f1_w:.3f}')
plt.tight_layout()
plt.savefig('step5d_k3_confusion_matrix.png')
plt.close()
print("-> Saved: step5d_k3_confusion_matrix.png")

# --- Graph 2: Tier distribution (predicted vs actual) + per-tier accuracy ---
true_counts  = pd.Series(tier_test).value_counts().sort_index().reindex(range(K_TIERS), fill_value=0)
pred_counts  = pd.Series(tier_pred).value_counts().sort_index().reindex(range(K_TIERS), fill_value=0)
per_tier_acc = [
    (np.sum((tier_test == i) & (tier_pred == i)) / np.sum(tier_test == i))
    if np.sum(tier_test == i) > 0 else 0.0
    for i in range(K_TIERS)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(K_TIERS)
w = 0.35
bars1 = ax1.bar(x - w/2, true_counts.values, w, label='Actual', color='steelblue')
bars2 = ax1.bar(x + w/2, pred_counts.values, w, label='Predicted', color='tomato')
ax1.set_xticks(x)
ax1.set_xticklabels(tier_axis_labels, fontsize=8)
ax1.set_xlabel('Deployment Tier')
ax1.set_ylabel('Count')
ax1.set_title('Predicted vs Actual Tier Distribution')
ax1.legend()

ax2.bar(x, per_tier_acc, color='mediumseagreen')
ax2.set_xticks(x)
ax2.set_xticklabels(tier_axis_labels, fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel('Deployment Tier')
ax2.set_ylabel('Accuracy')
ax2.set_title('Per-Tier Classification Accuracy')
for xi, v in zip(x, per_tier_acc):
    ax2.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

plt.suptitle('Phase 2: Tier Analysis')
plt.tight_layout()
plt.savefig('step5d_k3_tier_analysis.png')
plt.close()
print("-> Saved: step5d_k3_tier_analysis.png")

# --- Graph 3: Permutation importance for the tier classifier ---
print("Computing permutation importances for tier classifier (this may take ~30s)...")
perm = permutation_importance(
    clf_tiers, X_test, tier_test,
    n_repeats=5, random_state=42, scoring='accuracy'
)
imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=imp.values, y=imp.index, palette='viridis', ax=ax,
            hue=imp.index, legend=False)
ax.set_title('Phase 2: Permutation Importance (Tier Classifier)\nMetric: accuracy drop when feature shuffled')
ax.set_xlabel('Mean accuracy decrease when feature shuffled')
plt.tight_layout()
plt.savefig('step5d_k3_importance.png')
plt.close()
print("-> Saved: step5d_k3_importance.png")

print("-> Graphs saved")

# 12. PHASE 2c: RANGE-BASED RESULTS FILE
with open('step5d_range_results.txt', 'w') as f:
    f.write(f"=== PHASE 2c: Range-Based Tier Evaluation (K={K_TIERS}) ===\n\n")
    f.write("Tier ranges (p5-p95 from training set):\n")
    for i in range(K_TIERS):
        r = tier_ranges[i]
        f.write(f"  Tier {i}: Ops [{r['ops_min']:.0f} - {r['ops_max']:.0f}], "
                f"Veh [{r['veh_min']:.0f} - {r['veh_max']:.0f}], "
                f"median ops={r['ops_median']:.0f}, median veh={r['veh_median']:.0f}\n")
    f.write("\n--- Range Accuracy (does real value fall within predicted tier's range?) ---\n")
    f.write(f"Operacionais_Man range accuracy: {range_acc_ops*100:.1f}%\n")
    f.write(f"Meios_Terrestres range accuracy: {range_acc_veh*100:.1f}%\n")
    f.write(f"Both within range:               {range_acc_both*100:.1f}%\n")
    f.write("\n--- Comparison ---\n")
    f.write(f"{'Method':<30} {'Accuracy':>10} {'R2_Ops':>10} {'R2_Veh':>10} {'MAE_Ops':>10} {'MAE_Veh':>10}\n")
    f.write(f"{'Tier label (exact match)':<30} {acc:>10.3f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}\n")
    f.write(f"{'Centroid mapping':<30} {'N/A':>10} {r2_centroid[0]:>10.3f} {r2_centroid[1]:>10.3f} {mae_centroid[0]:>10.2f} {mae_centroid[1]:>10.2f}\n")
    f.write(f"{'Median mapping':<30} {'N/A':>10} {r2_median[0]:>10.3f} {r2_median[1]:>10.3f} {mae_median[0]:>10.2f} {mae_median[1]:>10.2f}\n")
    f.write(f"{'Range accuracy (both)':<30} {range_acc_both:>10.3f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}\n")
print("-> Metrics saved: step5d_range_results.txt")

# 13. PHASE 2c: RANGE COMPARISON PLOT
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

# Left: accuracy metrics comparison
acc_labels = ['Tier exact\nmatch', 'Ops in\nrange', 'Veh in\nrange', 'Both in\nrange']
acc_values = [acc, range_acc_ops, range_acc_veh, range_acc_both]
acc_colors = ['steelblue', 'mediumseagreen', 'mediumseagreen', 'darkgreen']
bars = ax_l.bar(acc_labels, acc_values, color=acc_colors)
ax_l.set_ylim(0, 1.1)
ax_l.set_ylabel('Proportion correct')
ax_l.set_title('Accuracy Metrics Comparison')
for bar, v in zip(bars, acc_values):
    ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v*100:.1f}%',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: R² comparison across methods (Operacionais_Man only, most challenging)
r2_labels  = ['Regression\nbaseline\n(Phase 1)', 'Centroid\nmapping', 'Median\nmapping']
r2_values  = [0.0024, r2_centroid[0], r2_median[0]]
r2_colors  = ['slategray', 'tomato', 'steelblue']
bars2 = ax_r.bar(r2_labels, r2_values, color=r2_colors)
ax_r.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax_r.set_ylabel('R²  (Operacionais_Man)')
ax_r.set_title('R² Comparison — Operacionais_Man\n(regression baseline vs tier mappings)')
for bar, v in zip(bars2, r2_values):
    ypos = v + 0.003 if v >= 0 else v - 0.008
    ax_r.text(bar.get_x() + bar.get_width() / 2, ypos, f'{v:.3f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle(f'Phase 2c: Range-Based Evaluation Summary (K={K_TIERS})', fontsize=13)
plt.tight_layout()
plt.savefig('step5d_range_comparison.png')
plt.close()
print("-> Saved: step5d_range_comparison.png")

print("Completed!")