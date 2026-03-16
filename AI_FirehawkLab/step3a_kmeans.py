import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# ============================================================
# STEP 3a: K-Means Exploration
# ------------------------------------------------------------
# PART A: K-Means on FEATURES
#   Question: do fires naturally cluster by weather/geography?
#   And if so, do those clusters map to different resource levels?
#   This answers whether classification has a feature-space basis.
#
# PART B: K-Means on TARGETS
#   Question: are there natural "tiers" in resource deployment
#   (Operacionais_Man + Meios_Terrestres) that are data-driven?
#   These centroids become the tier boundaries for Step 3b.
# ============================================================

# ------ 1. LOAD & PREPROCESS (mirrors trainlightmodel.py) ------
print("Loading data...")
df = pd.read_csv('dataset_final_clean.csv')
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

if 'Hora' in df.columns:
    df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M', errors='coerce').dt.hour

df['FWI_Wind_Interaction'] = df['FWI'] * df['VENTOINTENSIDADE']
df['FM_Slope_Interaction']  = df['FM']  * df['DECLIVEMEDIO']

# 18 base features — district encodings require a split so we skip them here.
# The purpose of this script is exploratory, not predictive.
base_features = [
    'LAT', 'LON',
    'Mes', 'Hora',
    'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FM',
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE', 'VPD_kPa',
    'DECLIVEMEDIO', 'ALTITUDEMEDIA',
    'FWI_Wind_Interaction', 'FM_Slope_Interaction',
]
targets = ['Operacionais_Man', 'Meios_Terrestres']

# Drop rows where any feature OR target is NaN — K-Means needs complete rows.
df_clean = df[base_features + targets].dropna()
print(f"Clean rows for clustering: {len(df_clean)} / {len(df)}")

X_raw = df_clean[base_features].values
y_raw = df_clean[targets].values

# Log-transform targets: both are heavily right-skewed (most fires: 1-5 resources,
# rare fires: 50-378). Log space makes the clustering distance metric meaningful —
# a jump from 2→5 units is comparable to 20→50 units.
y_log = np.log1p(y_raw)

K_RANGE = range(2, 8)   # test K=2 through 7

# ------ 2. PART A: K-Means on FEATURES ------
print("\n--- Part A: K-Means on Features ---")

# Scale features: K-Means uses Euclidean distance — unscaled features would let
# large-magnitude columns (e.g. TEMPERATURA in °C) dominate over small ones (FM 0-1).
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)

inertias_X, silhouettes_X = [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias_X.append(km.inertia_)
    # sample_size speeds up silhouette on 39k rows without losing accuracy
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    silhouettes_X.append(sil)
    print(f"  K={k}: inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

best_k_X = list(K_RANGE)[int(np.argmax(silhouettes_X))]
print(f"Best K (features): {best_k_X}  (silhouette={max(silhouettes_X):.4f})")

km_X_final = KMeans(n_clusters=best_k_X, init='k-means++', n_init=20, random_state=42)
labels_X = km_X_final.fit_predict(X_scaled)

# PCA → 2D for scatter visualisation.
# PCA is applied to scaled X so the projection preserves the same distances K-Means used.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_.sum() * 100

# Mean resource deployment per feature cluster — the KEY diagnostic:
# if cluster 0 averages 3 personnel and cluster 2 averages 25, features DO separate tiers.
# if all clusters average ~4 personnel, features carry no classification signal.
cluster_df = pd.DataFrame({
    'cluster':           labels_X,
    'Operacionais_Man':  y_raw[:, 0],
    'Meios_Terrestres':  y_raw[:, 1],
})
feat_cluster_means = cluster_df.groupby('cluster')[targets].mean()
feat_cluster_sizes = cluster_df.groupby('cluster').size().rename('n_fires')
print("\nMean targets per feature cluster:")
print(pd.concat([feat_cluster_means, feat_cluster_sizes], axis=1).to_string())

# ------ 3. PART B: K-Means on TARGETS ------
print("\n--- Part B: K-Means on Targets ---")

# Scale log-targets for the same reason as features.
scaler_y = StandardScaler()
y_log_scaled = scaler_y.fit_transform(y_log)

inertias_y, silhouettes_y = [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(y_log_scaled)
    inertias_y.append(km.inertia_)
    sil = silhouette_score(y_log_scaled, labels, random_state=42)
    silhouettes_y.append(sil)
    print(f"  K={k}: inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

best_k_y = list(K_RANGE)[int(np.argmax(silhouettes_y))]
print(f"Best K (targets): {best_k_y}  (silhouette={max(silhouettes_y):.4f})")

km_y_final = KMeans(n_clusters=best_k_y, init='k-means++', n_init=20, random_state=42)
labels_y = km_y_final.fit_predict(y_log_scaled)

# Centroids back to original scale: inverse of log1p is expm1.
# scaler_y.inverse_transform undoes StandardScaler, then expm1 undoes log1p.
centroids_log_std = km_y_final.cluster_centers_
centroids_log     = scaler_y.inverse_transform(centroids_log_std)
centroids_orig    = np.expm1(centroids_log)

# Sort clusters by Operacionais_Man centroid so labels are ordered Small→Large
sort_order  = np.argsort(centroids_orig[:, 0])
label_map   = {old: new for new, old in enumerate(sort_order)}
labels_y_sorted    = np.array([label_map[l] for l in labels_y])
centroids_orig_sorted = centroids_orig[sort_order]

print("\nTarget cluster centroids (original scale, sorted by Operacionais_Man):")
for i, c in enumerate(centroids_orig_sorted):
    n = (labels_y_sorted == i).sum()
    print(f"  Tier {i} ({n} fires, {n/len(labels_y_sorted)*100:.1f}%): "
          f"Operacionais_Man≈{c[0]:.1f}  Meios_Terrestres≈{c[1]:.1f}")

# Tier stats (percentiles — useful for setting hard boundaries in Step 3b)
tier_stats = pd.DataFrame({
    'tier':              labels_y_sorted,
    'Operacionais_Man':  y_raw[:, 0],
    'Meios_Terrestres':  y_raw[:, 1],
})
print("\nOperacionais_Man per tier (percentiles):")
print(tier_stats.groupby('tier')['Operacionais_Man']
      .describe(percentiles=[.25, .5, .75, .90]).to_string())
print("\nMeios_Terrestres per tier (percentiles):")
print(tier_stats.groupby('tier')['Meios_Terrestres']
      .describe(percentiles=[.25, .5, .75, .90]).to_string())

# ------ 4. GRAPHS ------
print("\nGenerating graphs...")
palette = sns.color_palette('tab10', n_colors=max(best_k_X, best_k_y))

# ---- FIGURE 1: Feature K-Means ----
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14))

# [0,0] Elbow curve
ax = axes1[0, 0]
ax.plot(list(K_RANGE), inertias_X, 'o-', color='royalblue')
ax.axvline(best_k_X, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k_X}')
ax.set_title('Elbow Curve — Feature Space')
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Inertia (within-cluster sum of squares)')
ax.legend()

# [0,1] Silhouette scores
ax = axes1[0, 1]
ax.plot(list(K_RANGE), silhouettes_X, 's-', color='seagreen')
ax.axvline(best_k_X, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k_X}')
ax.set_title('Silhouette Score — Feature Space')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score (higher = more distinct clusters)')
ax.legend()

# [1,0] PCA 2D scatter coloured by feature cluster
ax = axes1[1, 0]
for c in range(best_k_X):
    mask = labels_X == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=4, alpha=0.3, color=palette[c], label=f'Cluster {c}')
ax.set_title(f'PCA Projection of Feature Clusters (K={best_k_X})\n'
             f'({var_explained:.1f}% variance explained by PC1+PC2)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(markerscale=3)

# [1,1] Mean target per feature cluster
# If bars are similar height → feature clusters don't separate resource tiers.
# If bars vary significantly → feature space does carry classification signal.
ax = axes1[1, 1]
x_pos = np.arange(best_k_X)
w = 0.35
bars1 = ax.bar(x_pos - w/2, feat_cluster_means['Operacionais_Man'],
               width=w, label='Operacionais_Man', color='royalblue', alpha=0.8)
bars2 = ax.bar(x_pos + w/2, feat_cluster_means['Meios_Terrestres'],
               width=w, label='Meios_Terrestres', color='seagreen', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Cluster {c}\n(n={feat_cluster_sizes[c]:,})' for c in range(best_k_X)])
ax.set_title(f'Mean Resource Deployment per Feature Cluster\n'
             f'(Key diagnostic: do feature clusters predict resource tiers?)')
ax.set_ylabel('Mean count (original scale)')
ax.legend()

fig1.suptitle('Step 3a — Part A: K-Means on Features', fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig('step3a_features_kmeans.png', dpi=120)
plt.close(fig1)
print("-> Saved: step3a_features_kmeans.png")

# ---- FIGURE 2: Target K-Means ----
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

# [0,0] Elbow curve
ax = axes2[0, 0]
ax.plot(list(K_RANGE), inertias_y, 'o-', color='darkorange')
ax.axvline(best_k_y, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k_y}')
ax.set_title('Elbow Curve — Target Space')
ax.set_xlabel('K')
ax.set_ylabel('Inertia')
ax.legend()

# [0,1] Silhouette scores
ax = axes2[0, 1]
ax.plot(list(K_RANGE), silhouettes_y, 's-', color='tomato')
ax.axvline(best_k_y, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k_y}')
ax.set_title('Silhouette Score — Target Space')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')
ax.legend()

# [1,0] 2D scatter: log(Operacionais_Man) vs log(Meios_Terrestres), coloured by tier.
# Centroids marked with X. Log scale because original scale is too compressed near origin.
ax = axes2[1, 0]
for c in range(best_k_y):
    mask = labels_y_sorted == c
    n_c  = mask.sum()
    ax.scatter(y_log[mask, 0], y_log[mask, 1],
               s=6, alpha=0.3, color=palette[c],
               label=f'Tier {c} (n={n_c:,}, {n_c/len(labels_y_sorted)*100:.0f}%)')
# Mark centroids (in log space)
centroids_log_sorted = np.log1p(centroids_orig_sorted)
ax.scatter(centroids_log_sorted[:, 0], centroids_log_sorted[:, 1],
           s=200, marker='X', color='black', zorder=5, label='Centroids')
for i, c in enumerate(centroids_orig_sorted):
    ax.annotate(f'Tier {i}\n({c[0]:.0f} men\n{c[1]:.1f} veh)',
                xy=(centroids_log_sorted[i, 0], centroids_log_sorted[i, 1]),
                xytext=(8, 8), textcoords='offset points', fontsize=8)
ax.set_title(f'Target Clusters (K={best_k_y}) — log scale\n'
             f'Each tier = natural resource deployment level')
ax.set_xlabel('log1p(Operacionais_Man)')
ax.set_ylabel('log1p(Meios_Terrestres)')
ax.legend(markerscale=2, fontsize=8)

# [1,1] Box plots of Operacionais_Man per tier
ax = axes2[1, 1]
tier_data = [y_raw[labels_y_sorted == c, 0] for c in range(best_k_y)]
bp = ax.boxplot(tier_data, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], palette[:best_k_y]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(
    [f'Tier {c}\n≈{centroids_orig_sorted[c,0]:.0f} men' for c in range(best_k_y)]
)
ax.set_title('Operacionais_Man Distribution per Tier\n(outliers hidden for readability)')
ax.set_ylabel('Operacionais_Man (original scale)')

fig2.suptitle('Step 3a — Part B: K-Means on Targets', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig('step3a_targets_kmeans.png', dpi=120)
plt.close(fig2)
print("-> Saved: step3a_targets_kmeans.png")

# ------ 5. SAVE RESULTS TEXT ------
with open('step3a_kmeans_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== STEP 3a: K-Means Exploration ===\n\n")

    f.write("--- Part A: K-Means on Features ---\n")
    f.write(f"Best K: {best_k_X}  (silhouette={max(silhouettes_X):.4f})\n")
    f.write(f"PCA variance explained by PC1+PC2: {var_explained:.1f}%\n\n")
    f.write("Mean resource deployment per feature cluster:\n")
    f.write(pd.concat([feat_cluster_means.round(2), feat_cluster_sizes], axis=1).to_string())
    f.write("\n\nInterpretation: if cluster means for Operacionais_Man are similar, feature-space\n")
    f.write("clusters do NOT separate resource tiers. If they differ significantly, they do.\n\n")

    f.write("--- Part B: K-Means on Targets ---\n")
    f.write(f"Best K: {best_k_y}  (silhouette={max(silhouettes_y):.4f})\n\n")
    f.write("Cluster centroids (original scale, sorted Small → Large):\n")
    for i, c in enumerate(centroids_orig_sorted):
        n = (labels_y_sorted == i).sum()
        f.write(f"  Tier {i} ({n} fires, {n/len(labels_y_sorted)*100:.1f}%): "
                f"Operacionais_Man={c[0]:.1f}  Meios_Terrestres={c[1]:.1f}\n")
    f.write("\nOperacionais_Man per tier (percentiles):\n")
    f.write(tier_stats.groupby('tier')['Operacionais_Man']
            .describe(percentiles=[.25, .5, .75, .90]).to_string())
    f.write("\n\nMeios_Terrestres per tier (percentiles):\n")
    f.write(tier_stats.groupby('tier')['Meios_Terrestres']
            .describe(percentiles=[.25, .5, .75, .90]).to_string())
    f.write("\n\nThese centroids and percentile boundaries inform the tier thresholds for Step 3b.\n")

print("-> Saved: step3a_kmeans_results.txt")
print("\nCompleted! Review the graphs and results file before proceeding to Step 3b.")
