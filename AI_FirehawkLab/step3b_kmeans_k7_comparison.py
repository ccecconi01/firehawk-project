# Exploratory K-Means cluster-selection figure for thesis 5.3.1.
# Same base as the adopted k=3 tiers (train 2019-2022, log1p targets, NO StandardScaler,
# KMeans SEED=42), reusing step_temporal_validation.py's y_train so the two figures are
# directly comparable. Shows why k=3 is adopted over the silhouette-optimal k=7.
#
# Layout 2x2:
#   top    : elbow curve + silhouette score over k=2..7 (the k-selection diagnostics);
#   bottom : k=7 cluster scatter (log1p, centroids marked) + Operacionais_Man boxplot.
# The elbow/silhouette curves live ONLY here, not on the adopted k=3 figure.
# Run from AI_FirehawkLab:  python step3b_kmeans_k7_comparison.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import step_temporal_validation as stv

TARGETS = stv.TARGETS
SEED = stv.SEED
ADOPTED_K = stv.K_TIERS   # 3
K = 7
OUT = os.path.join(stv.OUT_DIR, 'kmeans_k7_selection.png')

y = stv.y_train[TARGETS].values
y_log = np.log1p(y)

# Elbow + silhouette over k=2..7 (same train/log1p basis, same KMeans config).
ks = list(range(2, 8))
inertias, silhouettes, models = [], [], {}
for k in ks:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(y_log)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(y_log, km.labels_, sample_size=10000, random_state=SEED))
    models[k] = km
sil_best_k = ks[int(np.argmax(silhouettes))]
print("Silhouette over k=2..7: " + ", ".join(f"k={k}:{s:.4f}" for k, s in zip(ks, silhouettes)))
print(f"silhouette-optimal k = {sil_best_k} | adopted k = {ADOPTED_K}")

# k=7 clusters for the bottom panels, sorted by Operacionais_Man centroid.
km7 = models[K]
cen_log = km7.cluster_centers_
cen = np.expm1(cen_log)
order = np.argsort(cen[:, 0])
remap = {o: n for n, o in enumerate(order)}
lab7 = np.array([remap[l] for l in km7.labels_])
cen_log, cen = cen_log[order], cen[order]
counts = np.bincount(lab7, minlength=K)
pct = 100.0 * counts / counts.sum()
print(f"k={K} centroids / frequency:")
for i in range(K):
    print(f"  C{i}: personnel={cen[i,0]:.2f}  vehicles={cen[i,1]:.2f}  n={counts[i]}  ({pct[i]:.1f}%)")

sns.set_theme(style='whitegrid', font_scale=1.05)
palette = sns.color_palette('viridis', K)
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
(axE, axS), (axSc, axBx) = axs

# top-left: elbow
axE.plot(ks, inertias, 'o-', color='darkorange')
axE.axvline(sil_best_k, color='gray', linestyle=':', alpha=0.8, label=f'Silhouette max K={sil_best_k}')
axE.axvline(ADOPTED_K, color='green', linestyle='--', alpha=0.85, label=f'Adopted K={ADOPTED_K}')
axE.set_xlabel('k'); axE.set_ylabel('Inertia'); axE.set_title('Elbow curve'); axE.legend()

# top-right: silhouette
axS.plot(ks, silhouettes, 's-', color='tomato')
axS.axvline(sil_best_k, color='gray', linestyle=':', alpha=0.8, label=f'Silhouette max K={sil_best_k}')
axS.axvline(ADOPTED_K, color='green', linestyle='--', alpha=0.85, label=f'Adopted K={ADOPTED_K}')
axS.set_xlabel('k'); axS.set_ylabel('Silhouette score'); axS.set_title('Silhouette score'); axS.legend()

# bottom-left: k=7 scatter
for i in range(K):
    m = lab7 == i
    axSc.scatter(y_log[m, 0], y_log[m, 1], s=12, alpha=0.35, color=palette[i], label=f'C{i}')
axSc.scatter(cen_log[:, 0], cen_log[:, 1], marker='X', s=200, c='red',
             edgecolor='black', linewidths=1.2, zorder=5, label='centroids')
axSc.set_xlabel('log1p(Operacionais_Man)'); axSc.set_ylabel('log1p(Meios_Terrestres)')
axSc.set_title(f'K-Means k={K} clusters (log1p)'); axSc.legend(fontsize=8, ncol=2, loc='upper left')

# bottom-right: boxplot
dfb = pd.DataFrame({'cluster': lab7, 'Operacionais_Man': y[:, 0]})
sns.boxplot(data=dfb, x='cluster', y='Operacionais_Man', hue='cluster',
            palette='viridis', legend=False, showfliers=False, ax=axBx)
axBx.set_xlabel('Cluster (ordered small -> large)'); axBx.set_ylabel('Operacionais_Man')
axBx.set_title(f'Operacionais_Man by cluster (k={K})')

plt.suptitle('K-Means cluster selection (train 2019-2022, log1p): silhouette-optimal k=7 vs adopted k=3', fontsize=14)
plt.tight_layout()
fig.savefig(OUT, dpi=120)
plt.close(fig)
print(f"-> saved {OUT}")
