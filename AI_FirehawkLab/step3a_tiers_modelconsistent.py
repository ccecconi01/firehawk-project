# Adopted three-tier (k=3) figure for thesis 5.3.1 — model-consistent with the pipeline.
# Reuses step_temporal_validation.py's fitted KMeans tier labels and centroids (train
# 2019-2022, log1p targets, NO StandardScaler, k=3, SEED=42), so the figure matches the
# adopted tiers exactly: T0/T1/T2 = 21.0/54.6/24.4 %, centroids ~2.0/5.4/12.1.
# Two panels only; elbow/silhouette live on the exploratory k=7 figure
# (step3b_kmeans_k7_comparison.py), never here.
# NOTE: step3a_kmeans.py is a DIFFERENT, deprecated exploratory version (full dataset +
# StandardScaler -> 55/33/13). Do not confuse the two.
# Run from AI_FirehawkLab:  python step3a_tiers_modelconsistent.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import step_temporal_validation as stv

TARGETS = stv.TARGETS
K = stv.K_TIERS
OUT = os.path.join(stv.OUT_DIR, 'tiers_kmeans_modelconsistent.png')
LABELS = ['Minimal', 'Standard', 'Reinforced']

y = stv.y_train[TARGETS].values
y_log = np.log1p(y)
tiers = stv.tier_train             # k=3 labels, already sorted small -> large
centroids = stv.centroids_orig     # original-scale centroids, sorted
centroids_log = np.log1p(centroids)

counts = np.bincount(tiers, minlength=K)
pct = 100.0 * counts / counts.sum()

print(f"\nAdopted k={K} tiers (train 2019-2022, log1p, SEED={stv.SEED}):")
for i in range(K):
    print(f"  T{i} ({LABELS[i]}): personnel={centroids[i,0]:.2f}  vehicles={centroids[i,1]:.2f}  "
          f"n={counts[i]}  ({pct[i]:.1f}%)")

sns.set_theme(style='whitegrid', font_scale=1.1)
palette = sns.color_palette('viridis', K)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# (a) scatter in log1p target space; n and % per tier in the legend
for i in range(K):
    m = tiers == i
    ax1.scatter(y_log[m, 0], y_log[m, 1], s=12, alpha=0.35, color=palette[i],
                label=f'T{i} {LABELS[i]} (n={counts[i]}, {pct[i]:.1f}%)')
ax1.scatter(centroids_log[:, 0], centroids_log[:, 1], marker='X', s=240, c='red',
            edgecolor='black', linewidths=1.2, zorder=5, label='centroids')
ax1.set_xlabel('log1p(Operacionais_Man)')
ax1.set_ylabel('log1p(Meios_Terrestres)')
ax1.set_title('Tiers in target space (log1p)')
ax1.legend(fontsize=9, loc='upper left')

# (b) boxplot of Operacionais_Man per tier, ORIGINAL scale
dfb = pd.DataFrame({'tier': [f'T{t}' for t in tiers], 'Operacionais_Man': y[:, 0]})
order = [f'T{i}' for i in range(K)]
sns.boxplot(data=dfb, x='tier', y='Operacionais_Man', order=order, hue='tier',
            palette='viridis', legend=False, showfliers=False, ax=ax2)
ax2.set_xlabel('Tier')
ax2.set_ylabel('Operacionais_Man (original scale)')
ax2.set_title('Operacionais_Man distribution by tier')

plt.suptitle('Adopted three-tier definition (train 2019-2022, log1p, k=3)')
plt.tight_layout()
fig.savefig(OUT, dpi=120)
plt.close(fig)
print(f"-> saved {OUT}")
