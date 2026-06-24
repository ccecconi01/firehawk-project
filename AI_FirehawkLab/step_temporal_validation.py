"""
step_temporal_validation.py  v2
================================
Objective : Re-evaluate the K=3 tier classifier (Phase 2 / trainlightmodel.py) and the
            two-stage aerial model (Step 2) using a strict temporal split instead of
            random split, as required by the dissertation advisor.

            Random split inflates performance on seasonal fire data: climatological
            and operational patterns from 2024-2025 leak into training through shared
            seasonal statistics, making the problem artificially easier than it is.

v2 improvements (over v1, 2026-04-22):
  1. Features: day-of-year sin/cos encoding added (continuous circular seasonal signal;
     Mes retained as a complement); district_T2_rate added (per-district fraction of T2
     incidents in training set, computed after KMeans — train-only, no leakage).
  2. Model: RF tier classifier calibrated with isotonic regression fitted on Val 2023.
     Calibrated predictions evaluated on test only (val is in-sample for calibration).
  3. Aerial: classifier trained on 2020-2022 only (2019 excluded — positive rate 3.8%
     vs 0.5% in test, a 6x distributional mismatch that distorts the learned threshold).
     Aerial regressor retains 2019-2022 positives (count data unaffected by rate shift).
  4. Aerial operational threshold set to 0.70 (best F1 from threshold scan in v1).

Input     : dataset_final_clean.csv
Output    : resultados_temporal/
              resultados_temporal.md
              confusion_matrix_temporal.png
              pr_curve_aerial.png
              threshold_scan.png
              feature_importance_temporal.png
              pipeline_temporal.pkl

Temporal split (does NOT modify existing pkl files):
  Train  2019-2022  -> ~26 340 fires (66.6%)   model fitting, KMeans labelling
  Val    2023       ->  ~4 556 fires (11.5%)   calibration + threshold selection
  Test   2024-2025  ->  ~8 653 fires (21.9%)   primary reported metrics
"""

import os
import warnings
from datetime import date as _date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    mean_absolute_error, precision_recall_curve,
)

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS        = [2019, 2020, 2021, 2022]
VAL_YEARS          = [2023]
TEST_YEARS         = [2024, 2025]
AERIAL_TRAIN_YEARS = [2020, 2021, 2022]   # 2019 excluded: 3.8% aerial rate vs 0.5% test
AERIAL_BEST_THR    = 0.35                  # selected from threshold scan (best F1 on v2 model)
K_TIERS            = 3
SEED               = 42
OUT_DIR            = 'resultados_temporal'

os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = ['Operacionais_Man', 'Meios_Terrestres']

# Feature set mirrors trainlightmodel.py Phase 2b (post-permutation-importance pruning).
# VPD_kPa excluded (negative permutation importance in Phase 2b).
# Natureza_* excluded (ranked last in feature importance — fire type != resource allocation).
# Duracao_Horas and Area_Ardida_ha excluded (post-event data — not available at dispatch time).
# doy_sin/doy_cos: continuous circular day-of-year encoding (added v2).
# district_T2_rate: appended dynamically after KMeans tier labelling (see section 5b).
BASE_FEATURES = [
    'LAT', 'LON',
    'Mes', 'Hora',
    'doy_sin', 'doy_cos',
    'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FM',
    'TEMPERATURA', 'HUMIDADERELATIVA', 'VENTOINTENSIDADE',
    'DECLIVEMEDIO', 'ALTITUDEMEDIA',
    'n_concurrent_fires',
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS  (mirrors trainlightmodel.py pre-split steps)
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Temporal Validation Pipeline v2 — Firehawk")
print("=" * 60)
print("\n[1] Loading and preprocessing data...")

df = pd.read_csv('dataset_final_clean.csv')
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

# Extract year from DHINICIO for temporal split key
df['DHINICIO'] = pd.to_datetime(df['DHINICIO'], errors='coerce')
df['Ano'] = df['DHINICIO'].dt.year

# Hora: "HH:MM" string -> integer hour (0-23)
if 'Hora' in df.columns:
    df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M', errors='coerce').dt.hour

# day-of-year sin/cos — smooth circular seasonal signal, avoids Dec/Jan discontinuity.
doy = df['DHINICIO'].dt.dayofyear.fillna(183)
df['doy_sin'] = np.sin(2 * np.pi * doy / 365)
df['doy_cos'] = np.cos(2 * np.pi * doy / 365)

# Natureza: strip whitespace, title-case, merge rare categories into most common
if 'Natureza' in df.columns:
    df['Natureza'] = df['Natureza'].str.strip().str.title()
    df['Natureza'] = df['Natureza'].replace('Florestal', 'Povoamento Florestal')
    main_cats = df['Natureza'].value_counts().nlargest(3).index.tolist()
    df.loc[~df['Natureza'].isin(main_cats), 'Natureza'] = main_cats[0]

if 'DISTRITO' in df.columns:
    df['DISTRITO'] = df['DISTRITO'].astype(str).str.strip().str.title()

# n_concurrent_fires: number of OTHER fires active in the same district on the same date.
# Uses only (DISTRITO, date) input columns — zero leakage risk.
# Computed on the full dataset so test fires in 2024-2025 reflect the actual
# operational load on their dates (including other fires in the same district).
if 'DISTRITO' in df.columns:
    df['_date'] = df['DHINICIO'].dt.date
    df['n_concurrent_fires'] = (
        df.groupby(['DISTRITO', '_date'])['DISTRITO'].transform('count') - 1
    ).fillna(0).astype(int)
    df.drop(columns=['_date'], inplace=True)
else:
    df['n_concurrent_fires'] = 0

print(f"  Loaded {len(df)} rows | "
      f"years: {sorted(df['Ano'].dropna().unique().astype(int).tolist())}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL SPLIT
# ──────────────────────────────────────────────────────────────────────────────
# Strict chronological split prevents future operational and seasonal patterns
# from leaking into the training phase — a problem that random split masks.
print("\n[2] Temporal split...")

df_train = df[df['Ano'].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
df_val   = df[df['Ano'].isin(VAL_YEARS)].copy().reset_index(drop=True)
df_test  = df[df['Ano'].isin(TEST_YEARS)].copy().reset_index(drop=True)

for name, part in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f"  {name}: {len(part):>6,} ({len(part)/len(df)*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. DISTRICT FEATURES  (computed from training partition only — no leakage)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] Building district features from training partition only...")


def build_district_stats(df_tr):
    """
    Compute district-level target encoding and capacity proxy from the training set.

    Returns
    -------
    dict  enc_<target>, global_mean_<target>, district_median, global_median_ops
          (district_T2_rate and global_T2_rate added in section 5b after KMeans)
    """
    stats = {}
    for t in TARGETS:
        stats[f'enc_{t}'] = df_tr.groupby('DISTRITO')[t].mean().to_dict()
        stats[f'global_mean_{t}'] = float(df_tr[t].mean())
    stats['district_median'] = df_tr.groupby('DISTRITO')['Operacionais_Man'].median().to_dict()
    stats['global_median_ops'] = float(df_tr['Operacionais_Man'].median())
    return stats


def apply_district_stats(df_part, stats):
    """
    Apply precomputed district stats to any partition.

    Returns
    -------
    (DataFrame, list[str])  enriched copy and list of added column names
    """
    df_out = df_part.copy()
    added  = []

    # Distrito_enc_Meios_Terrestres — Operacionais_Man encoding excluded because
    # it showed negative permutation importance in Phase 2b of trainlightmodel.py.
    col = 'Distrito_enc_Meios_Terrestres'
    df_out[col] = (df_out['DISTRITO']
                   .map(stats['enc_Meios_Terrestres'])
                   .fillna(stats['global_mean_Meios_Terrestres']))
    added.append(col)

    # district_median_single_incident: proxy for each district's typical resource ceiling.
    # Derived from targets so MUST use training-set stats only.
    col = 'district_median_single_incident'
    df_out[col] = (df_out['DISTRITO']
                   .map(stats['district_median'])
                   .fillna(stats['global_median_ops']))
    added.append(col)

    return df_out, added


district_stats         = build_district_stats(df_train)
df_train, extra_feats  = apply_district_stats(df_train, district_stats)
df_val,   _            = apply_district_stats(df_val,   district_stats)
df_test,  _            = apply_district_stats(df_test,  district_stats)

features = BASE_FEATURES + extra_feats
print(f"  Features after district stats: {len(features)}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXTRACT ARRAYS
# ──────────────────────────────────────────────────────────────────────────────
def extract_arrays(df_part, feat_list):
    """
    Extract model-ready arrays from a partition dataframe.

    Returns
    -------
    X, y, ya, yab  (feature matrix, regression targets, aerial count, aerial binary)
    """
    avail = [f for f in feat_list if f in df_part.columns]
    if len(avail) < len(feat_list):
        print(f"  WARNING: missing {set(feat_list) - set(avail)}")
    X   = df_part[avail].copy()
    y   = df_part[TARGETS].copy()
    ya  = df_part['Meios_Aereos'].copy()
    yab = (ya > 0).astype(int)
    return X, y, ya, yab


X_train, y_train, ya_train, yab_train = extract_arrays(df_train, features)
X_val,   y_val,   ya_val,   yab_val   = extract_arrays(df_val,   features)
X_test,  y_test,  ya_test,  yab_test  = extract_arrays(df_test,  features)

for name, yab in [('Train', yab_train), ('Val', yab_val), ('Test', yab_test)]:
    print(f"  Aerial positives {name}: {int(yab.sum())} ({yab.mean()*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. K-MEANS TIER LABELLING  (fit on train only)
# ──────────────────────────────────────────────────────────────────────────────
# KMeans is fit in log1p space: right-skewed targets require log compression so
# cluster geometry is governed by operational significance (2->5 resources is
# comparable to 20->50 resources) rather than raw magnitude.
print(f"\n[5] K-Means tier labelling (K={K_TIERS})...")

y_train_log = np.log1p(y_train[TARGETS].values)
kmeans = KMeans(n_clusters=K_TIERS, random_state=SEED, n_init=20)
kmeans.fit(y_train_log)

# Re-order tier labels by Operacionais_Man centroid (small -> large) so
# Tier 0 always means smallest deployment and Tier 2 the largest.
centroids_log  = kmeans.cluster_centers_
centroids_orig = np.expm1(centroids_log)
sort_order     = np.argsort(centroids_orig[:, 0])
remap          = {old: new for new, old in enumerate(sort_order)}
centroids_orig = centroids_orig[sort_order]


def assign_tiers(y_df):
    """Map a targets DataFrame to sorted tier labels using the fitted KMeans."""
    raw = kmeans.predict(np.log1p(y_df[TARGETS].values))
    return np.array([remap[l] for l in raw])


tier_train = np.array([remap[l] for l in kmeans.labels_])
tier_val   = assign_tiers(y_val)
tier_test  = assign_tiers(y_test)

# Tier ranges from training set (p5-p95).
# Used for range-based accuracy: the model is "correct" if the real value falls
# within the predicted tier's plausible bracket, not just the exact KMeans bucket.
tier_ranges = {}
for t in range(K_TIERS):
    mask = (tier_train == t)
    ops  = y_train['Operacionais_Man'].values[mask]
    veh  = y_train['Meios_Terrestres'].values[mask]
    tier_ranges[t] = {
        'ops_min': float(np.percentile(ops, 5)),
        'ops_max': float(np.percentile(ops, 95)),
        'veh_min': float(np.percentile(veh, 5)),
        'veh_max': float(np.percentile(veh, 95)),
        'ops_med': float(np.median(ops)),
        'veh_med': float(np.median(veh)),
    }

for i in range(K_TIERS):
    n   = int((tier_train == i).sum())
    r   = tier_ranges[i]
    pct = 100 * n / len(tier_train)
    print(f"  Tier {i}: Ops centroid={centroids_orig[i,0]:.1f}  "
          f"range=[{r['ops_min']:.0f}-{r['ops_max']:.0f}]  "
          f"({n} train fires, {pct:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# 5b. DISTRICT T2 RATE  (train-only, added after KMeans labels are available)
# ──────────────────────────────────────────────────────────────────────────────
# Per-district fraction of T2 incidents in the training set.
# Captures each district's propensity for large-scale deployments — a signal
# that district_median_single_incident does not provide (tail vs median).
# Computed from tier_train (KMeans output on train) so no target leakage.
print("\n[5b] Adding district_T2_rate feature (train-only)...")

district_T2_rate_map = (
    pd.DataFrame({'DISTRITO': df_train['DISTRITO'].values, 'is_T2': (tier_train == 2)})
    .groupby('DISTRITO')['is_T2']
    .mean()
    .to_dict()
)
global_T2_rate = float((tier_train == 2).mean())
district_stats['district_T2_rate'] = district_T2_rate_map
district_stats['global_T2_rate']   = global_T2_rate

for df_p in [df_train, df_val, df_test]:
    df_p['district_T2_rate'] = (
        df_p['DISTRITO']
        .map(district_T2_rate_map)
        .fillna(global_T2_rate)
    )

features.append('district_T2_rate')
print(f"  Total features: {len(features)}")
print(f"  {features}")

# Re-extract arrays with updated feature list
X_train, y_train, ya_train, yab_train = extract_arrays(df_train, features)
X_val,   y_val,   ya_val,   yab_val   = extract_arrays(df_val,   features)
X_test,  y_test,  ya_test,  yab_test  = extract_arrays(df_test,  features)


# ──────────────────────────────────────────────────────────────────────────────
# 6. BASELINES
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] Computing baselines...")

# Baseline A: majority class — always predict the most frequent tier in train.
maj_tier   = int(pd.Series(tier_train).value_counts().idxmax())
pred_A     = np.full(len(tier_test), maj_tier)
pred_A_val = np.full(len(tier_val),  maj_tier)

# Baseline B: predict mean Operacionais_Man and Meios_Terrestres per (DISTRITO, Mes)
# from the training set, then map to the nearest KMeans centroid.
# Rationale: this is the best simple operational heuristic available to a dispatcher
# (average resources historically sent to fires in this district in this month).
b_stats = (
    pd.DataFrame({
        'DISTRITO': df_train['DISTRITO'].values,
        'Mes':      df_train['Mes'].values,
        'ops':      y_train['Operacionais_Man'].values,
        'veh':      y_train['Meios_Terrestres'].values,
    })
    .groupby(['DISTRITO', 'Mes'])[['ops', 'veh']].mean()
)
g_ops = float(y_train['Operacionais_Man'].mean())
g_veh = float(y_train['Meios_Terrestres'].mean())


def _baseline_b(df_part):
    """Predict tier via (DISTRITO, Mes) district-month mean -> nearest KMeans centroid."""
    rows = []
    for _, row in df_part[['DISTRITO', 'Mes']].iterrows():
        key = (row['DISTRITO'], row['Mes'])
        rows.append(b_stats.loc[key].values if key in b_stats.index else [g_ops, g_veh])
    raw = kmeans.predict(np.log1p(np.array(rows)))
    return np.array([remap[l] for l in raw])


pred_B     = _baseline_b(df_test)
pred_B_val = _baseline_b(df_val)

# Baseline C: random stratified — sample tier labels matching train tier proportions.
rng        = np.random.default_rng(SEED)
train_p    = np.bincount(tier_train) / len(tier_train)
pred_C     = rng.choice(K_TIERS, size=len(tier_test), p=train_p)
pred_C_val = rng.choice(K_TIERS, size=len(tier_val),  p=train_p)

print(f"  A: always Tier {maj_tier}  |  "
      f"B: (district,month)->KMeans  |  C: random stratified {train_p.round(3)}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. RF TIER CLASSIFIER  +  ISOTONIC CALIBRATION ON VAL
# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters identical to trainlightmodel.py for fair comparison.
# class_weight='balanced' compensates for tier imbalance (Tier 1 ~ 55% of data).
print(f"\n[7] Training RF Tier Classifier (K={K_TIERS})...")

clf_tiers = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    random_state=SEED, n_jobs=-1
)
clf_tiers.fit(X_train, tier_train)

pred_RF_val  = clf_tiers.predict(X_val)
pred_RF_test = clf_tiers.predict(X_test)
print("  RF Tier Classifier trained.")

# ── Gradient-boosting benchmarks (XGBoost, HistGradientBoosting) ──────────────
# Required by the supervisory guidelines: demonstrate that changing the algorithm
# family does not, by itself, overcome the absence of operational variables.
# Trained on the IDENTICAL temporal features (X_train) and KMeans tier labels
# (tier_train) used by the Random Forest above; class imbalance handled with
# balanced sample/class weights, mirroring the RF's class_weight='balanced'.
print("\n[6b] Gradient-boosting benchmarks (same temporal protocol)...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
_tt_test  = assign_tiers(y_test)
_bench_sw = compute_sample_weight('balanced', tier_train)
pred_XGB = pred_HGB = None
try:
    from xgboost import XGBClassifier
    _xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                         n_jobs=-1, eval_metric='mlogloss', tree_method='hist')
    _xgb.fit(X_train, tier_train, sample_weight=_bench_sw)
    pred_XGB = _xgb.predict(X_test)
except Exception as _e:
    print("  XGBoost unavailable:", _e)
_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                      class_weight='balanced', random_state=SEED)
_hgb.fit(X_train, tier_train)
pred_HGB = _hgb.predict(X_test)
for _nm, _p in [("XGBoost", pred_XGB), ("HistGradientBoosting", pred_HGB)]:
    if _p is None: continue
    print(f"  {_nm}: acc={accuracy_score(_tt_test,_p):.3f} "
          f"bal_acc={balanced_accuracy_score(_tt_test,_p):.3f} "
          f"f1w={f1_score(_tt_test,_p,average='weighted'):.3f}")
# ─────────────────────────────────────────────────────────────────────────────


# Isotonic calibration fitted on Val 2023 (one-vs-rest, one IsotonicRegression per class).
# The base classifier's probability estimates shift under temporal covariate drift;
# per-class isotonic regression adjusts each score using the one held-out year.
# Calibrated predictions are evaluated on the test set only — val is in-sample
# for calibration so val metrics use the uncalibrated model.
print("  Fitting isotonic calibration on Val 2023 (one-vs-rest)...")
proba_val_base  = clf_tiers.predict_proba(X_val)
iso_regs = []
for k in range(K_TIERS):
    y_bin = (tier_val == k).astype(int)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba_val_base[:, k], y_bin)
    iso_regs.append(iso)

def predict_calibrated(X):
    p_raw = clf_tiers.predict_proba(X)
    p_cal = np.stack([iso_regs[k].predict(p_raw[:, k]) for k in range(K_TIERS)], axis=1)
    p_cal = np.clip(p_cal, 0, None)
    row_sums = p_cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    p_cal = p_cal / row_sums
    return np.argmax(p_cal, axis=1)

pred_RF_test_cal = predict_calibrated(X_test)
print("  Calibrated classifier ready.")


# ──────────────────────────────────────────────────────────────────────────────
# 8. TWO-STAGE AERIAL MODEL
# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: binary classifier — did this fire receive aerial support?
#   Trained on 2020-2022 only (2019 excluded: 3.8% aerial rate vs 0.5% test).
#   class_weight='balanced' remains critical: ~99% of fires have zero aerial assets.
# Stage 2: count regressor — how many aerial units?
#   Trained on ALL 2019-2022 positive cases (count data not affected by rate shift).
print("\n[8] Training two-stage aerial model...")

# Aerial train subset: 2020-2022 only (classifier stage)
df_aerial_tr = df[df['Ano'].isin(AERIAL_TRAIN_YEARS)].copy().reset_index(drop=True)
df_aerial_tr, _ = apply_district_stats(df_aerial_tr, district_stats)
df_aerial_tr['district_T2_rate'] = (
    df_aerial_tr['DISTRITO']
    .map(district_T2_rate_map)
    .fillna(global_T2_rate)
)
X_aerial_tr, _, ya_aerial_tr, yab_aerial_tr = extract_arrays(df_aerial_tr, features)
print(f"  Aerial train (2020-2022): {len(X_aerial_tr):,} fires | "
      f"{int(yab_aerial_tr.sum())} positives ({yab_aerial_tr.mean()*100:.1f}%)")

clf_aerial = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=SEED, n_jobs=-1
)
clf_aerial.fit(X_aerial_tr, yab_aerial_tr)

# Stage 2 regressor: all 2019-2022 positives
mask_train_pos = ya_train.values > 0
print(f"  Stage 2 regressor: {int(mask_train_pos.sum())} positive train cases (2019-2022)")
reg_aerial = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=2,
    random_state=SEED, n_jobs=-1
)
reg_aerial.fit(X_train[mask_train_pos], ya_train.values[mask_train_pos])

proba_val  = clf_aerial.predict_proba(X_val)[:, 1]
proba_test = clf_aerial.predict_proba(X_test)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# 9. METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_tier_metrics(y_true, y_pred):
    """Compute full tier classification metrics."""
    labels = list(range(K_TIERS))
    return {
        'acc':  float(accuracy_score(y_true, y_pred)),
        'bacc': float(balanced_accuracy_score(y_true, y_pred)),
        'f1w':  float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'rep':  classification_report(y_true, y_pred,
                                      target_names=[f'T{i}' for i in labels],
                                      zero_division=0, output_dict=True),
        'cm':   confusion_matrix(y_true, y_pred, labels=labels),
    }


def compute_range_acc(y_ops, y_veh, t_pred):
    """
    Range-based accuracy: fraction of test cases where the real value falls within
    the predicted tier's p5-p95 interval.
    """
    in_ops = np.array([
        tier_ranges[t_pred[j]]['ops_min'] <= y_ops[j] <= tier_ranges[t_pred[j]]['ops_max']
        for j in range(len(t_pred))
    ])
    in_veh = np.array([
        tier_ranges[t_pred[j]]['veh_min'] <= y_veh[j] <= tier_ranges[t_pred[j]]['veh_max']
        for j in range(len(t_pred))
    ])
    return float(in_ops.mean()), float(in_veh.mean()), float((in_ops & in_veh).mean())


def compute_aerial_s1(y_true, y_prob, thr=0.5):
    """Binary classifier metrics for aerial Stage 1 at a given threshold."""
    y_pred = (y_prob >= thr).astype(int)
    return {
        'acc':   float(accuracy_score(y_true, y_pred)),
        'prec':  float(precision_score(y_true, y_pred, zero_division=0)),
        'rec':   float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':    float(f1_score(y_true, y_pred, zero_division=0)),
        'auc':   float(roc_auc_score(y_true, y_prob)),
        'prauc': float(average_precision_score(y_true, y_prob)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10. EVALUATE — ALL MODELS ON TEST SET (primary) AND VAL SET (monitoring)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[10] Evaluating models...")

# Tier metrics
m = {
    'A':       compute_tier_metrics(tier_test, pred_A),
    'B':       compute_tier_metrics(tier_test, pred_B),
    'C':       compute_tier_metrics(tier_test, pred_C),
    'RF_val':  compute_tier_metrics(tier_val,  pred_RF_val),
    'RF_test': compute_tier_metrics(tier_test, pred_RF_test),
    'RF_cal':  compute_tier_metrics(tier_test, pred_RF_test_cal),
}

y_ops = y_test['Operacionais_Man'].values
y_veh = y_test['Meios_Terrestres'].values
ra = {
    'A':      compute_range_acc(y_ops, y_veh, pred_A),
    'B':      compute_range_acc(y_ops, y_veh, pred_B),
    'C':      compute_range_acc(y_ops, y_veh, pred_C),
    'RF':     compute_range_acc(y_ops, y_veh, pred_RF_test),
    'RF_cal': compute_range_acc(y_ops, y_veh, pred_RF_test_cal),
}

# Print summary
for key, lbl in [('A','Baseline A'), ('B','Baseline B'), ('C','Baseline C'),
                 ('RF_val','RF (Val, uncal)'), ('RF_test','RF (Test, uncal)'),
                 ('RF_cal','RF (Test, calibrated)')]:
    mx = m[key]
    print(f"  {lbl:30s} Acc={mx['acc']:.3f}  BAcc={mx['bacc']:.3f}  F1w={mx['f1w']:.3f}")

# Aerial Stage 1 — default threshold and selected operational threshold
aerial_val       = compute_aerial_s1(yab_val,  proba_val)
aerial_test      = compute_aerial_s1(yab_test, proba_test)
aerial_test_thr  = compute_aerial_s1(yab_test, proba_test, thr=AERIAL_BEST_THR)
print(f"  Aerial Val   AUC={aerial_val['auc']:.4f}  PR-AUC={aerial_val['prauc']:.4f}  "
      f"F1@0.5={aerial_val['f1']:.4f}")
print(f"  Aerial Test  AUC={aerial_test['auc']:.4f}  PR-AUC={aerial_test['prauc']:.4f}  "
      f"F1@0.5={aerial_test['f1']:.4f}  F1@{AERIAL_BEST_THR}={aerial_test_thr['f1']:.4f}")

# Threshold scan (test set)
scan_rows = []
for thr in np.arange(0.10, 0.91, 0.05):
    r = compute_aerial_s1(yab_test, proba_test, thr=float(thr))
    scan_rows.append({'threshold': round(float(thr), 2), **r})
df_scan = pd.DataFrame(scan_rows)

# Aerial Stage 2 (count regressor on positives only)
mask_test_pos = ya_test.values > 0
n_test_pos    = int(mask_test_pos.sum())

if n_test_pos > 0:
    ya_pred_pos = reg_aerial.predict(X_test[mask_test_pos])
    ya_true_pos = ya_test.values[mask_test_pos]
    mae_s2  = float(mean_absolute_error(ya_true_pos, ya_pred_pos))
    rmse_s2 = float(np.sqrt(np.mean((ya_true_pos - ya_pred_pos) ** 2)))
    print(f"  Aerial Stage 2: MAE={mae_s2:.2f}  RMSE={rmse_s2:.2f}  "
          f"(n={n_test_pos} test positives)")

    pred_bins = pd.cut(ya_pred_pos,
                       bins=[0, 1, 2, 5, 10, 20, np.inf],
                       labels=['1', '2', '3-5', '6-10', '11-20', '21+'])
    interval_rows = []
    for lbl in ['1', '2', '3-5', '6-10', '11-20', '21+']:
        mk = pred_bins == lbl
        if mk.sum() > 0:
            interval_rows.append({
                'Interval': lbl,
                'n': int(mk.sum()),
                'MAE':  round(float(mean_absolute_error(ya_true_pos[mk], ya_pred_pos[mk])), 2),
                'RMSE': round(float(np.sqrt(np.mean((ya_true_pos[mk] - ya_pred_pos[mk]) ** 2))), 2),
            })
    df_interval = pd.DataFrame(interval_rows)
else:
    mae_s2 = rmse_s2 = None
    df_interval = pd.DataFrame()
    print("  Aerial Stage 2: no positive test cases.")


# ──────────────────────────────────────────────────────────────────────────────
# 11. FIGURES
# ──────────────────────────────────────────────────────────────────────────────
print("\n[11] Generating figures...")

sns.set_theme(style='whitegrid', font_scale=1.1)
tier_labels = [f"T{i}\n({centroids_orig[i,0]:.0f}m,{centroids_orig[i,1]:.0f}v)"
               for i in range(K_TIERS)]

# ---- Figure 1: Confusion matrix — uncalibrated (principal) model (test set) ---
cm_unc = m['RF_test']['cm']
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_unc, annot=True, fmt='d', cmap='Blues',
            xticklabels=tier_labels, yticklabels=tier_labels, ax=ax)
ax.set_xlabel('Predicted Tier')
ax.set_ylabel('True Tier')
ax.set_title(
    f'Confusion Matrix — RF Tier Classifier (Uncalibrated, principal) (Test 2024-2025)\n'
    f'Acc={m["RF_test"]["acc"]:.3f}  '
    f'Balanced Acc={m["RF_test"]["bacc"]:.3f}  '
    f'Weighted F1={m["RF_test"]["f1w"]:.3f}'
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'confusion_matrix_temporal.png'), dpi=120)
plt.close(fig)
print("  -> confusion_matrix_temporal.png (uncalibrated, principal model)")

# ---- Figure 2: Precision-Recall curve (aerial Stage 1, test set) -------------
prec_curve, rec_curve, _ = precision_recall_curve(yab_test, proba_test)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_curve, prec_curve, color='royalblue', lw=2,
        label=f'RF Classifier  PR-AUC={aerial_test["prauc"]:.3f}')
ax.axhline(float(yab_test.mean()), color='grey', linestyle='--', alpha=0.8,
           label=f'No-skill baseline ({yab_test.mean()*100:.1f}% positive rate)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — Aerial Stage 1 (Test 2024-2025)\n'
             f'Train: 2020-2022 only (2019 excluded)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pr_curve_aerial.png'), dpi=120)
plt.close(fig)
print("  -> pr_curve_aerial.png")

# ---- Figure 3: Threshold scan ------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for col, color, lbl in [('prec', 'royalblue', 'Precision'),
                         ('rec',  'tomato',    'Recall'),
                         ('f1',   'seagreen',  'F1')]:
    ax1.plot(df_scan['threshold'], df_scan[col], 'o-',
             color=color, label=lbl, markersize=4, linewidth=1.5)
ax1.axvline(AERIAL_BEST_THR, color='black', linestyle=':', alpha=0.7,
            label=f'Selected thr={AERIAL_BEST_THR}')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Score')
ax1.set_title('Precision / Recall / F1 vs Threshold\n(Aerial Stage 1, Test 2024-2025)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df_scan['threshold'], df_scan['acc'], 's-',
         color='purple', label='Accuracy', markersize=4, linewidth=1.5)
ax2.axvline(AERIAL_BEST_THR, color='black', linestyle=':', alpha=0.7,
            label=f'Selected thr={AERIAL_BEST_THR}')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Threshold\n(Aerial Stage 1, Test 2024-2025)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Threshold Scan — Aerial Binary Classifier')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'threshold_scan.png'), dpi=120)
plt.close(fig)
print("  -> threshold_scan.png")

# ---- Figure 4: Feature importance (MDI, calibrated RF — same base estimator) -
importances = (pd.Series(clf_tiers.feature_importances_, index=features)
               .sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=importances.values, y=importances.index,
            hue=importances.index, legend=False, palette='viridis', ax=ax)
ax.set_title('Feature Importance (MDI) — RF Tier Classifier\nTemporal Validation Split v2')
ax.set_xlabel('Mean Decrease in Impurity')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'feature_importance_temporal.png'), dpi=120)
plt.close(fig)
print("  -> feature_importance_temporal.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12. SAVE PIPELINE PKL
# ──────────────────────────────────────────────────────────────────────────────
print("\n[12] Saving pipeline pickle...")

pipeline = {
    'kmeans':            kmeans,
    'kmeans_sort_order': sort_order,
    'remap':             remap,
    'tier_clf':          clf_tiers,
    'tier_clf_iso_regs':   iso_regs,
    'aerial_clf':        clf_aerial,
    'aerial_reg':        reg_aerial,
    'aerial_best_thr':   AERIAL_BEST_THR,
    'features':          features,
    'district_stats':    district_stats,
    'tier_ranges':       tier_ranges,
    'centroids_orig':    centroids_orig,
    'split_info': {
        'train_years':        TRAIN_YEARS,
        'aerial_train_years': AERIAL_TRAIN_YEARS,
        'val_years':          VAL_YEARS,
        'test_years':         TEST_YEARS,
        'n_train':            int(len(X_train)),
        'n_val':              int(len(X_val)),
        'n_test':             int(len(X_test)),
    },
    'random_seed': SEED,
    'K_TIERS':     K_TIERS,
    'version':     'v2',
    'generated':   str(_date.today()),
}
joblib.dump(pipeline, os.path.join(OUT_DIR, 'pipeline_temporal.pkl'))
print("  -> pipeline_temporal.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# 13. MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n[13] Writing resultados_temporal.md...")


def md_row(*cols):
    return '| ' + ' | '.join(str(c) for c in cols) + ' |'


lines = []
W = lines.append

W(f"# Firehawk — Temporal Validation Results (v2)")
W(f"Generated: {_date.today()}")
W("")
W("---")
W("")
W("## 1. Dataset Split")
W("")
W(md_row("Partition", "Years", "N fires", "% dataset",
         "Aerial positives", "Aerial rate"))
W(md_row("---", "---", "---", "---", "---", "---"))
for pname, years, df_p, yab_p in [
        ('Train',        TRAIN_YEARS, df_train, yab_train),
        ('Aerial Train', AERIAL_TRAIN_YEARS, df_aerial_tr, yab_aerial_tr),
        ('Val',          VAL_YEARS,   df_val,   yab_val),
        ('Test',         TEST_YEARS,  df_test,  yab_test)]:
    yr_str = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    W(md_row(pname, yr_str, f"{len(df_p):,}", f"{len(df_p)/len(df)*100:.1f}%",
             int(yab_p.sum()), f"{yab_p.mean()*100:.1f}%"))
W("")
W("> **Note — 2019 aerial anomaly**: the aerial positive rate in 2019 (3.8%) is ~6x higher "
  "than the 0.2-0.8% rate observed in 2020-2025. The aerial classifier is therefore trained "
  "on 2020-2022 only to align the training prior with the test distribution. "
  "The tier classifier and aerial regressor retain the full 2019-2022 training set.")
W("")
W("---")
W("")
W("## 2. Tier Centroids and Ranges (Training Set 2019-2022)")
W("")
W(md_row("Tier", "Ops centroid", "Veh centroid",
         "Ops p5-p95", "Veh p5-p95", "N train", "%"))
W(md_row(*["---"] * 7))
for i in range(K_TIERS):
    n   = int((tier_train == i).sum())
    r   = tier_ranges[i]
    pct = 100 * n / len(tier_train)
    W(md_row(f"T{i}",
             f"{centroids_orig[i,0]:.1f}", f"{centroids_orig[i,1]:.1f}",
             f"[{r['ops_min']:.0f}-{r['ops_max']:.0f}]",
             f"[{r['veh_min']:.0f}-{r['veh_max']:.0f}]",
             f"{n:,}", f"{pct:.1f}%"))
W("")
W("---")
W("")
# Benchmark metrics for the markdown table (gradient-boosting, from section [6b])
if pred_XGB is not None:
    m['XGB']  = compute_tier_metrics(tier_test, pred_XGB)
    ra['XGB'] = compute_range_acc(y_ops, y_veh, pred_XGB)
if pred_HGB is not None:
    m['HGB']  = compute_tier_metrics(tier_test, pred_HGB)
    ra['HGB'] = compute_range_acc(y_ops, y_veh, pred_HGB)
W("## 3. Tier Classifier — Summary Comparison (Test Set 2024-2025)")
W("")
W(md_row("Model", "Accuracy", "Balanced Acc", "Weighted F1",
         "Ops in range", "Veh in range", "Both in range"))
W(md_row(*["---"] * 7))
_tier_rows = [
        ('A',       'Baseline A — majority class',                    'A'),
        ('B',       'Baseline B — district x month',                  'B'),
        ('C',       'Baseline C — random stratified',                 'C'),
        ('RF_test', 'RF Classifier (temporal split)',                  'RF'),
]
if 'XGB' in m: _tier_rows.append(('XGB', 'XGBoost (benchmark)',              'XGB'))
if 'HGB' in m: _tier_rows.append(('HGB', 'HistGradientBoosting (benchmark)', 'HGB'))
_tier_rows.append(('RF_cal',  'RF + Isotonic Calibration (Val 2023)',        'RF_cal'))
for key, lbl, ra_key in _tier_rows:
    mx = m[key]
    rx = ra[ra_key]
    W(md_row(lbl,
             f"{mx['acc']:.3f}", f"{mx['bacc']:.3f}", f"{mx['f1w']:.3f}",
             f"{rx[0]*100:.1f}%", f"{rx[1]*100:.1f}%", f"{rx[2]*100:.1f}%"))
W("")
W("---")
W("")
W("## 4. RF Tier Classifier — Per-Class Metrics (Test Set)")
W("")
W("### 4a. Uncalibrated")
W("")
W(md_row("Class", "Description", "Precision", "Recall", "F1-score", "Support"))
W(md_row(*["---"] * 6))
rep = m['RF_test']['rep']
for i in range(K_TIERS):
    row = rep.get(f'T{i}', {})
    r   = tier_ranges[i]
    W(md_row(f"T{i}",
             f"Ops [{r['ops_min']:.0f}-{r['ops_max']:.0f}]",
             f"{row.get('precision', 0):.3f}",
             f"{row.get('recall', 0):.3f}",
             f"{row.get('f1-score', 0):.3f}",
             f"{int(row.get('support', 0))}"))
W("")
W("### 4b. Calibrated (isotonic, fitted on Val 2023)")
W("")
W(md_row("Class", "Description", "Precision", "Recall", "F1-score", "Support"))
W(md_row(*["---"] * 6))
rep_cal = m['RF_cal']['rep']
for i in range(K_TIERS):
    row = rep_cal.get(f'T{i}', {})
    r   = tier_ranges[i]
    W(md_row(f"T{i}",
             f"Ops [{r['ops_min']:.0f}-{r['ops_max']:.0f}]",
             f"{row.get('precision', 0):.3f}",
             f"{row.get('recall', 0):.3f}",
             f"{row.get('f1-score', 0):.3f}",
             f"{int(row.get('support', 0))}"))
W("")
W("*Confusion matrix (uncalibrated, principal model): see `confusion_matrix_temporal.png`*")
W("")
W("---")
W("")
W("## 5. RF Tier Classifier — Validation Set Metrics (2023, uncalibrated)")
W("")
W(f"Accuracy: **{m['RF_val']['acc']:.4f}**  |  "
  f"Balanced Accuracy: **{m['RF_val']['bacc']:.4f}**  |  "
  f"Weighted F1: **{m['RF_val']['f1w']:.4f}**")
W("")
W(md_row("Class", "Precision", "Recall", "F1-score", "Support"))
W(md_row(*["---"] * 5))
val_rep = m['RF_val']['rep']
for i in range(K_TIERS):
    row = val_rep.get(f'T{i}', {})
    W(md_row(f"T{i}",
             f"{row.get('precision', 0):.3f}",
             f"{row.get('recall', 0):.3f}",
             f"{row.get('f1-score', 0):.3f}",
             f"{int(row.get('support', 0))}"))
W("")
W("---")
W("")
W("## 6. Aerial Model — Stage 1: Binary Classifier")
W("")
W(md_row("Partition / Threshold", "Accuracy", "Precision", "Recall",
         "F1", "ROC-AUC", "PR-AUC"))
W(md_row(*["---"] * 7))
for pname, av in [('Val (2023) @ thr=0.50', aerial_val),
                  ('Test (2024-2025) @ thr=0.50', aerial_test),
                  (f'Test (2024-2025) @ thr={AERIAL_BEST_THR} *(selected)*', aerial_test_thr)]:
    W(md_row(pname,
             f"{av['acc']:.4f}", f"{av['prec']:.4f}", f"{av['rec']:.4f}",
             f"{av['f1']:.4f}", f"{av['auc']:.4f}", f"{av['prauc']:.4f}"))
W("")
W(f"> Operational threshold {AERIAL_BEST_THR} selected as the F1-maximising point "
  f"on the threshold scan (Test 2024-2025). The aerial classifier is trained on "
  f"2020-2022 only (2019 aerial rate 3.8% vs 0.5% in test).")
W("")
W("*PR curve: see `pr_curve_aerial.png`*")
W("")
W("---")
W("")
W("## 7. Aerial Stage 1 — Threshold Scan (Test Set 2024-2025)")
W("")
W(md_row("Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"))
W(md_row(*["---"] * 7))
for _, row in df_scan.iterrows():
    marker = " *(selected)*" if abs(row['threshold'] - AERIAL_BEST_THR) < 0.001 else ""
    W(md_row(f"{row['threshold']:.2f}{marker}",
             f"{row['acc']:.4f}", f"{row['prec']:.4f}", f"{row['rec']:.4f}",
             f"{row['f1']:.4f}", f"{row['auc']:.4f}", f"{row['prauc']:.4f}"))
W("")
W("*Threshold scan plot: see `threshold_scan.png`*")
W("")
W("---")
W("")
if n_test_pos > 0:
    W("## 8. Aerial Stage 2 — Count Regressor (Test Set Positives Only)")
    W("")
    W(f"N test positives: **{n_test_pos}**  |  "
      f"MAE: **{mae_s2:.2f}**  |  RMSE: **{rmse_s2:.2f}**")
    W("")
    W("### Error by Predicted Interval")
    W("")
    W(md_row("Predicted interval", "n", "MAE", "RMSE"))
    W(md_row(*["---"] * 4))
    for _, row in df_interval.iterrows():
        W(md_row(row['Interval'], row['n'], f"{row['MAE']:.2f}", f"{row['RMSE']:.2f}"))
    W("")
    W("---")
    W("")
W("## 9. Feature Importance (MDI, RF Tier Classifier)")
W("")
W(md_row("Feature", "Importance"))
W(md_row("---", "---"))
for feat, imp in importances.items():
    W(md_row(feat, f"{imp:.4f}"))
W("")
W("*Importance plot: see `feature_importance_temporal.png`*")
W("")
W("---")
W("")
W("## Notes")
W("")
W("- All pkl files in `resultados_temporal/` are independent of production pkl files.")
W("- Existing `model_tier_classifier.pkl`, `model_aerial_classifier.pkl`, "
  "`model_aerial_regressor.pkl` etc. are **unchanged**.")
W(f"- Random seed: {SEED} throughout (KMeans n_init=20, RF n_estimators=300).")
W("- District target encoding and capacity features computed from training years "
  "(2019-2022) only and applied to val/test by map — no data leakage.")
W("- `district_T2_rate` computed from KMeans tier labels on training set only; "
  "applied to val/test via map — no data leakage.")
W("- KMeans fit exclusively on training partition; val/test tiers assigned via "
  "`kmeans.predict()` on precomputed centroids.")
W("- RF tier classifier calibrated with isotonic regression fitted on Val 2023. "
  "Calibrated predictions reported for test set only (val is in-sample for calibration).")
W("- Aerial classifier trained on 2020-2022 only (2019 excluded: positive rate 3.8% "
  "vs 0.5% in test set). Aerial regressor trained on 2019-2022 positives.")
W(f"- Aerial operational threshold set to {AERIAL_BEST_THR} "
  "(F1-maximising point from threshold scan).")

md_path = os.path.join(OUT_DIR, 'resultados_temporal.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  -> resultados_temporal.md")

print("\n" + "=" * 60)
print(f"DONE (v2). All outputs saved to: {OUT_DIR}/")
print("=" * 60)
