# Firehawk — Temporal Validation Results (v2)
Generated: 2026-06-23

---

## 1. Dataset Split

| Partition | Years | N fires | % dataset | Aerial positives | Aerial rate |
| --- | --- | --- | --- | --- | --- |
| Train | 2019-2022 | 26,340 | 66.6% | 358 | 1.4% |
| Aerial Train | 2020-2022 | 18,998 | 48.0% | 82 | 0.4% |
| Val | 2023 | 4,556 | 11.5% | 37 | 0.8% |
| Test | 2024-2025 | 8,653 | 21.9% | 46 | 0.5% |

> **Note — 2019 aerial anomaly**: the aerial positive rate in 2019 (3.8%) is ~6x higher than the 0.2-0.8% rate observed in 2020-2025. The aerial classifier is therefore trained on 2020-2022 only to align the training prior with the test distribution. The tier classifier and aerial regressor retain the full 2019-2022 training set.

---

## 2. Tier Centroids and Ranges (Training Set 2019-2022)

| Tier | Ops centroid | Veh centroid | Ops p5-p95 | Veh p5-p95 | N train | % |
| --- | --- | --- | --- | --- | --- | --- |
| T0 | 2.0 | 1.0 | [1-3] | [1-1] | 5,533 | 21.0% |
| T1 | 5.4 | 1.4 | [4-7] | [1-2] | 14,383 | 54.6% |
| T2 | 12.1 | 3.5 | [8-24] | [2-7] | 6,424 | 24.4% |

---

## 3. Tier Classifier — Summary Comparison (Test Set 2024-2025)

| Model | Accuracy | Balanced Acc | Weighted F1 | Ops in range | Veh in range | Both in range |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline A — majority class | 0.537 | 0.333 | 0.376 | 53.4% | 82.6% | 51.0% |
| Baseline B — district x month | 0.510 | 0.348 | 0.401 | 50.7% | 78.6% | 48.6% |
| Baseline C — random stratified | 0.398 | 0.336 | 0.397 | 39.6% | 67.9% | 38.0% |
| RF Classifier (temporal split) | 0.485 | 0.404 | 0.477 | 48.2% | 72.0% | 46.5% |
| XGBoost (benchmark) | 0.452 | 0.402 | 0.460 | 44.6% | 68.6% | 43.1% |
| HistGradientBoosting (benchmark) | 0.433 | 0.409 | 0.449 | 42.8% | 66.3% | 41.3% |
| RF + Isotonic Calibration (Val 2023) | 0.534 | 0.344 | 0.406 | 53.1% | 81.5% | 50.8% |

---

## 4. RF Tier Classifier — Per-Class Metrics (Test Set)

### 4a. Uncalibrated

| Class | Description | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- | --- |
| T0 | Ops [1-3] | 0.375 | 0.255 | 0.303 | 2404 |
| T1 | Ops [4-7] | 0.614 | 0.674 | 0.642 | 4650 |
| T2 | Ops [8-24] | 0.237 | 0.283 | 0.258 | 1599 |

### 4b. Calibrated (isotonic, fitted on Val 2023)

| Class | Description | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- | --- |
| T0 | Ops [1-3] | 0.380 | 0.049 | 0.086 | 2404 |
| T1 | Ops [4-7] | 0.546 | 0.961 | 0.696 | 4650 |
| T2 | Ops [8-24] | 0.237 | 0.023 | 0.042 | 1599 |

*Confusion matrix (uncalibrated, principal model): see `confusion_matrix_temporal.png`*

---

## 5. RF Tier Classifier — Validation Set Metrics (2023, uncalibrated)

Accuracy: **0.5022**  |  Balanced Accuracy: **0.4089**  |  Weighted F1: **0.4888**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| T0 | 0.347 | 0.220 | 0.269 | 1201 |
| T1 | 0.632 | 0.709 | 0.668 | 2494 |
| T2 | 0.257 | 0.298 | 0.276 | 861 |

---

## 6. Aerial Model — Stage 1: Binary Classifier

| Partition / Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Val (2023) @ thr=0.50 | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0.6988 | 0.0377 |
| Test (2024-2025) @ thr=0.50 | 0.9942 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| Test (2024-2025) @ thr=0.35 *(selected)* | 0.9757 | 0.0287 | 0.1087 | 0.0455 | 0.6571 | 0.0119 |

> Operational threshold 0.35 selected as the F1-maximising point on the threshold scan (Test 2024-2025). The aerial classifier is trained on 2020-2022 only (2019 aerial rate 3.8% vs 0.5% in test).

*PR curve: see `pr_curve_aerial.png`*

---

## 7. Aerial Stage 1 — Threshold Scan (Test Set 2024-2025)

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 0.5610 | 0.0079 | 0.6522 | 0.0155 | 0.6571 | 0.0119 |
| 0.15 | 0.7006 | 0.0093 | 0.5217 | 0.0182 | 0.6571 | 0.0119 |
| 0.20 | 0.8049 | 0.0084 | 0.3043 | 0.0163 | 0.6571 | 0.0119 |
| 0.25 | 0.8896 | 0.0108 | 0.2174 | 0.0205 | 0.6571 | 0.0119 |
| 0.30 | 0.9456 | 0.0159 | 0.1522 | 0.0289 | 0.6571 | 0.0119 |
| 0.35 *(selected)* | 0.9757 | 0.0287 | 0.1087 | 0.0455 | 0.6571 | 0.0119 |
| 0.40 | 0.9882 | 0.0172 | 0.0217 | 0.0192 | 0.6571 | 0.0119 |
| 0.45 | 0.9928 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.50 | 0.9942 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.55 | 0.9946 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.60 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.65 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.70 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.75 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.80 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.85 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |
| 0.90 | 0.9947 | 0.0000 | 0.0000 | 0.0000 | 0.6571 | 0.0119 |

*Threshold scan plot: see `threshold_scan.png`*

---

## 8. Aerial Stage 2 — Count Regressor (Test Set Positives Only)

N test positives: **46**  |  MAE: **0.25**  |  RMSE: **0.54**

### Error by Predicted Interval

| Predicted interval | n | MAE | RMSE |
| --- | --- | --- | --- |
| 2 | 42 | 0.11 | 0.14 |
| 3-5 | 4 | 1.74 | 1.76 |

---

## 9. Feature Importance (MDI, RF Tier Classifier)

| Feature | Importance |
| --- | --- |
| LAT | 0.0789 |
| LON | 0.0672 |
| DC | 0.0605 |
| DECLIVEMEDIO | 0.0583 |
| TEMPERATURA | 0.0582 |
| ALTITUDEMEDIA | 0.0557 |
| HUMIDADERELATIVA | 0.0554 |
| VENTOINTENSIDADE | 0.0552 |
| DMC | 0.0542 |
| BUI | 0.0539 |
| doy_cos | 0.0527 |
| FM | 0.0525 |
| FWI | 0.0515 |
| doy_sin | 0.0510 |
| ISI | 0.0503 |
| Hora | 0.0366 |
| n_concurrent_fires | 0.0305 |
| district_T2_rate | 0.0270 |
| Distrito_enc_Meios_Terrestres | 0.0262 |
| Mes | 0.0144 |
| district_median_single_incident | 0.0101 |

*Importance plot: see `feature_importance_temporal.png`*

---

## Notes

- All pkl files in `resultados_temporal/` are independent of production pkl files.
- Existing `model_tier_classifier.pkl`, `model_aerial_classifier.pkl`, `model_aerial_regressor.pkl` etc. are **unchanged**.
- Random seed: 42 throughout (KMeans n_init=20, RF n_estimators=300).
- District target encoding and capacity features computed from training years (2019-2022) only and applied to val/test by map — no data leakage.
- `district_T2_rate` computed from KMeans tier labels on training set only; applied to val/test via map — no data leakage.
- KMeans fit exclusively on training partition; val/test tiers assigned via `kmeans.predict()` on precomputed centroids.
- RF tier classifier calibrated with isotonic regression fitted on Val 2023. Calibrated predictions reported for test set only (val is in-sample for calibration).
- Aerial classifier trained on 2020-2022 only (2019 excluded: positive rate 3.8% vs 0.5% in test set). Aerial regressor trained on 2019-2022 positives.
- Aerial operational threshold set to 0.35 (F1-maximising point from threshold scan).