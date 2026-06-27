# Firehawk — Temporal Validation Results (v2)
Generated: 2026-06-27

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
| RF Classifier (temporal split) | 0.488 | 0.406 | 0.480 | 48.4% | 72.2% | 46.7% |
| XGBoost (benchmark) | 0.460 | 0.412 | 0.468 | 45.6% | 69.3% | 44.1% |
| HistGradientBoosting (benchmark) | 0.437 | 0.411 | 0.453 | 43.2% | 66.4% | 41.7% |
| RF + Isotonic Calibration (Val 2023) | 0.533 | 0.342 | 0.402 | 53.0% | 81.5% | 50.7% |

---

## 4. RF Tier Classifier — Per-Class Metrics (Test Set)

### 4a. Uncalibrated

| Class | Description | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- | --- |
| T0 | Ops [1-3] | 0.382 | 0.261 | 0.310 | 2404 |
| T1 | Ops [4-7] | 0.615 | 0.677 | 0.644 | 4650 |
| T2 | Ops [8-24] | 0.237 | 0.280 | 0.257 | 1599 |

### 4b. Calibrated (isotonic, fitted on Val 2023)

| Class | Description | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- | --- |
| T0 | Ops [1-3] | 0.363 | 0.049 | 0.086 | 2404 |
| T1 | Ops [4-7] | 0.543 | 0.963 | 0.695 | 4650 |
| T2 | Ops [8-24] | 0.239 | 0.014 | 0.026 | 1599 |

*Confusion matrix (uncalibrated, principal model): see `confusion_matrix_temporal.png`*

---

## 5. RF Tier Classifier — Validation Set Metrics (2023, uncalibrated)

Accuracy: **0.5022**  |  Balanced Accuracy: **0.4103**  |  Weighted F1: **0.4902**

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| T0 | 0.347 | 0.232 | 0.278 | 1201 |
| T1 | 0.632 | 0.704 | 0.666 | 2494 |
| T2 | 0.261 | 0.295 | 0.277 | 861 |

---

## 6. Aerial Model — Stage 1: Binary Classifier

| Partition / Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Val (2023) @ thr=0.50 | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| Test (2024-2025) @ thr=0.50 | 0.9939 | 0.0000 | 0.0000 | 0.0000 | 0.6549 | 0.0119 |
| Test (2024-2025) @ thr=0.4 *(selected)* | 0.9846 | 0.0323 | 0.0652 | 0.0432 | 0.6549 | 0.0119 |

> Operational threshold 0.4 selected as the F1-maximising point on the threshold scan over the **Validation set (2023)**, then frozen and applied to the test set (no test-set tuning). The aerial classifier is trained on 2020-2022 only (2019 aerial rate 3.8% vs 0.5% in test).

*PR curve: see `pr_curve_aerial.png`*

---

## 7. Aerial Stage 1 — Threshold Scan (Validation Set 2023 — threshold selection)

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.4493 | 0.0115 | 0.7838 | 0.0226 | 0.7129 | 0.0407 |
| 0.10 | 0.6271 | 0.0146 | 0.6757 | 0.0286 | 0.7129 | 0.0407 |
| 0.15 | 0.7695 | 0.0190 | 0.5405 | 0.0367 | 0.7129 | 0.0407 |
| 0.20 | 0.8593 | 0.0237 | 0.4054 | 0.0447 | 0.7129 | 0.0407 |
| 0.25 | 0.9124 | 0.0286 | 0.2973 | 0.0523 | 0.7129 | 0.0407 |
| 0.30 | 0.9484 | 0.0374 | 0.2162 | 0.0637 | 0.7129 | 0.0407 |
| 0.35 | 0.9704 | 0.0545 | 0.1622 | 0.0816 | 0.7129 | 0.0407 |
| 0.40 *(selected)* | 0.9820 | 0.0755 | 0.1081 | 0.0889 | 0.7129 | 0.0407 |
| 0.45 | 0.9884 | 0.0556 | 0.0270 | 0.0364 | 0.7129 | 0.0407 |
| 0.50 | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.55 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.60 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.65 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.70 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.75 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.80 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.85 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.90 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |
| 0.95 | 0.9919 | 0.0000 | 0.0000 | 0.0000 | 0.7129 | 0.0407 |

*Threshold scan plot: see `threshold_scan.png`*

---

## 8. Aerial Stage 2 — Count Regressor (Test Set Positives Only)

N test positives: **46**  |  MAE: **0.25**  |  RMSE: **0.54**

### Error by Predicted Interval

| Predicted interval | n | MAE | RMSE |
| --- | --- | --- | --- |
| 2 | 42 | 0.10 | 0.14 |
| 3-5 | 4 | 1.74 | 1.76 |

---

## 9. Feature Importance (MDI, RF Tier Classifier)

| Feature | Importance |
| --- | --- |
| LAT | 0.0796 |
| LON | 0.0682 |
| DC | 0.0614 |
| TEMPERATURA | 0.0589 |
| DECLIVEMEDIO | 0.0581 |
| ALTITUDEMEDIA | 0.0556 |
| VENTOINTENSIDADE | 0.0553 |
| HUMIDADERELATIVA | 0.0550 |
| DMC | 0.0547 |
| BUI | 0.0546 |
| doy_cos | 0.0535 |
| FWI | 0.0524 |
| FM | 0.0523 |
| doy_sin | 0.0517 |
| ISI | 0.0506 |
| Hora | 0.0355 |
| district_T2_rate | 0.0272 |
| Distrito_enc_Meios_Terrestres | 0.0258 |
| n_concurrent_fires | 0.0247 |
| Mes | 0.0146 |
| district_median_single_incident | 0.0102 |

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
- Aerial operational threshold = 0.4, selected as the F1-maximising point of the threshold scan on the Validation set (2023), then frozen and applied to the test set (no test-set tuning).