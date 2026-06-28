Firehawk Project - Release Notes

Version: v2.2 (Dissertation Tier Model & Methodology Hardening) Date: June 28, 2026 Deployment Target: Railway PaaS / Local Hybrid Environment

🦅 Overview
Release v2.2 aligns the served model with the dissertation's final formulation. The exact-count resource regressor (model_resources_lite.pkl) is replaced by the tier classifier and two-stage aerial model (model_tier_pipeline.pkl): for each incident the system now reports an expected response tier (minimal / standard / reinforced) with a characteristic resource range and an aerial-mobilisation probability, instead of exact counts. This release also incorporates two methodology corrections and makes the manual refresh non-blocking.

✨ Changes
- Model migration: the active bundle is now model_tier_pipeline.pkl (KMeans tiers fitted on the training partition + Random Forest tier classifier + two-stage aerial model). The UI reports tier, resource range and aerial probability; the "Expected vs Real" panel contrasts the prediction with the live deployment.
- Leakage-free concurrency feature: n_concurrent_fires now counts only fires in the same district that ignited at or before the incident on the same day (no look-ahead); incident end time and duration are not used. It is a minor predictor (MDI importance rank ~19 of 21).
- Honest aerial threshold: the operating threshold (0.40) is the F1-maximising point on the validation set (2023), frozen before test reporting — no longer tuned on the test set.
- Non-blocking manual refresh: POST /api/refresh-data starts the pipeline in a background thread and returns HTTP 202; the frontend polls fires.json for the result. A run already in progress returns HTTP 409.

📊 Model results (temporal split: train 2019-2022 / val 2023 / test 2024-2025)
- Tier classifier (Random Forest), test 2024-2025: accuracy 0.488, balanced accuracy 0.406, weighted F1 0.480; per-tier recall T0/T1/T2 = 0.261 / 0.677 / 0.280; vehicle range accuracy 72.2%.
- Two-stage aerial model: threshold 0.40 (validation-selected, frozen); test ROC-AUC 0.655, PR-AUC 0.012.

🐛 Corrections & Fixes
- Removed a temporal look-ahead in n_concurrent_fires (previously counted all same-calendar-day fires, including later ignitions).
- Removed test-set tuning of the aerial operating threshold.

Notes
- The tier formulation predicts an operational response level and range, not exact resource counts; reinforced-tier magnitudes reflect national-level reinforcement that open pre-event data cannot fully anticipate.
- The system remains a research demonstrator, not validated in a live operational environment.

Authorized by Firehawk Development Team Status: Stable / Production Ready
