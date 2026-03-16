# PROJECT CONTEXT — Forest Fire Resource Predictor
## For Claude Code Session Recovery

---

## Project Overview
Final bachelor project: ML pipeline predicting operational resources (Operacionais_Man, MeiosTerrestres, MeiosAereos) needed for forest fire incidents in Portugal. Dataset merges historical XML data from ICNF (fire weather indices like FWI, terrain data like DECLIVEMEDIO) with operational CSV datasets using NCCO/id keys.

## Current Model State
- **Algorithm**: Random Forest (RF) preferred — GBR was tested but performed worse.
- **Architecture**: Separate classifier/regressor for aerial assets (MeiosAereos), joint model for (Operacionais_Man, MeiosTerrestres).
- **DISTRITO is used as an encoded feature** (this helped performance).
- **R² is low** for the (Operacionais_Man, MeiosTerrestres) targets.
- Environmental + fire data (FWI, weather, etc.) showed very low correlation with targets.
- The bottleneck is **missing operational/geopolitical context** — we don't have data on how many staff/vehicles each district actually has available.

## Key Results Files (already in project)
- `step3a_kmeans_results.txt` — K-means exploration on target values for range-based forecasting
- `step3_gbr_results.txt` — GBR test results (worse than RF)
- `step2_aerial_results.txt` — Aerial assets separate model results

---

## NEXT STEPS — Implementation Plan

### PHASE 1: Feature Engineering (DO THIS FIRST)
Add new features derived from the dataset itself to capture operational context.

#### Feature 1: `n_concurrent_fires`
- **What**: For each fire incident, count how many OTHER fires were active in the same DISTRITO on the same date.
- **Why**: When multiple fires burn simultaneously, resources are split between them. This is a strong signal for how many operatives each fire gets.
- **How**: Group by (DISTRITO, date), count incidents per group, merge back, subtract 1 (the fire itself).
- **Leakage risk**: NONE — this uses only input data (date + district), not the target.

#### Feature 2: `district_max_single_incident`
- **What**: For each DISTRITO, the maximum Operacionais_Man ever allocated to a single incident.
- **Why**: Proxy for the "ceiling" of how many operatives that district can attract for one fire.
- **How**: Compute ONLY on train set, merge to test set by DISTRITO.
- **Leakage risk**: YES — derived from target. Must compute on train only.

#### Feature 3: `district_median_single_incident`
- **What**: For each DISTRITO, the median Operacionais_Man per incident.
- **Why**: Proxy for the "typical" allocation in that district. Complements the max.
- **How**: Compute ONLY on train set, merge to test set by DISTRITO.
- **Leakage risk**: YES — same as above, train only.

#### After implementing: run RF model, measure R², compare with previous results, log delta.

---

### PHASE 2: Target Ranges (DO THIS SECOND, after Phase 1 results are stable)
Convert continuous target predictions into range-based forecasts.

#### Step 1: K-Means on targets
- Run k-means on Operacionais_Man (and MeiosTerrestres) from train set.
- Test k=3, k=4, k=5. Evaluate with silhouette score / elbow method.
- The k-means results from `step3a_kmeans_results.txt` are the starting point.

#### Step 2: Convert to classification
- Each incident gets a range label instead of exact value.
- Switch from RF Regressor to RF Classifier.
- Measure accuracy, F1-score per class, confusion matrix.

#### Step 3: Compare
- Translate range centroids back to numeric values for comparison with old R².
- Document: "R² base = X, R² with new features = Y, classification accuracy = Z".

---

## EXTERNAL DATA SOURCES (for future enrichment, lower priority)

### Source A: INE — Number of Firefighters by NUTS region
- Annual headcount of firefighters disaggregated by NUTS II and NUTS III regions.
- JSON via REST API:
  ```
  # Data endpoint
  https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0013371&lang=PT
  # Metadata endpoint
  https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp?varcd=0013371&lang=PT
  ```
- **Use**: Map NUTS III → DISTRITO, create feature `firefighters_in_region`.
- **Limitation**: Headcount only (not vehicles/aerial), NUTS III granularity.
- **Applies to**: Operacionais_Man only (but could estimate vehicle limits via minimum crew ratios).

### Source B: RCAAP Academic Paper — RNBP 2018 Tables
- Master's dissertation with tables extracted from National Firefighter Census (2018).
- PDF URL: `https://comum.rcaap.pt/server/api/core/bitstreams/27aa09b6-4167-43c1-ab77-dc59a6fdc601/content`
- **Contains per-district data**:
  - Table 6: Active-duty firefighters per district (18 districts)
  - Table 7: Firefighter-to-population ratio per district
  - Table 9: Minimum vehicle allocation per fire brigade type (regulatory)
  - Table 10: Minimum crew per vehicle and response times
  - Table 33: Number of fire stations per district
- **Use**: Create lookup {district → staff count, station count, min vehicles}. Static 2018 data.
- **Limitation**: Snapshot from 2018, not time series. Vehicles are regulatory minimums, not actual inventory.

### Source C: ANEPC National Aggregates (context/validation only)
- 434 fire brigade units nationwide, ~30,433 total firefighters.
- DECIR 2026: 15,064 operatives, 3,438 vehicles, 76 aerial assets at peak (Delta level).
- DECIR 2025: 11,161 operatives, 2,417 vehicles, 76 aerial at Delta.
- **Use**: Sanity checks only (e.g., model should never predict >15,000 ops for a single fire). Not usable as features (no district-level breakdown).

---

## IMPORTANT NOTES

### About resource sharing between districts
- Portugal's fire response follows the SGO (Operations Management System) with escalation phases.
- Initial attack uses local resources. When that fails, the district command (CDOS) requests reinforcement from the national command (CNOS).
- CNOS sends inter-district reinforcement groups (GRIF, GRUATA) from other districts.
- There is NO fixed threshold (e.g., ">500 hectares") that triggers inter-district sharing. It's a dynamic operational decision.
- **Implication for the model**: High values of Operacionais_Man likely include inter-district reinforcements. The `district_max_single_incident` feature captures this implicitly.

### About the separate aerial model
- MeiosAereos has many zeros (most fires don't get aerial assets).
- A separate classifier (fire gets aerial? yes/no) + regressor (if yes, how many?) was already implemented and should be maintained.

### Logging
- All changes must be logged with before/after R² comparisons.
- Each phase should be tested independently before combining.
