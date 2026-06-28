# Step 2 (temporal): exact-count multi-output regression baseline under temporal validation.
#
# Reference for the Chapter 6 footnote. Reproduces a RandomForestRegressor on the exact
# resource counts (Operacionais_Man, Meios_Terrestres) under the SAME temporal split and
# the SAME 21 leakage-free features as step_temporal_validation.py.
#
# Preprocessing is SHARED, not duplicated: this imports step_temporal_validation.py and
# reuses its already-built X_train / X_test / y_train / y_test, so the leakage-free
# n_concurrent_fires (earlier same-day ignitions only), the district stats and the
# 21-feature set are identical and the numbers match the temporal pipeline.
#
# NOTE: importing that module runs the temporal pipeline end to end (it is a procedural
# script) before this regression runs. That is intentional. Run from AI_FirehawkLab:
#     python step2_model_temporal.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import step_temporal_validation as stv

# Guard the shared protocol: temporal split, no post-event leakage.
assert stv.TRAIN_YEARS == [2019, 2020, 2021, 2022], stv.TRAIN_YEARS
assert stv.TEST_YEARS == [2024, 2025], stv.TEST_YEARS

# Exact-count regression is the PRE-tier baseline, so it must NOT use district_T2_rate
# (that feature is derived from the KMeans tier labels). Drop it -> 20 features.
features = [f for f in stv.features if f != 'district_T2_rate']
assert len(features) == 20, len(features)
assert 'district_T2_rate' not in features
for banned in ('Area_Ardida_ha', 'Duracao_Horas'):
    assert banned not in features, f"{banned} must stay out of the feature set"

X_train, X_test = stv.X_train[features], stv.X_test[features]
y_train, y_test = stv.y_train, stv.y_test
y_test_arr = y_test.values

print("\n" + "=" * 60)
print("Step 2 (temporal): exact-count multi-output RF regression")
print("=" * 60)
print(f"Train {stv.TRAIN_YEARS[0]}-{stv.TRAIN_YEARS[-1]}  n={len(X_train):,}")
print(f"Test  {stv.TEST_YEARS[0]}-{stv.TEST_YEARS[-1]}   n={len(X_test):,}  (Val 2023 not used for training)")
print(f"Features ({len(features)}): {features}")
print(f"Targets: {stv.TARGETS}")

reg = RandomForestRegressor(n_estimators=300, random_state=stv.SEED, n_jobs=-1)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print("\n--- R2 on Test (2024-2025) ---")
print(f"  Global (uniform average) : {r2_score(y_test_arr, pred, multioutput='uniform_average'):.4f}")
for i, t in enumerate(stv.TARGETS):
    print(f"  {t:24s}: {r2_score(y_test_arr[:, i], pred[:, i]):.4f}")
print("\nExact-count regression yields R2 < 0 under temporal validation (worse than the")
print("test-mean predictor) — the motivation for the tier reformulation in Chapter 6.")
