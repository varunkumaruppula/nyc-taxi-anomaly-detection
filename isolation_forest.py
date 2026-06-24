import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# -----------------------------------------------------------------------
# INPUT: reads Step 1's output directly. Keeping these as two separate
# scripts (rather than one combined file) means Step 1 can be re-run on a
# different window or re-validated independently of anything that happens
# to the model here -- the parquet file is the contract between them.
# -----------------------------------------------------------------------
INPUT_PATH = "silver_taxi_features.parquet"

print(f"Loading engineered features from {INPUT_PATH}...")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} rows.")

# -----------------------------------------------------------------------
# FEATURE SELECTION -- a deliberate subset of the 22 columns Step 1
# produced, not "everything in the dataframe."
#
# Excluded, and why:
#   - PULocationID, DOLocationID, vendor_id: these are categorical labels
#     stored as integers/strings. Feeding them to Isolation Forest as raw
#     numeric values would imply zone 264 is "greater than" zone 140 in
#     some meaningful sense, which is false -- they're identifiers, not
#     magnitudes. Their relevant information is already captured properly
#     through route_avg_fare_per_mile and fare_per_mile_deviation_imputed,
#     which encode "how unusual is this route's pricing" without treating
#     the ID itself as an ordered quantity.
#   - pickup_time: not numeric. Already decomposed into hour_of_day and
#     day_of_week in Step 1, which ARE included below.
#   - fare_amount, total_amount, tip_amount, tolls_amount,
#     congestion_surcharge: largely redundant with fare_per_mile and
#     fare_total_gap, which already normalize these into more comparable
#     ratios. Including both the raw dollar amounts and their derived
#     ratios would let the same underlying signal count twice.
#   - has_reliable_route_baseline: CONFIRMED, via direct empirical testing
#     after the first real run of this script, to be the actual cause of
#     a severe bias: unreliable-baseline rows were only 0.90% of the
#     validation set but made up 40.44% of everything Isolation Forest
#     flagged (a 45x over-representation). Isolating the cause on
#     synthetic data showed the bias persisted even after changing the
#     imputed deviation value, and disappeared entirely (45x -> 0x) the
#     moment this flag was removed from the model's input features --
#     meaning the rare boolean itself, not the imputation choice, was
#     giving Isolation Forest's tree-splitting an easy, cheap way to
#     "isolate" these rows that had nothing to do with genuine fare or
#     behavioral anomalies. The flag is still computed in Step 1 and
#     still kept in this dataframe -- it remains genuinely useful as
#     METADATA for interpreting results (e.g. filtering or annotating
#     SHAP case studies with "this route had limited historical data"),
#     just no longer passed to the model as one of its own features.
FEATURE_COLS = [
    'hour_of_day',
    'day_of_week',
    'trip_distance',
    'duration_minutes',
    'speed_mph',
    'fare_per_mile',
    'fare_total_gap',
    'fare_per_mile_deviation_imputed',
]

missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Expected columns missing from {INPUT_PATH}: {missing_cols}. "
        f"This usually means Step 1's script version doesn't match what "
        f"this script expects -- confirm silver_taxi_features.parquet was "
        f"produced by the latest feature_engineering.py, not an older one."
    )

nan_counts = df[FEATURE_COLS].isna().sum()
if nan_counts.sum() > 0:
    raise ValueError(
        f"NaNs found in model input features, which Isolation Forest cannot "
        f"accept:\n{nan_counts[nan_counts > 0]}\n"
        f"This should not happen if fare_per_mile_deviation_imputed is being "
        f"used correctly -- check Step 1's imputation logic if this fires."
    )
print(f"Confirmed: zero NaNs across all {len(FEATURE_COLS)} selected feature columns.")

# -----------------------------------------------------------------------
# TIME-BASED SPLIT -- not random. Random shuffling would let validation-
# period information (e.g. a route's pricing pattern from a date AFTER
# the model is supposed to have been trained) leak into training, which
# would make the model look better than it actually is at the one thing
# this kind of system needs to do in practice: generalize to data it
# hasn't seen yet.
#
# 83rd percentile of pickup_time within the window approximates a 25/5
# day train/validation split for a 30-day window, without hardcoding a
# specific day count that would break if WINDOW_START/WINDOW_END in
# Step 1 are ever changed to a window of a different length.
# -----------------------------------------------------------------------
split_date = df['pickup_time'].quantile(0.83)
train_df = df[df['pickup_time'] <= split_date].copy()
val_df = df[df['pickup_time'] > split_date].copy()

assert train_df['pickup_time'].max() < val_df['pickup_time'].min(), (
    "Train/validation date ranges overlap -- this should be impossible "
    "given the split logic above, and indicates a bug if it fires."
)

print(f"\nTime-based split at {split_date}:")
print(f"  Train: {len(train_df)} rows "
      f"({train_df['pickup_time'].min()} to {train_df['pickup_time'].max()})")
print(f"  Validation: {len(val_df)} rows "
      f"({val_df['pickup_time'].min()} to {val_df['pickup_time'].max()})")

X_train = train_df[FEATURE_COLS].astype(float)
X_val = val_df[FEATURE_COLS].astype(float)

# -----------------------------------------------------------------------
# CONTAMINATION PARAMETER -- the fraction of training data Isolation
# Forest should assume is anomalous. There is no ground truth to tune
# this against (this is genuinely unsupervised data, no fraud/anomaly
# labels exist), so this is a documented business assumption, not a
# blindly-accepted library default.
#
# 1% is chosen as a starting assumption consistent with typical anomaly/
# fraud base rates in transactional data, and deliberately separate from
# the already-identified, rule-based degenerate-trip pattern (the
# zero-distance/high-fare cases from Step 1, ~2.2% of all trips) -- that
# pattern is excluded from this dataset entirely already (Step 1's
# trip_distance > 0.1 filter), so this 1% is meant to capture a DIFFERENT,
# subtler kind of anomaly among trips that already passed basic sanity
# checks, not a re-detection of the same obvious pattern a SQL filter
# already caught with perfect precision.
# -----------------------------------------------------------------------
CONTAMINATION = 0.01

print(f"\nTraining Isolation Forest (contamination={CONTAMINATION})...")
iso_forest = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42,
    n_estimators=200,
)
iso_forest.fit(X_train)

# score_samples returns higher values for normal points and lower
# (more negative) values for anomalies. Negating it here so that, for
# every score column produced in this script, HIGHER always means
# MORE anomalous -- one consistent convention, rather than having to
# remember different signs for different methods.
val_df['iso_forest_score'] = -iso_forest.score_samples(X_val)
val_df['iso_forest_flag'] = iso_forest.predict(X_val) == -1  # -1 = anomaly, 1 = normal

n_flagged = val_df['iso_forest_flag'].sum()
print(f"Isolation Forest flagged {n_flagged} / {len(val_df)} validation rows "
      f"({100 * n_flagged / len(val_df):.2f}%) as anomalous.")

# -----------------------------------------------------------------------
# BASELINE: per-feature z-score ensemble. This exists so "Isolation Forest
# found something interesting" can be checked against "...that a much
# simpler method would have missed," rather than asserted without
# comparison. For each feature, compute how many standard deviations a
# validation row sits from the TRAINING set's mean (not the validation
# set's own mean -- using the same train/val split as the real model, so
# the comparison is fair). A row's baseline anomaly score is the maximum
# absolute z-score across all features: one wildly unusual feature is
# enough to flag a row, mirroring how an analyst skimming a few columns
# by eye might actually do this without any modeling at all.
#
# All of FEATURE_COLS are numeric now that has_reliable_route_baseline has
# been moved out of the model's feature set (see note above) -- no
# boolean-column filtering needed here anymore.
# -----------------------------------------------------------------------
NUMERIC_FEATURE_COLS = FEATURE_COLS

train_means = X_train[NUMERIC_FEATURE_COLS].mean()
train_stds = X_train[NUMERIC_FEATURE_COLS].std().replace(0, np.nan)  # guard against zero-variance columns

z_scores = (X_val[NUMERIC_FEATURE_COLS] - train_means) / train_stds
val_df['baseline_zscore'] = z_scores.abs().max(axis=1)

# A z-score baseline needs its own threshold to produce a comparable flag
# rate. Setting it so the baseline flags approximately the same percentage
# of rows as Isolation Forest did, rather than an arbitrary fixed cutoff --
# this makes the "what did each method actually catch" comparison below
# meaningful (same flag budget, different selection), instead of comparing
# two methods that flagged very different total counts.
baseline_threshold = val_df['baseline_zscore'].quantile(1 - CONTAMINATION)
val_df['baseline_flag'] = val_df['baseline_zscore'] >= baseline_threshold

n_baseline_flagged = val_df['baseline_flag'].sum()
print(f"Z-score baseline (threshold={baseline_threshold:.2f}) flagged "
      f"{n_baseline_flagged} / {len(val_df)} validation rows "
      f"({100 * n_baseline_flagged / len(val_df):.2f}%) as anomalous.")

# -----------------------------------------------------------------------
# COMPARISON: how much do the two methods actually agree?
# This is the real evaluation artifact for this step -- without ground
# truth labels, "agreement with a simpler method" and "what each method
# uniquely catches" are the honest things that CAN be measured, and are
# worth reporting plainly rather than implying a precision/recall number
# that doesn't exist for this problem.
# -----------------------------------------------------------------------
both_flagged = (val_df['iso_forest_flag'] & val_df['baseline_flag']).sum()
only_iso = (val_df['iso_forest_flag'] & ~val_df['baseline_flag']).sum()
only_baseline = (~val_df['iso_forest_flag'] & val_df['baseline_flag']).sum()

print(f"\n--- Method Comparison ---")
print(f"Flagged by BOTH methods: {both_flagged}")
print(f"Flagged by Isolation Forest ONLY: {only_iso}")
print(f"Flagged by z-score baseline ONLY: {only_baseline}")
print(f"(Agreement rate: {100 * both_flagged / max(n_flagged, 1):.1f}% of "
      f"Isolation Forest's flags were also caught by the simple baseline)")

# -----------------------------------------------------------------------
# Save validation results with all scores attached, for the SHAP
# explainability work and case-study writeups planned in the next step --
# this is what those build on top of.
# -----------------------------------------------------------------------
OUTPUT_PATH = "validation_scored.parquet"
val_df.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
print(f"\nSaved scored validation set to: {OUTPUT_PATH}")

print("\nTop 10 highest Isolation Forest anomaly scores in validation set:")
top10 = val_df.sort_values('iso_forest_score', ascending=False)
display_cols = ['PULocationID', 'DOLocationID', 'speed_mph', 'fare_per_mile',
                 'fare_per_mile_deviation_imputed', 'has_reliable_route_baseline',
                 'iso_forest_score', 'baseline_zscore']
print(top10[display_cols].head(10).to_string())