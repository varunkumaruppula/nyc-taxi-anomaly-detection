import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

INPUT_PATH = "silver_taxi_features.parquet"

print(f"Loading engineered features from {INPUT_PATH}...")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} rows.")

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

CONTAMINATION = 0.01

print(f"\nTraining Isolation Forest (contamination={CONTAMINATION})...")
iso_forest = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42,
    n_estimators=200,
)

iso_forest.fit(X_train)

val_df['iso_forest_score'] = -iso_forest.score_samples(X_val)
val_df['iso_forest_flag'] = iso_forest.predict(X_val) == -1

n_flagged = val_df['iso_forest_flag'].sum()

print(f"Isolation Forest flagged {n_flagged} / {len(val_df)} validation rows "
      f"({100 * n_flagged / len(val_df):.2f}%) as anomalous.")

NUMERIC_FEATURE_COLS = FEATURE_COLS

train_means = X_train[NUMERIC_FEATURE_COLS].mean()
train_stds = X_train[NUMERIC_FEATURE_COLS].std().replace(0, np.nan)

z_scores = (X_val[NUMERIC_FEATURE_COLS] - train_means) / train_stds

val_df['baseline_zscore'] = z_scores.abs().max(axis=1)

baseline_threshold = val_df['baseline_zscore'].quantile(1 - CONTAMINATION)

val_df['baseline_flag'] = val_df['baseline_zscore'] >= baseline_threshold

n_baseline_flagged = val_df['baseline_flag'].sum()

print(f"Z-score baseline (threshold={baseline_threshold:.2f}) flagged "
      f"{n_baseline_flagged} / {len(val_df)} validation rows "
      f"({100 * n_baseline_flagged / len(val_df):.2f}%) as anomalous.")

both_flagged = (val_df['iso_forest_flag'] & val_df['baseline_flag']).sum()
only_iso = (val_df['iso_forest_flag'] & ~val_df['baseline_flag']).sum()
only_baseline = (~val_df['iso_forest_flag'] & val_df['baseline_flag']).sum()

print(f"\n--- Method Comparison ---")
print(f"Flagged by BOTH methods: {both_flagged}")
print(f"Flagged by Isolation Forest ONLY: {only_iso}")
print(f"Flagged by z-score baseline ONLY: {only_baseline}")
print(f"(Agreement rate: {100 * both_flagged / max(n_flagged, 1):.1f}% of "
      f"Isolation Forest's flags were also caught by the simple baseline)")

OUTPUT_PATH = "validation_scored.parquet"

val_df.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)

print(f"\nSaved scored validation set to: {OUTPUT_PATH}")

print("\nTop 10 highest Isolation Forest anomaly scores in validation set:")

top10 = val_df.sort_values('iso_forest_score', ascending=False)

display_cols = [
    'PULocationID',
    'DOLocationID',
    'speed_mph',
    'fare_per_mile',
    'fare_per_mile_deviation_imputed',
    'has_reliable_route_baseline',
    'iso_forest_score',
    'baseline_zscore'
]

print(top10[display_cols].head(10).to_string())
