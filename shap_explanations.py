import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap

# -----------------------------------------------------------------------
# INPUT: this script re-trains the same Isolation Forest configuration
# from Step 2 (same features, same contamination, same random_state) so
# that SHAP's TreeExplainer has direct access to the underlying model
# object. Step 2's validation_scored.parquet already has iso_forest_score
# and iso_forest_flag, but TreeExplainer needs the live sklearn model
# itself, not just its saved predictions -- retraining here with an
# identical configuration is simpler and more transparent than trying to
# serialize/deserialize the model between scripts, and is cheap enough
# (a few seconds) that the duplication isn't a meaningful cost.
# -----------------------------------------------------------------------
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
# has_reliable_route_baseline deliberately excluded from the model here too
# -- see EVALUATION.md for the full reasoning (it was confirmed to cause a
# 45x flagging bias when included, traced to how a rare boolean feature
# interacts with Isolation Forest's tree-splitting).

missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Expected columns missing from {INPUT_PATH}: {missing_cols}")

split_date = df['pickup_time'].quantile(0.83)
train_df = df[df['pickup_time'] <= split_date].copy()
val_df = df[df['pickup_time'] > split_date].copy()

X_train = train_df[FEATURE_COLS].astype(float)
X_val = val_df[FEATURE_COLS].astype(float)

CONTAMINATION = 0.01
print(f"Retraining Isolation Forest (same config as Step 2, contamination={CONTAMINATION})...")
iso_forest = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42,
    n_estimators=200,
)
iso_forest.fit(X_train)

val_df['iso_forest_score'] = -iso_forest.score_samples(X_val)
val_df['iso_forest_flag'] = iso_forest.predict(X_val) == -1

n_flagged = val_df['iso_forest_flag'].sum()
print(f"Confirmed: {n_flagged} flagged rows in validation set "
      f"(should match Step 2's reported count if the configuration is identical).")

# -----------------------------------------------------------------------
# SHAP EXPLANATIONS
#
# TreeExplainer is used because Isolation Forest is a tree-based model --
# this is dramatically faster and exact (not sampling-based) compared to
# the generic KernelExplainer, which would be impractically slow at this
# row count.
#
# SIGN CONVENTION: raw SHAP values for IsolationForest follow
# score_samples' own convention, where LOWER values mean MORE anomalous.
# This was confirmed directly (not assumed) by testing against a row with
# a deliberately injected extreme anomaly: the raw SHAP value for the
# obviously-anomalous feature came back strongly NEGATIVE. Negating SHAP
# values here makes positive = "pushed this row toward being MORE
# anomalous", consistent with the already-negated iso_forest_score
# convention from Step 2 -- one consistent sign convention across the
# whole project, rather than two different ones that would be easy to
# mix up when writing case studies.
# -----------------------------------------------------------------------
print("\nComputing SHAP values for all validation rows...")
explainer = shap.TreeExplainer(iso_forest)
raw_shap_values = explainer.shap_values(X_val)
shap_values = -raw_shap_values  # see sign convention note above

shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLS],
                        index=val_df.index)
val_df = pd.concat([val_df, shap_df], axis=1)

# -----------------------------------------------------------------------
# CASE STUDY SELECTION
#
# Per EVALUATION.md's documented residual finding (a 9.10x over-
# representation of has_reliable_route_baseline == False rows among
# flagged anomalies, only partially explained by genuine signal), primary
# case studies are deliberately drawn from RELIABLE-baseline flagged rows
# first. This isn't hiding the unreliable-baseline rows -- they're still
# in the full output below -- it's making sure the headline examples are
# the unambiguously trustworthy ones, with any unreliable-baseline example
# presented separately and explicitly labeled as such.
# -----------------------------------------------------------------------
flagged = val_df[val_df['iso_forest_flag']].copy()
reliable_flagged = flagged[flagged['has_reliable_route_baseline']]
unreliable_flagged = flagged[~flagged['has_reliable_route_baseline']]

print(f"\nFlagged rows with reliable route baseline: {len(reliable_flagged)}")
print(f"Flagged rows with UNRELIABLE route baseline: {len(unreliable_flagged)} "
      f"(see EVALUATION.md before using these as primary examples)")

shap_cols = [f"shap_{c}" for c in FEATURE_COLS]

def print_case_study(row, label):
    print(f"\n--- {label} ---")
    print(f"Route: PULocationID {row['PULocationID']} -> DOLocationID {row['DOLocationID']}")
    print(f"Reliable baseline: {row['has_reliable_route_baseline']}  "
          f"(route_trip_count: {row['route_trip_count']})")
    print(f"iso_forest_score: {row['iso_forest_score']:.4f}")
    print(f"Raw values: speed_mph={row['speed_mph']:.2f}, "
          f"fare_per_mile={row['fare_per_mile']:.2f}, "
          f"fare_total_gap={row['fare_total_gap']:.2f}, "
          f"fare_per_mile_deviation_imputed={row['fare_per_mile_deviation_imputed']:.2f}")
    print("Feature contributions (positive = pushed toward MORE anomalous), "
          "sorted by magnitude:")
    contributions = row[shap_cols].sort_values(key=abs, ascending=False)
    for feat, val in contributions.items():
        clean_name = feat.replace('shap_', '')
        print(f"  {clean_name}: {val:+.4f}")

print("\n=== TOP 5 CASE STUDIES (reliable route baseline) ===")
top5_reliable = reliable_flagged.sort_values('iso_forest_score', ascending=False).head(5)
for i, (_, row) in enumerate(top5_reliable.iterrows(), 1):
    print_case_study(row, f"Case Study {i} (reliable baseline)")

if len(unreliable_flagged) > 0:
    print("\n=== 1 ADDITIONAL CASE STUDY (unreliable route baseline, presented separately) ===")
    top_unreliable = unreliable_flagged.sort_values('iso_forest_score', ascending=False).head(1)
    for _, row in top_unreliable.iterrows():
        print_case_study(row, "Unreliable-baseline example -- interpret per EVALUATION.md")

# -----------------------------------------------------------------------
# Save everything, including per-feature SHAP contributions, so case
# studies can be written up later directly from this file rather than
# re-running the model.
# -----------------------------------------------------------------------
OUTPUT_PATH = "validation_with_shap.parquet"
val_df.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
print(f"\nSaved validation set with SHAP contributions to: {OUTPUT_PATH}")

# -----------------------------------------------------------------------
# GLOBAL FEATURE IMPORTANCE: mean absolute SHAP value per feature, across
# all flagged rows. This answers a different, complementary question to
# the individual case studies above -- not "why was this ONE trip
# flagged" but "which features matter most across ALL flagged anomalies
# in general."
# -----------------------------------------------------------------------
print("\n=== Global feature importance (mean |SHAP value| across flagged rows) ===")
importance = flagged[shap_cols].abs().mean().sort_values(ascending=False)
for feat, val in importance.items():
    clean_name = feat.replace('shap_', '')
    print(f"  {clean_name}: {val:.4f}")