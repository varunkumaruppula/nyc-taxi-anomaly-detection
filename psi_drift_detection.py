import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# INPUT: Step 1's feature output. PSI compares the distribution the model
# was trained on (reference) against new incoming data (current), one
# feature at a time, to answer: "has the world drifted away from what the
# model learned as normal?" -- the signal that a model needs retraining.
# -----------------------------------------------------------------------
INPUT_PATH = "silver_taxi_features.parquet"

print(f"Loading engineered features from {INPUT_PATH}...")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} rows.")

# Same feature set and same time-based split as Steps 2 and 3, so the
# "reference" here is genuinely what the model trained on.
MODEL_FEATURE_COLS = [
    'hour_of_day',
    'day_of_week',
    'trip_distance',
    'duration_minutes',
    'speed_mph',
    'fare_per_mile',
    'fare_total_gap',
    'fare_per_mile_deviation_imputed',
]

# -----------------------------------------------------------------------
# PSI is computed on a DELIBERATELY NARROWER set than the model uses:
# the calendar features (day_of_week, hour_of_day) are excluded from
# drift monitoring, and this exclusion was driven by a real finding, not
# a precaution.
#
# On the genuine (unsimulated) data, day_of_week produced a PSI of 2.52 --
# enormous. Investigation confirmed the cause: the time-based split cuts
# a ~5-day current window (Jan 26-31) out of a ~26-day reference window,
# and those 5 contiguous days simply don't contain every weekday in the
# same proportions a full month does. In this data, the current window
# contains NO Thursday at all (day 4: 17.5% of the reference period, 0%
# of the current period). PSI was correctly reporting that the day-of-week
# distributions differ -- but that difference is a calendar-windowing
# artifact, not the behavioral drift PSI is meant to catch. You don't
# retrain a fare-anomaly model because a 5-day window happened to miss a
# Thursday.
#
# Drift monitoring therefore focuses on the BEHAVIORAL features -- the
# ones whose distribution shifting genuinely signals the model may have
# gone stale (fares, distances, speeds, durations). Calendar features
# remain valid MODEL inputs (a 3am trip really is different from a 3pm
# trip); they're just not meaningful DRIFT signals on a short, non-
# calendar-aligned window.
# -----------------------------------------------------------------------
PSI_MONITORED_COLS = [
    'trip_distance',
    'duration_minutes',
    'speed_mph',
    'fare_per_mile',
    'fare_total_gap',
    'fare_per_mile_deviation_imputed',
]
FEATURE_COLS = PSI_MONITORED_COLS  # PSI reporting below iterates over this

split_date = df['pickup_time'].quantile(0.83)
reference_df = df[df['pickup_time'] <= split_date].copy()  # what the model trained on
current_df = df[df['pickup_time'] > split_date].copy()     # new data arriving after

print(f"\nReference period (model's training distribution): {len(reference_df)} rows "
      f"({reference_df['pickup_time'].min()} to {reference_df['pickup_time'].max()})")
print(f"Current period (new incoming data): {len(current_df)} rows "
      f"({current_df['pickup_time'].min()} to {current_df['pickup_time'].max()})")


# -----------------------------------------------------------------------
# PSI CALCULATION
# Validated against known cases before use: ~0 for identical
# distributions, small for minor shifts, >0.2 for significant drift, and
# numerically stable when current data falls entirely outside the
# reference range (the divide-by-zero / log(0) edge case).
#
# Bin edges come from the REFERENCE distribution's deciles -- bins are
# defined by what the model considers "normal," then we measure how the
# current data falls into those reference bins. Outer edges are extended
# to +/- inf so no current value, however extreme, falls outside all bins.
# A small epsilon replaces any zero proportion to keep the log and
# division finite.
# -----------------------------------------------------------------------
def calculate_psi(reference, current, bins=10):
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 1e-6
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))


# Standard PSI interpretation thresholds.
PSI_MODERATE = 0.1   # below this: no meaningful drift
PSI_SIGNIFICANT = 0.2  # at/above this: significant drift, retraining warranted


def psi_verdict(psi):
    if psi < PSI_MODERATE:
        return "stable"
    elif psi < PSI_SIGNIFICANT:
        return "moderate drift (watch)"
    else:
        return "SIGNIFICANT drift (retrain)"


def report_psi(reference_df, current_df, label):
    print(f"\n=== PSI Report: {label} ===")
    print(f"{'feature':<35} {'PSI':>8}   verdict")
    print("-" * 70)
    any_significant = False
    results = {}
    for col in FEATURE_COLS:
        psi = calculate_psi(reference_df[col].values, current_df[col].values)
        results[col] = psi
        verdict = psi_verdict(psi)
        if psi >= PSI_SIGNIFICANT:
            any_significant = True
        print(f"{col:<35} {psi:>8.4f}   {verdict}")
    print("-" * 70)
    if any_significant:
        print(">>> ACTION: at least one feature shows significant drift. "
              "In production this would trigger a retraining job.")
    else:
        print(">>> No significant drift detected. Model can continue serving "
              "without retraining.")
    return results


# -----------------------------------------------------------------------
# PART A: REAL DRIFT CHECK -- reference vs. genuine current data.
# This is the honest baseline: is there actually drift between the early
# and late parts of the same January window? Likely minimal, since it's
# all one month of the same year -- which is the correct, expected result
# and sets up the contrast for the simulated event below.
# -----------------------------------------------------------------------
real_results = report_psi(reference_df, current_df,
                           "Reference vs. genuine current data (no simulation)")


# -----------------------------------------------------------------------
# PART B: SIMULATED DRIFT EVENT, AT ESCALATING INTENSITIES.
# Real drift detection is unconvincing if nothing ever drifts. Here we
# deliberately inject a known, realistic distribution shift into copies
# of the current data -- at three escalating magnitudes -- and confirm
# PSI not only catches it but responds PROPORTIONALLY: a small shift
# should read "moderate," a large one "significant." That graded response
# (stable -> moderate -> significant as the injected shift grows) is a
# stronger demonstration than a single binary "drift detected," because
# it shows the detector tracks drift MAGNITUDE, not just presence.
#
# Simulated scenario: a fare increase (e.g. a fare-policy change or
# sustained surge pricing) applied across all fare-derived features
# (fare_per_mile, fare_total_gap, fare_per_mile_deviation_imputed),
# exactly as a real coherent fare change would propagate -- so we're
# testing detection of a realistic shift, not noise added to one column.
# -----------------------------------------------------------------------
FARE_FEATURES = ['fare_per_mile', 'fare_total_gap', 'fare_per_mile_deviation_imputed']
SHIFT_LEVELS = [0.20, 0.35, 0.50]  # 20%, 35%, 50% fare increases

escalating_results = {}
for shift in SHIFT_LEVELS:
    print("\n" + "=" * 70)
    print(f"SIMULATING DRIFT: a {int(shift*100)}% fare increase across current data")
    print("=" * 70)
    drifted_df = current_df.copy()
    multiplier = 1.0 + shift
    for feat in FARE_FEATURES:
        drifted_df[feat] = drifted_df[feat] * multiplier
    escalating_results[shift] = report_psi(
        reference_df, drifted_df,
        f"Reference vs. SIMULATED +{int(shift*100)}% fare data"
    )


# -----------------------------------------------------------------------
# SUMMARY: the escalation contrast, side by side. The core demonstration
# is that fare-derived features climb monotonically through the PSI
# thresholds as the injected shift grows, while non-fare features stay
# flat across all levels -- proving PSI isolates WHAT drifted AND tracks
# HOW MUCH.
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("ESCALATION CONTRAST (this is the core demonstration)")
print("=" * 70)
header = f"{'feature':<33} {'real':>8}"
for shift in SHIFT_LEVELS:
    header += f" {'+' + str(int(shift*100)) + '%':>9}"
print(header)
print("-" * 70)
for col in FEATURE_COLS:
    line = f"{col:<33} {real_results[col]:>8.4f}"
    for shift in SHIFT_LEVELS:
        line += f" {escalating_results[shift][col]:>9.4f}"
    print(line)
print("-" * 70)
print("Read across each fare-derived feature's row: PSI should climb from "
      "'stable' through 'moderate' to 'significant (retrain)' as the shift "
      "grows, while non-fare features (hour_of_day, trip_distance, etc.) "
      "stay flat across every column -- confirming PSI isolates WHAT drifted "
      "and tracks HOW MUCH it drifted.")

# Save all result sets for the dashboard / writeup later.
summary_rows = []
for col in FEATURE_COLS:
    row = {'feature': col, 'psi_real': real_results[col],
           'verdict_real': psi_verdict(real_results[col])}
    for shift in SHIFT_LEVELS:
        key = f"psi_shift_{int(shift*100)}pct"
        row[key] = escalating_results[shift][col]
        row[f"verdict_shift_{int(shift*100)}pct"] = psi_verdict(escalating_results[shift][col])
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
OUTPUT_PATH = "psi_drift_report.parquet"
summary.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved PSI drift report to: {OUTPUT_PATH}")