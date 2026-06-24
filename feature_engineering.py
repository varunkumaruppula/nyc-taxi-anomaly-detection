import duckdb
import gcsfs
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables for GCP
load_dotenv()
bucket_name = os.getenv('GCS_BUCKET_NAME')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-key.json'

print("Authenticating with Google Cloud Storage and initializing DuckDB...")
fs = gcsfs.GCSFileSystem()
con = duckdb.connect()
con.register_filesystem(fs)

# -----------------------------------------------------------------------
# SOURCE NOTE -- read this before anything else.
# Earlier versions of this script read from gs://{bucket}/raw/*.parquet,
# which is fed by the Streaming Lakehouse's Kafka producer. That producer
# was confirmed to drop 12 real columns somewhere in its Avro
# serialization, including PULocationID and DOLocationID -- which made
# the original route-level baseline feature (the whole point of this
# project's most important feature) impossible to build.
#
# This version reads instead from a separate, manually-uploaded copy of
# the full-schema source file at gs://{bucket}/raw_source/, confirmed via
# DESCRIBE/local inspection to have the genuine 18-column TLC schema,
# including PULocationID and DOLocationID. This is a deliberate,
# documented substitution -- worth stating in ARCHITECTURE.md exactly
# this way: the Kafka pipeline's Avro schema needs to be extended to
# include the dropped columns before route-level features can be computed
# on streaming data; until that's done, this project uses the full-schema
# source directly for the feature engineering and modeling work.
#
# The vendor_id type also changed: the Kafka-streamed output had vendor_id
# as a string (e.g. "6"), while this source file has VendorID as int32
# (1, 2). Casting handles this explicitly below.
# -----------------------------------------------------------------------
RAW_SOURCE_PATH = f"gs://{bucket_name}/raw_source/yellow_tripdata.parquet"

# -----------------------------------------------------------------------
# WINDOW DEFINITION
# Documented decision, not an arbitrary number: a 30-day window is large
# enough to give per-route baselines reasonable sample sizes, while
# staying current enough that we're not blending months of stale pricing
# behavior into what counts as "normal."
#
# Confirmed via local inspection that this source file's real pickup
# timestamps span late Dec 2023 through early Feb 2024, with January 2024
# being the dominant month -- so this window is chosen to match data that
# actually exists, not a guess.
# -----------------------------------------------------------------------
WINDOW_START = "2024-01-01"
WINDOW_END = "2024-01-31"

query = f"""
WITH typed_source AS (
    -- 0. Cast and rename to a consistent internal schema. Source columns
    --    are confirmed via local inspection: VendorID (int32),
    --    tpep_pickup_datetime / tpep_dropoff_datetime (already proper
    --    TIMESTAMP type in this file, unlike the Kafka-streamed output
    --    which had these as VARCHAR -- still casting explicitly here as
    --    a defensive habit, since a wrong assumption about timestamp
    --    typing is exactly what broke an earlier version of this script).
    SELECT
        VendorID AS vendor_id,
        tpep_pickup_datetime::TIMESTAMP AS pickup_time,
        tpep_dropoff_datetime::TIMESTAMP AS dropoff_time,
        trip_distance,
        fare_amount,
        total_amount,
        tip_amount,
        tolls_amount,
        congestion_surcharge,
        passenger_count,
        PULocationID,
        DOLocationID
    FROM read_parquet('{RAW_SOURCE_PATH}', union_by_name=True)
),
raw_data AS (
    -- 1. Restrict to our defined window FIRST, so every downstream
    --    aggregate -- especially route_avg_fare -- is computed over a
    --    consistent, deliberately chosen slice of time.
    --
    --    Three filters carried over from earlier debugging on this same
    --    dataset, all confirmed necessary via direct empirical testing,
    --    not precautionary guesses:
    --      - trip_distance > 0.1: trips at or near zero distance with a
    --        nonzero fare are a real, confirmed pattern in this source
    --        data (2.2% of all rows; e.g. a $5000 fare at PULocationID =
    --        DOLocationID = 264, the TLC's reserved "Unknown" zone code,
    --        almost certainly GPS/location capture failures). Including
    --        them in a per-mile ratio corrupts any average computed over
    --        the group they fall into. They are captured separately
    --        below in degenerate_trips.parquet, not discarded.
    --      - pickup_time bounded to a plausible real range: this file
    --        contains at least one corrupted timestamp from the year
    --        2002, alongside genuine Jan 2024 data. The WINDOW_START/
    --        WINDOW_END bounds already exclude this, since 2002 falls
    --        outside the window -- noted here so the reason isn't a
    --        mystery later.
    --      - PULocationID/DOLocationID NOT IN (264, 265): these are TLC's
    --        reserved "Unknown" (264) and "Outside of NYC" (265) zone
    --        codes, not real geographic zones. A "route" ending in 265
    --        could represent a 3-mile suburban trip or a 60-mile
    --        out-of-state trip -- wildly different fare-per-mile ratios
    --        all pooled under one route_avg_fare_per_mile baseline. This
    --        was found, not assumed: after Step 2's first real model run,
    --        empirical testing confirmed DOLocationID==265 rows were
    --        40.73x over-represented among flagged anomalies relative to
    --        their 0.38% overall rate (17.79x when combined with zone
    --        264), even after fixing a separate, unrelated bias from the
    --        has_reliable_route_baseline flag. Excluding these zones from
    --        the route-baseline population, rather than letting a
    --        geographically incoherent "route" masquerade as a normal
    --        one, fixes this at its actual source. As with the distance
    --        filter, these rows are captured separately below
    --        (degenerate_trips.parquet) rather than silently discarded --
    --        they're a different category of excluded data (geographic
    --        coding limitation, not GPS failure), worth keeping visible
    --        as such rather than merged into the same bucket as the
    --        zero-distance trips.
    SELECT *
    FROM typed_source
    WHERE trip_distance > 0.1
      AND fare_amount > 0
      AND total_amount > 0
      AND pickup_time IS NOT NULL
      AND dropoff_time IS NOT NULL
      AND PULocationID IS NOT NULL
      AND DOLocationID IS NOT NULL
      AND PULocationID NOT IN (264, 265)
      AND DOLocationID NOT IN (264, 265)
      AND pickup_time >= TIMESTAMP '{WINDOW_START}'
      AND pickup_time <  TIMESTAMP '{WINDOW_END}' + INTERVAL 1 DAY
),
time_features AS (
    -- 2. Derive time-based features and filter out impossible 0-minute trips
    SELECT
        *,
        date_diff('minute', pickup_time, dropoff_time) AS duration_minutes,
        extract('hour' FROM pickup_time) AS hour_of_day,
        extract('isodow' FROM pickup_time) AS day_of_week
    FROM raw_data
    WHERE date_diff('minute', pickup_time, dropoff_time) > 0
),
engineered_features AS (
    -- 3. Derive behavioral and spatial-context features.
    SELECT
        vendor_id,
        pickup_time,
        duration_minutes,
        trip_distance,
        fare_amount,
        total_amount,
        tip_amount,
        tolls_amount,
        congestion_surcharge,
        passenger_count,
        hour_of_day,
        day_of_week,
        PULocationID,
        DOLocationID,

        ROUND((fare_amount / trip_distance), 2) AS fare_per_mile,
        ROUND((trip_distance / (duration_minutes / 60.0)), 2) AS speed_mph,

        -- total_amount - fare_amount captured tolls/surcharges/tips as one
        -- undifferentiated number in the earlier vendor/hour version of
        -- this script, because those component columns weren't available
        -- then. They are now, so the gap is broken into its real parts
        -- instead of left as one opaque figure.
        ROUND((total_amount - fare_amount), 2) AS fare_total_gap,

        -- THE FEATURE THIS REWRITE EXISTS TO RESTORE: average fare-per-mile
        -- for this specific pickup-to-dropoff zone pair, giving each trip
        -- a genuine "what's normal for this exact route" baseline rather
        -- than the coarser vendor/hour substitute used previously.
        --
        -- Computed as SUM(fare_amount)/SUM(trip_distance) per route group
        -- (a ratio of sums), NOT AVG(fare_amount/trip_distance) (a mean of
        -- ratios) -- confirmed in earlier debugging on this same dataset
        -- that the latter lets a handful of extreme individual trips
        -- dominate a group's average even with hundreds of trips in that
        -- group. The distance filter above already removes the most
        -- extreme cases, but the ratio-of-sums approach is kept as a
        -- second, independent layer of protection against the same
        -- failure mode resurfacing with a different set of outlier trips.
        ROUND(SUM(fare_amount) OVER (
            PARTITION BY PULocationID, DOLocationID
        ) / SUM(trip_distance) OVER (
            PARTITION BY PULocationID, DOLocationID
        ), 2) AS route_avg_fare_per_mile,

        -- Sample size behind that baseline. A route with only a handful of
        -- trips in this window has a much noisier "average" than a
        -- high-volume route -- this is what lets that distinction be made
        -- explicitly later (in EVALUATION.md and in any modeling decision
        -- about whether to trust or downweight low-volume route baselines)
        -- rather than treating every route's baseline as equally reliable.
        COUNT(*) OVER (
            PARTITION BY PULocationID, DOLocationID
        ) AS route_trip_count

    FROM time_features
)
-- 4. Final selection and contextual deviation.
--
--    LOW-VOLUME ROUTE HANDLING (Option 1, decided after reviewing the
--    first real run of this query): 1.0% of rows sat on a route baseline
--    with fewer than 5 trips in the window -- some routes had only 2.
--    A baseline built on 2 trips isn't a meaningful "what's normal here"
--    comparison, and feeding it into fare_per_mile_deviation as if it
--    were equally reliable as a 5000-trip route's baseline would let
--    Isolation Forest learn from noise rather than signal on exactly
--    those rows.
--
--    The fix applies only to fare_per_mile_deviation, not the whole row:
--    where route_trip_count < MIN_ROUTE_TRIPS, the deviation is set to
--    NULL rather than computed from an untrustworthy baseline. The row
--    itself, and its other features (speed_mph, fare_total_gap,
--    route_avg_fare_per_mile, route_trip_count itself), are NOT dropped --
--    only this one comparison is withheld where it can't be trusted.
--    route_trip_count stays in the output specifically so this decision
--    is visible and auditable in the saved data, not hidden inside the
--    query logic.
--
--    MODEL-READINESS LAYER (added after deciding how Step 2 will consume
--    this column): Isolation Forest cannot accept nulls directly, so the
--    raw nullable fare_per_mile_deviation above is NOT what gets fed to
--    the model as-is. Two additional columns are added instead of just
--    filling the null in place, specifically to avoid erasing the
--    distinction between "this trip's deviation is genuinely close to its
--    route's baseline" and "we don't have a trustworthy baseline for this
--    route at all":
--      - has_reliable_route_baseline: TRUE when route_trip_count >= 5,
--        FALSE otherwise. This is what Step 2 should pass to the model as
--        its own feature, so Isolation Forest can learn that a FALSE here
--        means "treat the imputed deviation below with less weight,"
--        rather than silently treating every row's deviation as equally
--        informative.
--      - fare_per_mile_deviation_imputed: the actual model input. Equal to
--        fare_per_mile_deviation where the baseline is reliable; 0 where
--        it isn't (0 chosen as a neutral "no evidence of deviation"
--        value, rather than guessing a non-zero number with no basis).
--        The original nullable fare_per_mile_deviation column is KEPT
--        alongside this, unchanged, for any reporting or case-study work
--        where showing an honest null is preferable to a 0 that could be
--        misread as a real measured deviation.
SELECT
    * EXCLUDE (fare_per_mile, route_avg_fare_per_mile),
    fare_per_mile,
    route_avg_fare_per_mile,
    CASE
        WHEN route_trip_count < 5 THEN NULL
        ELSE ROUND((fare_per_mile - route_avg_fare_per_mile), 2)
    END AS fare_per_mile_deviation,
    (route_trip_count >= 5) AS has_reliable_route_baseline,
    CASE
        WHEN route_trip_count < 5 THEN 0.0
        ELSE ROUND((fare_per_mile - route_avg_fare_per_mile), 2)
    END AS fare_per_mile_deviation_imputed
FROM engineered_features
ORDER BY pickup_time;
"""

print(f"Executing feature engineering query for window {WINDOW_START} to {WINDOW_END}...")
features_df = con.execute(query).df()

print("\n--- Feature Engineering Complete ---")
print(f"Extracted {len(features_df)} valid trips within the defined window.")

if len(features_df) == 0:
    print("\nWARNING: zero rows returned. Check RAW_SOURCE_PATH is correct "
          "and that WINDOW_START/WINDOW_END overlap this file's real date "
          "range -- a mismatch here fails silently, not with an error.")
else:
    low_volume_routes = features_df[features_df['route_trip_count'] < 5]
    print(f"Rows on low-volume routes (<5 trips in window): "
          f"{len(low_volume_routes)} "
          f"({100 * len(low_volume_routes) / len(features_df):.1f}% of total)")
    print(f"  -> fare_per_mile_deviation set to NULL for these rows "
          f"(route_avg_fare_per_mile not trustworthy enough to compare against).")
    nulled_deviation_count = features_df['fare_per_mile_deviation'].isna().sum()
    print(f"  -> Confirmed: {nulled_deviation_count} rows have a NULL "
          f"fare_per_mile_deviation in the saved output.")

    print("\nSample Features (First 5 rows):")
    cols_to_show = ['PULocationID', 'DOLocationID', 'duration_minutes', 'trip_distance',
                     'speed_mph', 'fare_per_mile', 'fare_total_gap',
                     'route_avg_fare_per_mile', 'route_trip_count', 'fare_per_mile_deviation']
    print(features_df[cols_to_show].head())

    local_save_path = "silver_taxi_features.parquet"
    features_df.to_parquet(local_save_path, engine='pyarrow', index=False)
    print(f"\nSaved engineered features to local disk: {local_save_path}")

# -----------------------------------------------------------------------
# EXCLUDED TRIPS: rows excluded from the main feature set above, for two
# DISTINCT reasons, each kept separately labeled rather than merged into
# one undifferentiated bucket:
#   - 'near_zero_distance': trip_distance <= 0.1 miles with a nonzero
#     fare. Confirmed a real pattern in the genuine source data (2.2% of
#     all rows), not a pipeline artifact -- many cluster at PULocationID
#     = DOLocationID = 264, suggesting GPS/location capture failures.
#   - 'reserved_zone_code': PULocationID or DOLocationID is 264 (Unknown)
#     or 265 (Outside of NYC) without necessarily having near-zero
#     distance. Confirmed via empirical testing in Step 2 that including
#     these in route-baseline calculations caused a 40.73x (zone 265
#     alone) / 17.79x (264+265 combined) over-representation in flagged
#     anomalies, because these codes don't represent a single coherent
#     geographic "route" the way real zone pairs do.
# A row can match both reasons; reason_excluded reflects the first one
# checked, prioritizing the distance reason since it's the more severe
# data quality issue of the two.
# Captured here as their own output rather than silently discarded, so
# they remain available for a dedicated rule-based anomaly flag later.
# -----------------------------------------------------------------------
degenerate_query = f"""
WITH typed_source AS (
    SELECT
        VendorID AS vendor_id,
        tpep_pickup_datetime::TIMESTAMP AS pickup_time,
        tpep_dropoff_datetime::TIMESTAMP AS dropoff_time,
        trip_distance,
        fare_amount,
        total_amount,
        PULocationID,
        DOLocationID
    FROM read_parquet('{RAW_SOURCE_PATH}', union_by_name=True)
)
SELECT
    vendor_id,
    pickup_time,
    dropoff_time,
    trip_distance,
    fare_amount,
    total_amount,
    PULocationID,
    DOLocationID,
    extract('hour' FROM pickup_time) AS hour_of_day,
    extract('isodow' FROM pickup_time) AS day_of_week,
    CASE
        WHEN trip_distance <= 0.1 THEN 'near_zero_distance'
        WHEN PULocationID IN (264, 265) OR DOLocationID IN (264, 265) THEN 'reserved_zone_code'
        ELSE 'other'
    END AS reason_excluded
FROM typed_source
WHERE fare_amount > 0
  AND total_amount > 0
  AND pickup_time IS NOT NULL
  AND dropoff_time IS NOT NULL
  AND pickup_time >= TIMESTAMP '{WINDOW_START}'
  AND pickup_time <  TIMESTAMP '{WINDOW_END}' + INTERVAL 1 DAY
  AND (
        trip_distance <= 0.1
        OR PULocationID IN (264, 265)
        OR DOLocationID IN (264, 265)
      )
ORDER BY pickup_time;
"""

print(f"\nExecuting excluded-trips query for the same window "
      f"({WINDOW_START} to {WINDOW_END})...")
degenerate_df = con.execute(degenerate_query).df()

print(f"Captured {len(degenerate_df)} excluded trips in this window, by reason:")
if len(degenerate_df) > 0:
    print(degenerate_df['reason_excluded'].value_counts().to_string())

if len(degenerate_df) > 0:
    print("\nSample excluded trips (highest fare first):")
    print(degenerate_df.sort_values('fare_amount', ascending=False)
          [['vendor_id', 'hour_of_day', 'trip_distance', 'fare_amount',
            'PULocationID', 'DOLocationID', 'reason_excluded']]
          .head())

    degenerate_save_path = "degenerate_trips.parquet"
    degenerate_df.to_parquet(degenerate_save_path, engine='pyarrow', index=False)
    print(f"\nSaved excluded trips to local disk: {degenerate_save_path}")
else:
    print("No excluded trips found in this window -- nothing to save.")