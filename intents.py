"""
Canonical intent registry for the taxi data explorer.

The architecture (from the canonical-intent design): instead of storing
hundreds of near-duplicate queries, we store a small set of CANONICAL
INTENTS, each with:
  - a stable name
  - a one-line plain-language description (also what an LLM router would
    match a user's question against, in the later LLM layer)
  - the source file it queries (the full feature set vs. the scored
    validation set -- they hold different things)
  - one tested SQL template
  - a hint for how to visualize the result

This is a STARTER SET of ~12 intents spanning the main categories, built
to prove the engine end-to-end. New intents are added by appending one
INTENTS entry -- no other code changes needed. Only columns confirmed to
exist in the real schema are referenced; payment_type-based intents are
intentionally omitted because that column isn't in the feature file.
"""

# Source files an intent can query. Different artifacts hold different
# things: the full feature set has every trip; the scored set has the
# model's anomaly flags (validation period only).
FEATURES = "silver_taxi_features.parquet"
SCORED = "validation_scored.parquet"

# Each intent: name -> dict. {chart} placeholder describes how the result
# should be drawn in the dashboard layer (bar / line / table / metric).
INTENTS = {
    # ---------------- EXECUTIVE / SUMMARY ----------------
    "executive_summary": {
        "category": "Executive",
        "description": "High-level overview: total trips, total revenue, "
                       "average fare, average trip distance.",
        "source": FEATURES,
        "chart": "metrics",
        "sql": """
            SELECT
                COUNT(*)                  AS total_trips,
                ROUND(SUM(total_amount))  AS total_revenue,
                ROUND(AVG(fare_amount),2) AS avg_fare,
                ROUND(AVG(trip_distance),2) AS avg_distance_miles
            FROM data
        """,
    },

    # ---------------- REVENUE ----------------
    "revenue_by_hour": {
        "category": "Revenue",
        "description": "Total revenue broken down by hour of the day.",
        "source": FEATURES,
        "chart": "line",
        "x": "hour_of_day", "y": "revenue",
        "sql": """
            SELECT hour_of_day,
                   ROUND(SUM(total_amount)) AS revenue
            FROM data
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,
    },
    "top_fares": {
        "category": "Revenue",
        "description": "The single highest-fare individual trips.",
        "source": FEATURES,
        "chart": "table",
        "sql": """
            SELECT PULocationID, DOLocationID, trip_distance,
                   fare_amount, total_amount
            FROM data
            ORDER BY fare_amount DESC
            LIMIT 20
        """,
    },

    # ---------------- TIPS ----------------
    "avg_tip_by_hour": {
        "category": "Tips",
        "description": "Average tip amount by hour of the day.",
        "source": FEATURES,
        "chart": "line",
        "x": "hour_of_day", "y": "avg_tip",
        "sql": """
            SELECT hour_of_day,
                   ROUND(AVG(tip_amount),2) AS avg_tip
            FROM data
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,
    },

    # ---------------- ROUTES ----------------
    "popular_routes": {
        "category": "Routes",
        "description": "The most frequently travelled pickup-to-dropoff routes.",
        "source": FEATURES,
        "chart": "table",
        "sql": """
            SELECT PULocationID, DOLocationID,
                   COUNT(*) AS trips,
                   ROUND(AVG(fare_amount),2) AS avg_fare
            FROM data
            GROUP BY PULocationID, DOLocationID
            ORDER BY trips DESC
            LIMIT 20
        """,
    },
    "expensive_routes": {
        "category": "Routes",
        "description": "Routes with the highest average fare per mile "
                       "(among routes with enough trips to be reliable).",
        "source": FEATURES,
        "chart": "table",
        "sql": """
            SELECT PULocationID, DOLocationID,
                   route_trip_count,
                   ROUND(route_avg_fare_per_mile,2) AS avg_fare_per_mile
            FROM data
            WHERE has_reliable_route_baseline = TRUE
            GROUP BY PULocationID, DOLocationID, route_trip_count,
                     route_avg_fare_per_mile
            ORDER BY avg_fare_per_mile DESC
            LIMIT 20
        """,
    },

    # ---------------- LOCATION ----------------
    "pickup_hotspots": {
        "category": "Location",
        "description": "The busiest pickup zones by trip count.",
        "source": FEATURES,
        "chart": "bar",
        "x": "PULocationID", "y": "trips",
        "sql": """
            SELECT PULocationID,
                   COUNT(*) AS trips
            FROM data
            GROUP BY PULocationID
            ORDER BY trips DESC
            LIMIT 15
        """,
    },

    # ---------------- TIME ----------------
    "busiest_hours": {
        "category": "Time",
        "description": "Number of trips by hour of the day (demand rhythm).",
        "source": FEATURES,
        "chart": "bar",
        "x": "hour_of_day", "y": "trips",
        "sql": """
            SELECT hour_of_day,
                   COUNT(*) AS trips
            FROM data
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,
    },

    # ---------------- DISTANCE / SPEED ----------------
    "longest_trips": {
        "category": "Distance",
        "description": "The longest individual trips by distance.",
        "source": FEATURES,
        "chart": "table",
        "sql": """
            SELECT PULocationID, DOLocationID, trip_distance,
                   duration_minutes, fare_amount
            FROM data
            ORDER BY trip_distance DESC
            LIMIT 20
        """,
    },
    "avg_speed_by_hour": {
        "category": "Speed",
        "description": "Average trip speed by hour, showing congestion patterns.",
        "source": FEATURES,
        "chart": "line",
        "x": "hour_of_day", "y": "avg_speed",
        "sql": """
            SELECT hour_of_day,
                   ROUND(AVG(speed_mph),1) AS avg_speed
            FROM data
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,
    },

    # ---------------- VENDOR ----------------
    "vendor_comparison": {
        "category": "Vendor",
        "description": "Compare vendors by trip count, average fare, and "
                       "average tip.",
        "source": FEATURES,
        "chart": "table",
        "sql": """
            SELECT vendor_id,
                   COUNT(*) AS trips,
                   ROUND(AVG(fare_amount),2) AS avg_fare,
                   ROUND(AVG(tip_amount),2) AS avg_tip
            FROM data
            GROUP BY vendor_id
            ORDER BY trips DESC
        """,
    },

    # ---------------- ANOMALIES (scored set) ----------------
    "top_anomalies": {
        "category": "Anomalies",
        "description": "The trips the model flagged as most unusual, with "
                       "their characteristics.",
        "source": SCORED,
        "chart": "table",
        "sql": """
            SELECT PULocationID, DOLocationID, trip_distance,
                   speed_mph, fare_per_mile,
                   ROUND(iso_forest_score,4) AS unusualness_score
            FROM data
            WHERE iso_forest_flag = TRUE
            ORDER BY iso_forest_score DESC
            LIMIT 20
        """,
    },
}


def list_intents():
    """Return intents grouped by category, for the dashboard dropdown."""
    grouped = {}
    for name, spec in INTENTS.items():
        grouped.setdefault(spec["category"], []).append(name)
    return grouped