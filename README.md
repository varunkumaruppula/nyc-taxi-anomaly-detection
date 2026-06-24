# NYC Taxi Anomaly Detection Engine

A self-contained, end-to-end unsupervised anomaly detection system for NYC
taxi trip data, built entirely on local hardware (8GB RAM laptop) plus
free-tier cloud storage — no API keys, no paid services. Spans data
engineering, data science, ML explainability, ML lifecycle monitoring, and
a self-explaining AI-assisted dashboard.

## What it does

Flags individual taxi trips with unusual fare, speed, distance, or duration
characteristics relative to learned norms for their specific route, explains
*why* each was flagged, and detects when the model itself has gone stale and
needs retraining — all without any labeled fraud data (fully unsupervised).

## Data

NYC TLC `yellow_tripdata.parquet`, January 2024. ~2.86M raw rows, reduced to
**2,831,892 rows** after quality filtering.

## Architecture

```
Stage 1: feature_engineering.py   -> silver_taxi_features.parquet
Stage 2: isolation_forest.py      -> validation_scored.parquet
Stage 3: shap_explanations.py     -> validation_with_shap.parquet
Stage 4: psi_drift_detection.py   -> psi_drift_report.parquet
```

Each stage consumes the previous stage's Parquet output as a contract — any
stage can be re-run and re-validated in isolation. This modularity made it
possible to test each stage against synthetic data with known ground truth
before running it on the real dataset.

**Stage 1 — Feature Engineering.** Reads the full-schema TLC source via
DuckDB. Centerpiece feature: a route-level fare baseline — each trip's
fare-per-mile compared against the historical average for its specific
pickup→dropoff zone pair, computed as a **ratio-of-sums, not a
mean-of-ratios**, for numerical stability. Near-zero-distance trips and
trips touching reserved TLC zone codes (264/265) are excluded from this
baseline and captured separately rather than discarded.

**Stage 2 — Isolation Forest + Baseline.** Isolation Forest
(`contamination=0.01`) on a strict **time-based split** (train: first ~26
days, validate: last ~5 days — never random, to prevent leakage). A z-score
baseline runs alongside to measure, not assume, the more complex model's
added value.

**Stage 3 — SHAP Explanations.** TreeExplainer explains every flagged trip's
anomaly score in terms of individual feature contributions. Sign convention
unified across the project (positive = more anomalous) after confirming via
direct testing that raw SHAP values for Isolation Forest follow the
opposite convention.

**Stage 4 — PSI Drift Detection.** Compares the model's training
distribution against new data per feature. Demonstrated via a simulated
escalating fare increase (20%/35%/50%), showing PSI climb through
stable→moderate→significant thresholds while non-fare features stay flat.

**Dashboard (`dashboard.py`).** A 7-tab Streamlit app (About, Overview, Data
Quality, Anomalies, Why Flagged, Drift Monitoring, Ask the Data) designed so
a non-technical viewer can understand the system without anyone explaining
it. "Ask the Data" uses a curated library of pre-tested SQL intents
(`intents.py`) matched via a local LLM (`llm_router.py`, Qwen2.5-Coder via
Ollama) — the LLM only *selects* from a safe menu, it never writes SQL
itself. Designed for graceful degradation: a 2-second health check, a
12-second timeout, and automatic fallback to the dropdown menu if the LLM is
slow or unavailable.

## Hurdles found and fixed

1. **Schema mismatch** — a companion streaming pipeline silently dropped 12
   of 18 source columns (`check_schema.py`, `check_columns.py`). Fixed by
   reading the full-schema source file directly.
2. **Numerical instability in route baselines** — mean-of-ratios let two
   extreme short trips skew a route's average to 50.66 against a true
   median of 4.84. Fixed by switching to ratio-of-sums.
3. **Degenerate trips** (`check_casestudy_trips.py`) — 65,369 trips (2.2%)
   had near-zero distance with real fares up to $5,000, mostly GPS-failure
   artifacts. Confirmed as a genuine source-data pattern; excluded from
   per-mile features, preserved separately rather than dropped.
4. **The 45x model bias** (`check_baseline_bias.py`) — unreliable-baseline
   rows were 45x over-represented among flagged anomalies. Root cause: a
   rare binary "low-volume route" flag gave Isolation Forest's
   tree-splitting a cheap, irrelevant way to isolate rows. Fixed by removing
   that flag from the model's features entirely (45x → 0x on synthetic
   data).
5. **The zone-265 bias** (`check_zone265_bias.py`) — after fix #4, a second
   bias remained: trips to reserved TLC zone codes (264 "Unknown", 265
   "Outside of NYC") were 40.73x over-represented, since those codes aren't
   coherent geographic routes. Fixed by excluding them from route baselines
   at the feature-engineering layer; confirmed 0.00x after the fix.
6. **The residual 9x bias** (`check_residual_bias.py`) — 9.10x remained
   after both fixes. Verified independently via the z-score baseline (which
   never saw the reliability flag), which showed its own 2.37x elevation —
   proving ~25-30% of the residual is genuine signal (low-volume routes
   really are noisier), the rest a documented, quantified, unresolved
   model-specific effect rather than something papered over.
7. **SHAP sign-convention trap** — raw SHAP values for Isolation Forest
   follow the opposite convention from the rest of the project. Caught by
   testing against a deliberately injected anomaly; fixed by negating
   values for one consistent convention throughout.
8. **The day-of-week false drift alarm** (`check_dayofweek_drift.py`) — PSI
   flagged `day_of_week` at 2.52 (a false "retrain now" signal). Traced to
   the 5-day validation window containing zero Thursdays — a
   calendar-windowing artifact, not real drift. Fixed by excluding calendar
   features from drift monitoring while keeping them as valid model inputs.

## Key findings

- Isolation Forest and a simple z-score baseline agreed on only **~29-33%**
  of flagged anomalies — meaning ~70% of what the multivariate model catches,
  a simple per-feature threshold would miss.
- Across all flagged rows, `trip_distance` (mean |SHAP| 1.43) and
  `duration_minutes` (0.96) actually outrank `fare_per_mile` (0.88) in
  importance.
- Top anomaly across every run: a 2.73-mile trip that took 4h 41m, charging
  $201 (~7x the route's normal per-mile rate) — on a 13,880-trip,
  well-established route.
- Simulated drift event: `fare_per_mile`'s PSI climbed
  **0.0029 → 0.3923 → 0.8710 → 1.4648** across escalating 20/35/50% fare
  shifts, crossing every threshold proportionally, while non-fare features
  stayed completely flat.

## Tech stack

Python, DuckDB (SQL over Parquet), Google Cloud Storage, scikit-learn
(Isolation Forest), SHAP (TreeExplainer), custom PSI implementation,
Streamlit, Altair, Ollama (Qwen2.5-Coder 1.5B, local, zero API cost),
Docker.

## Running it

Large intermediate Parquet outputs (`silver_taxi_features`,
`validation_scored`, `validation_with_shap`, `degenerate_trips`) are
git-ignored — re-run the pipeline stages in order to regenerate them.
`psi_drift_report.parquet` is small enough to keep as-is and is included
as a real artifact.

```bash
pip install -r requirements.txt
python feature_engineering.py
python isolation_forest.py
python shap_explanations.py
python psi_drift_detection.py
streamlit run dashboard.py
```

Requires a GCS service account key (`gcp-key.json`, not included — see
`.gitignore`) and a local Ollama install with `qwen2.5-coder:1.5b` pulled
for the "Ask the Data" tab.