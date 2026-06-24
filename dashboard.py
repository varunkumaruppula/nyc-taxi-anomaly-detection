"""
Taxi Anomaly Detection -- self-explaining results dashboard.

Designed so a first-time, non-technical viewer (e.g. an executive) can
understand what they're looking at without anyone explaining it:
  - an "About this project" tab orients them first,
  - every chart has a short plain-language caption always visible,
  - every section has an expandable "Tell me more" for the deeper detail.

Reads ONLY pre-computed pipeline outputs, so it loads fast and never runs
heavy work live. Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import duckdb
from pathlib import Path
from intents import INTENTS, list_intents
from llm_router import ollama_available, route_question

st.set_page_config(page_title="Taxi Anomaly Engine", page_icon="\U0001F695", layout="wide")

ACCENT = "#F2B707"
ACCENT_DARK = "#C99405"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
BG_CARD = "#FAFAF8"
GREY = "#BFBFBF"
RED = "#C0392B"

st.markdown(f"""
<style>
  .stApp {{ background: #FFFFFF; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  .block-container {{ padding-top: 2rem; max-width: 1200px; }}
  .metric-card {{
    background: {BG_CARD}; border: 1px solid #ECECEC;
    border-left: 3px solid {ACCENT}; border-radius: 4px; padding: 16px 18px;
  }}
  .metric-value {{ font-size: 26px; font-weight: 700; color: {INK}; }}
  .metric-label {{ font-size: 12px; color: {MUTED}; text-transform: uppercase;
                   letter-spacing: 0.05em; }}
  .plain {{ background: #FFFDF5; border: 1px solid #F0E4B8; border-radius: 4px;
            padding: 12px 16px; margin: 6px 0 14px 0; font-size: 15px;
            color: #4A4A4A; }}
  .plain b {{ color: {INK}; }}
  .finding {{ background: {BG_CARD}; border: 1px solid #ECECEC;
              border-left: 3px solid {ACCENT}; border-radius: 4px;
              padding: 14px 16px; margin-bottom: 10px; }}
  .section-intro {{ color: {MUTED}; font-size: 15px; margin-bottom: 8px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_parquet(name):
    p = Path(name)
    return pd.read_parquet(p) if p.exists() else None


def metric_card(col, value, label):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def plain(text):
    st.markdown(f'<div class="plain">{text}</div>', unsafe_allow_html=True)


def missing(filename, stage):
    st.info(f"`{filename}` not found. Run **{stage}** first. This dashboard "
            f"reads pre-computed results; it doesn't run the pipeline itself.")


features = load_parquet("silver_taxi_features.parquet")
degenerate = load_parquet("degenerate_trips.parquet")
scored = load_parquet("validation_scored.parquet")
shap_data = load_parquet("validation_with_shap.parquet")
psi = load_parquet("psi_drift_report.parquet")

st.title("NYC Taxi Anomaly Detection")
st.markdown(f"<span style='color:{MUTED}'>A system that automatically finds "
            f"unusual taxi trips and explains why each one is unusual \u2014 built "
            f"on roughly 2.8 million real trips.</span>", unsafe_allow_html=True)

tabs = st.tabs(["About this project", "Overview", "Data Quality",
                "Anomalies", "Why Flagged", "Drift Monitoring", "Ask the Data"])


# ---- TAB 0: ABOUT ----
with tabs[0]:
    st.subheader("What this is, in plain terms")
    plain("Every day, taxis make millions of trips. <b>Almost all are normal. "
          "A few are not</b> \u2014 a fare far higher than the trip should cost, a "
          "trip that barely moved but still charged a lot, a journey that took "
          "implausibly long. This project automatically finds those unusual "
          "trips, explains <b>why</b> each was flagged, and watches for the day "
          "the patterns change enough that the system needs updating.")

    st.markdown("#### What you'll find in each tab")
    intro_items = [
        ("Overview", "The big picture: how many trips were analyzed, how many "
                     "were flagged as unusual, and what normal pricing looks like."),
        ("Data Quality", "Problems found in the raw data before any analysis \u2014 "
                         "and how each was handled. Trustworthy results start with "
                         "trustworthy data."),
        ("Anomalies", "The unusual trips themselves: how many, how often, and "
                      "the most extreme examples."),
        ("Why Flagged", "Pick any flagged trip and see, in plain bars, exactly "
                        "which characteristics made it look unusual. No black box."),
        ("Drift Monitoring", "How we'd know the system has gone out of date \u2014 "
                            "demonstrated by simulating a fare change and showing "
                            "the alarm respond."),
    ]
    for name, desc in intro_items:
        st.markdown(f"<div class='finding'><b>{name}</b><br>"
                    f"<span style='color:{MUTED}'>{desc}</span></div>",
                    unsafe_allow_html=True)

    with st.expander("Tell me more \u2014 how the system works under the hood"):
        st.markdown("""
The pipeline runs in four stages, each building on the last:

1. **Feature engineering** \u2014 turn raw trip records into meaningful measures:
   speed, fare-per-mile, how a trip's fare compares to what's normal for its
   specific route.
2. **Anomaly detection** \u2014 an algorithm (Isolation Forest) learns what normal
   trips look like, then scores how unusual each new trip is. A simple
   statistical method runs alongside it as a sanity-check baseline.
3. **Explanation** \u2014 for every flagged trip, SHAP analysis shows which exact
   characteristics drove the decision, so no flag is a mystery.
4. **Drift monitoring** \u2014 PSI tracking watches whether new trips still look
   like the data the system learned from; if the world changes enough, it
   signals that the model should be retrained.

There are no "fraud labels" in this data, so the system is *unsupervised* \u2014
it finds what's unusual rather than being told in advance what fraud looks
like. Everything here is measured and verified, including its limitations.
        """)


# ---- TAB 1: OVERVIEW ----
with tabs[1]:
    st.subheader("The big picture")
    if features is None:
        missing("silver_taxi_features.parquet", "feature engineering (stage 1)")
    else:
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, f"{len(features):,}", "Trips analyzed")
        n_excl = len(degenerate) if degenerate is not None else 0
        metric_card(c2, f"{n_excl:,}", "Set aside (quality)")
        if scored is not None:
            metric_card(c3, f"{int(scored['iso_forest_flag'].sum()):,}", "Flagged unusual")
        else:
            metric_card(c3, "\u2014", "Flagged unusual")
        n_routes = features[["PULocationID", "DOLocationID"]].drop_duplicates().shape[0]
        metric_card(c4, f"{n_routes:,}", "Distinct routes")
        plain("These are the headline numbers. Out of millions of trips, only a "
              "small fraction are flagged as unusual \u2014 that's expected and "
              "healthy. If a large share were flagged, the system would be "
              "crying wolf.")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**What normal pricing looks like**")
            sample = features.sample(min(20000, len(features)), random_state=1)
            fpm = sample[sample["fare_per_mile"] < sample["fare_per_mile"].quantile(0.99)]
            chart = (alt.Chart(fpm).mark_bar(color=ACCENT, opacity=0.85)
                     .encode(alt.X("fare_per_mile:Q", bin=alt.Bin(maxbins=40),
                                   title="Fare per mile ($)"),
                             alt.Y("count():Q", title="Number of trips"))
                     .properties(height=240))
            st.altair_chart(chart, width='stretch')
            plain("Most trips cluster around a typical fare-per-mile. The unusual "
                  "trips live far out in the long tail \u2014 the rare high values "
                  "this chart trims off so the normal range stays readable.")
        with col_b:
            st.markdown("**When trips happen**")
            if "hour_of_day" in features.columns:
                by_hour = features.groupby("hour_of_day").size().reset_index(name="trips")
                hour_chart = (alt.Chart(by_hour).mark_area(color=ACCENT, opacity=0.7, line=True)
                              .encode(alt.X("hour_of_day:Q", title="Hour of day"),
                                      alt.Y("trips:Q", title="Number of trips"))
                              .properties(height=240))
                st.altair_chart(hour_chart, width='stretch')
                plain("Trip volume across the day. This is context, not anomaly "
                      "detection \u2014 it shows the rhythm of normal demand the "
                      "system learns from.")

        with st.expander("Tell me more \u2014 what 'analyzed' vs 'set aside' means"):
            st.markdown(
                "Some trips are **set aside before analysis** because their data "
                "is unreliable \u2014 for example, a trip recording zero distance but "
                "a real fare (a GPS failure, not a real journey). Removing these "
                "first keeps the anomaly detection focused on genuine trips. "
                "They aren't deleted \u2014 they're kept separately and shown in the "
                "Data Quality tab, because they're informative in their own right.")


# ---- TAB 2: DATA QUALITY ----
with tabs[2]:
    st.subheader("What we found in the raw data")
    st.markdown("<div class='section-intro'>Good analysis starts by being honest "
                "about the data's flaws. Each issue below was confirmed against the "
                "real source, not assumed.</div>", unsafe_allow_html=True)
    if degenerate is None:
        missing("degenerate_trips.parquet", "feature engineering (stage 1)")
    else:
        if "reason_excluded" in degenerate.columns:
            counts = degenerate["reason_excluded"].value_counts()
            c1, c2 = st.columns(2)
            metric_card(c1, f"{int(counts.get('near_zero_distance',0)):,}",
                        "Zero-distance, real fare")
            metric_card(c2, f"{int(counts.get('reserved_zone_code',0)):,}",
                        "Undefined-location trips")
            plain("Two kinds of unreliable trips were set aside. Both are real "
                  "patterns in the official data, not mistakes we introduced.")
            st.markdown("---")
            st.markdown("**How the set-aside trips break down**")
            reason_df = counts.reset_index()
            reason_df.columns = ["reason", "count"]
            pie = (alt.Chart(reason_df).mark_arc(innerRadius=50)
                   .encode(theta="count:Q",
                           color=alt.Color("reason:N",
                               scale=alt.Scale(range=[ACCENT, GREY]), title="Reason"))
                   .properties(height=240))
            st.altair_chart(pie, width='stretch')

        st.markdown("---")
        st.markdown("**The three issues, in plain terms**")
        st.markdown('<div class="finding"><b>Trips that charged a fare but barely '
                    'moved.</b><br>About 1 in 45 trips recorded almost no distance '
                    'yet charged real money \u2014 sometimes thousands of dollars. These '
                    'are almost certainly GPS failures. We set them aside so they '
                    'don\'t distort what "normal" looks like.</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding"><b>Trips to undefined locations.</b><br>'
                    'Some trips list their destination as "Unknown" or "Outside of '
                    'NYC" \u2014 catch-all codes, not real places. A "typical fare" for '
                    'these is meaningless, so they were excluded from route '
                    'comparisons.</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding"><b>Missing data in the original feed.</b><br>'
                    'The live data feed was quietly dropping most of its columns, '
                    'including the location information the analysis depends on. We '
                    'caught this and switched to the complete source data.</div>',
                    unsafe_allow_html=True)

        with st.expander("Tell me more \u2014 see the actual set-aside trips"):
            show = [c for c in ["fare_amount", "trip_distance", "PULocationID",
                                "DOLocationID", "reason_excluded"] if c in degenerate.columns]
            st.dataframe(degenerate.sort_values("fare_amount", ascending=False)[show].head(15),
                         width='stretch', hide_index=True)
            st.caption("The highest-fare set-aside trips. Note the fares charged "
                       "against zero or near-zero recorded distance.")


# ---- TAB 3: ANOMALIES ----
with tabs[3]:
    st.subheader("The unusual trips")
    if scored is None:
        missing("validation_scored.parquet", "Isolation Forest (stage 2)")
    else:
        flagged = scored[scored["iso_forest_flag"]].copy()
        c1, c2, c3 = st.columns(3)
        metric_card(c1, f"{len(flagged):,}", "Flagged as unusual")
        metric_card(c2, f"{100*len(flagged)/len(scored):.2f}%", "Of trips checked")
        if "baseline_flag" in scored.columns:
            both = int((scored["iso_forest_flag"] & scored["baseline_flag"]).sum())
            metric_card(c3, f"{100*both/max(len(flagged),1):.0f}%", "Also caught by simple check")
        plain("The system flags only a small percentage as unusual. Notice how few "
              "of these a <b>simple</b> check would also catch \u2014 most require the "
              "smarter method, which is exactly why it's worth using one.")

        st.markdown("---")
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("**How unusual is each flagged trip?**")
            sc = (alt.Chart(flagged).mark_bar(color=ACCENT, opacity=0.85)
                  .encode(alt.X("iso_forest_score:Q", bin=alt.Bin(maxbins=30),
                                title="Unusualness score (higher = more unusual)"),
                          alt.Y("count():Q", title="Flagged trips"))
                  .properties(height=240))
            st.altair_chart(sc, width='stretch')
            plain("Most flagged trips are mildly unusual; a few are extreme. The "
                  "extreme ones (far right) are the ones most worth a human's "
                  "attention first.")
        with col_b:
            st.markdown("**Smart method vs. simple check**")
            if "baseline_flag" in scored.columns:
                both = int((scored["iso_forest_flag"] & scored["baseline_flag"]).sum())
                only_smart = int((scored["iso_forest_flag"] & ~scored["baseline_flag"]).sum())
                venn_df = pd.DataFrame({"group": ["Both methods", "Only the smart method"],
                                        "count": [both, only_smart]})
                vc = (alt.Chart(venn_df).mark_bar()
                      .encode(x=alt.X("count:Q", title="Trips"),
                              y=alt.Y("group:N", title=None, sort="-x"),
                              color=alt.Color("group:N",
                                  scale=alt.Scale(range=[ACCENT, ACCENT_DARK]), legend=None))
                      .properties(height=240))
                st.altair_chart(vc, width='stretch')
                plain("The larger bar is trips only the smart method caught \u2014 the "
                      "value it adds over a basic rule.")

        st.markdown("---")
        st.markdown("**Most unusual trips**")
        cols = [c for c in ["PULocationID", "DOLocationID", "trip_distance",
                            "duration_minutes", "speed_mph", "fare_per_mile",
                            "iso_forest_score", "has_reliable_route_baseline"]
                if c in flagged.columns]
        st.dataframe(flagged.sort_values("iso_forest_score", ascending=False)[cols].head(15),
                     width='stretch', hide_index=True)
        with st.expander("Tell me more \u2014 how to read this table"):
            st.markdown(
                "Each row is one flagged trip. **speed_mph** near zero with a high "
                "**fare_per_mile** means a trip that charged a lot while barely "
                "moving. **has_reliable_route_baseline = True** means we had enough "
                "history for that route to trust the comparison \u2014 those are the "
                "most dependable flags.")


# ---- TAB 4: WHY FLAGGED ----
with tabs[4]:
    st.subheader("Why was a trip flagged?")
    plain("This is the opposite of a black box. Pick any flagged trip and see, in "
          "plain bars, exactly which of its characteristics made it look unusual.")
    if shap_data is None:
        missing("validation_with_shap.parquet", "SHAP explanations (stage 3)")
    else:
        shap_cols = [c for c in shap_data.columns if c.startswith("shap_")]
        flagged = shap_data[shap_data["iso_forest_flag"]].copy()
        reliable = (flagged[flagged["has_reliable_route_baseline"]]
                    if "has_reliable_route_baseline" in flagged.columns else flagged)
        top = reliable.sort_values("iso_forest_score", ascending=False).head(20).reset_index(drop=True)

        labels = [f"#{i+1}  Route {int(r['PULocationID'])}\u2192{int(r['DOLocationID'])}"
                  f"  (score {r['iso_forest_score']:.3f})" for i, r in top.iterrows()]
        choice = st.selectbox("Choose a flagged trip", range(len(labels)),
                              format_func=lambda i: labels[i])
        row = top.iloc[choice]

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, f"{row['trip_distance']:.2f} mi", "Distance")
        metric_card(c2, f"{row['duration_minutes']:.0f} min", "Duration")
        metric_card(c3, f"{row['speed_mph']:.1f} mph", "Avg speed")
        metric_card(c4, f"${row['fare_per_mile']:.2f}", "Fare per mile")

        st.markdown("---")
        contrib = pd.DataFrame({
            "feature": [c.replace("shap_", "").replace("_", " ") for c in shap_cols],
            "contribution": [row[c] for c in shap_cols],
        }).sort_values("contribution", key=abs, ascending=False)
        st.markdown("**What made this trip look unusual**")
        bar = (alt.Chart(contrib).mark_bar()
               .encode(x=alt.X("contribution:Q", title="Pushed toward unusual  \u2192"),
                       y=alt.Y("feature:N", sort="-x", title=None),
                       color=alt.condition(alt.datum.contribution > 0,
                           alt.value(ACCENT), alt.value(GREY)))
               .properties(height=280))
        st.altair_chart(bar, width='stretch')
        plain("Yellow bars pushed this trip toward being flagged; grey bars pushed "
              "against. The longest yellow bar is the main reason this trip stood out.")

        with st.expander("Tell me more \u2014 what these characteristics mean"):
            st.markdown(
                "- **fare per mile** \u2014 the cost per mile traveled\n"
                "- **fare per mile deviation** \u2014 how far that differs from normal "
                "for this exact route\n"
                "- **speed mph** \u2014 average speed; near-zero with a high fare is a "
                "classic red flag\n"
                "- **duration minutes / trip distance** \u2014 unusual combinations "
                "(long time, short distance) stand out\n"
                "- **fare total gap** \u2014 extra charges (tolls, surcharges, tips) on "
                "top of the base fare")


# ---- TAB 5: DRIFT MONITORING ----
with tabs[5]:
    st.subheader("Is the system still up to date?")
    plain("A model trained on today's patterns slowly goes stale as the world "
          "changes. This tab shows how we'd <b>know</b> \u2014 by measuring how far new "
          "trips drift from what the system learned. We simulate a fare increase to "
          "prove the alarm actually responds.")
    if psi is None:
        missing("psi_drift_report.parquet", "PSI drift detection (stage 4)")
    else:
        shift_cols = [c for c in psi.columns if c.startswith("psi_shift_")]
        if "psi_real" in psi.columns and shift_cols:
            rows = []
            for _, r in psi.iterrows():
                rows.append({"feature": r["feature"], "order": 0, "psi": r["psi_real"]})
                for sc in shift_cols:
                    pct = int(sc.replace("psi_shift_", "").replace("pct", ""))
                    rows.append({"feature": r["feature"], "order": pct, "psi": r[sc]})
            long_df = pd.DataFrame(rows)
            line = (alt.Chart(long_df).mark_line(point=True)
                    .encode(x=alt.X("order:Q", title="Simulated fare increase (%)"),
                            y=alt.Y("psi:Q", title="Drift score"),
                            color=alt.Color("feature:N", title="Feature"))
                    .properties(height=320))
            thresh = (alt.Chart(pd.DataFrame({"y": [0.2]}))
                      .mark_rule(strokeDash=[5, 5], color=RED).encode(y="y:Q"))
            st.altair_chart(line + thresh, width='stretch')
            plain("The dashed red line is the 'time to retrain' threshold. As the "
                  "simulated fare increase grows, the fare-related lines climb past "
                  "it \u2014 the alarm working. The flat lines are unaffected measures, "
                  "showing the system pinpoints <b>what</b> changed, not just that "
                  "something did.")

        with st.expander("Tell me more \u2014 why some lines stay flat"):
            st.markdown(
                "We simulated a **fare increase**, so only fare-related measures "
                "should move \u2014 and they do, climbing past the retrain threshold as "
                "the increase grows. Measures like trip distance and speed stay "
                "flat, because a fare change doesn't affect them. That selectivity "
                "is the point: the monitor tells you *what* drifted, which tells you "
                "*why*, not just that something is off.")
            st.dataframe(psi, width='stretch', hide_index=True)

# ---- TAB 6: ASK THE DATA (canonical-intent query explorer) ----
with tabs[6]:
    st.subheader("Ask the data a question")
    plain("Type a question in plain English, or pick one from the menu. Either "
          "way the system runs a pre-built, verified query behind the scenes \u2014 "
          "no code needed. Each answer comes with the result, a chart where it "
          "helps, and the exact query on demand.")

    grouped = list_intents()

    # The local AI assistant is OPTIONAL. Check once whether it's reachable;
    # if not (or if the machine is too busy), the plain-English box still
    # appears but routing falls back to the menu selection, which always works.
    assistant_up = ollama_available()
    if assistant_up:
        st.caption("\U0001F7E2 Local AI assistant connected \u2014 plain-English "
                   "questions enabled.")
    else:
        st.caption("\u26AA Local AI assistant not detected (start Ollama to enable "
                   "plain-English questions). The menu below works regardless.")

    typed = st.text_input(
        "Ask in plain English",
        placeholder="e.g. which trips look suspicious? / busiest hours? / highest fares?",
        disabled=not assistant_up,
    )

    routed_intent = None
    if typed and assistant_up:
        with st.spinner("Understanding your question..."):
            routed_intent = route_question(typed, INTENTS)
        if routed_intent:
            st.success(f"Matched to: **{INTENTS[routed_intent]['description']}**")
        else:
            st.warning("Couldn't confidently match that to a known question \u2014 "
                       "try rephrasing, or pick from the menu below.")

    st.markdown("**Or pick from the menu:**")
    cat_col, q_col = st.columns([1, 2])
    with cat_col:
        category = st.selectbox("Category", sorted(grouped.keys()))
    with q_col:
        menu_intent = st.selectbox(
            "Question",
            grouped[category],
            format_func=lambda n: INTENTS[n]["description"],
        )

    # A successful plain-English match takes priority; otherwise use the menu.
    intent_name = routed_intent if routed_intent else menu_intent

    spec = INTENTS[intent_name]
    source = spec["source"]

    if not Path(source).exists():
        missing(source, "the pipeline stage that produces it")
    else:
        # Run the intent's tested SQL via DuckDB against the source parquet.
        sql = spec["sql"].replace("FROM data", f"FROM read_parquet('{source}')")
        try:
            result = duckdb.connect().execute(sql).df()
        except Exception as e:
            st.error(f"Query failed: {e}")
            result = None

        if result is not None and len(result) > 0:
            chart_type = spec.get("chart", "table")

            if chart_type == "metrics":
                cols = st.columns(len(result.columns))
                for i, c in enumerate(result.columns):
                    val = result.iloc[0][c]
                    val_str = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
                    metric_card(cols[i], val_str, c.replace("_", " ").title())

            elif chart_type in ("bar", "line"):
                x, y = spec["x"], spec["y"]
                mark = alt.Chart(result).mark_bar(color=ACCENT) if chart_type == "bar" \
                    else alt.Chart(result).mark_line(point=True, color=ACCENT)
                ch = mark.encode(
                    x=alt.X(f"{x}:O" if chart_type == "bar" else f"{x}:Q",
                            title=x.replace("_", " ")),
                    y=alt.Y(f"{y}:Q", title=y.replace("_", " ")),
                ).properties(height=320)
                st.altair_chart(ch, width='stretch')
                with st.expander("See the underlying numbers"):
                    st.dataframe(result, width='stretch', hide_index=True)

            else:  # table
                st.dataframe(result, width='stretch', hide_index=True)

            plain(f"<b>Answering:</b> {spec['description']}")

            with st.expander("Show the query (for the technically curious)"):
                st.code(spec["sql"].strip(), language="sql")
                st.caption(f"Runs against: {source}")
        elif result is not None:
            st.info("That query returned no rows for the current data.")