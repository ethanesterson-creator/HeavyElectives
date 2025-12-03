import numpy as np
import pandas as pd
import streamlit as st
from math import ceil
from ortools.sat.python import cp_model

# -------------------------------------------------------------------
# Page Setup + Style
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bauercrest Electives Cycles Matcher (Heavy)",
    page_icon="🏕️",
    layout="wide",
)

PRIMARY_COLOR = "#150b4f"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: white;
    }}
    h1, h2, h3, h4, h5 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
ELECTIVES_PER_DAY = 2        # fixed: 2 electives per day
NUM_CYCLES = 3               # Thu–Wed, repeated 3 times
TOTAL_ELECTIVES_PER_CAMPER = ELECTIVES_PER_DAY * NUM_CYCLES  # 6

# -------------------------------------------------------------------
# Header (logo + title)
# -------------------------------------------------------------------
cols = st.columns([1, 4])
with cols[0]:
    try:
        st.image("logo-header-2.png", use_container_width=True)
    except Exception:
        st.write("")

with cols[1]:
    st.title("Bauercrest Electives Cycles Matcher — HEAVY")
    st.subheader("CP-SAT solver: guaranteed 6 electives per camper if mathematically possible")

st.markdown("---")

# -------------------------------------------------------------------
# Sidebar Settings
# -------------------------------------------------------------------
st.sidebar.header("Settings")
seed = st.sidebar.number_input(
    "Random seed (for tie-breaking in solver search)",
    min_value=0,
    max_value=10000,
    value=42,
    step=1,
)
time_limit = st.sidebar.number_input(
    "Solver time limit (seconds)",
    min_value=5.0,
    max_value=120.0,
    value=30.0,
    step=5.0,
)

# -------------------------------------------------------------------
# Template DataFrames
# -------------------------------------------------------------------
def template_campers_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "first_name": ["Jake", "Noah"],
        "last_name": ["Rosen", "Levy"],
        "bunk": ["Bunk 1", "Bunk 3"],
        "age_group": ["Freshman", "Junior"],
    })

def template_electives_df():
    return pd.DataFrame({
        "elective_id": ["WBK", "BBSK", "FWTB"],
        "elective_name": ["Wakeboarding", "Basketball Skills", "Fun with the Boys"],
        "cycle_capacity": [40, 40, 60],
    })

def template_ballots_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "rank_1": ["WBK", "WSK"],
        "rank_2": ["BBSK", "TUB"],
        "rank_3": ["FWTB", "SAIL"],
        "rank_4": ["SOCC", "BBSK"],
        "rank_5": ["DM", "NJA"],
        "rank_6": ["PHOT", "DM"],
        "rank_7": ["YRBK", "IMPR"],
        "rank_8": ["ULT", "GLF"],
        "rank_9": ["WOOD", "CERM"],
        "rank_10": ["CMPC", "FWTB"],
    })

# -------------------------------------------------------------------
# Helper: compute satisfaction stats
# -------------------------------------------------------------------
def compute_preference_stats(camper_ids, assignments_by_cycle, ballot_map):
    records = []
    for cid in camper_ids:
        # flatten electives across cycles
        electives = []
        for cycle in range(1, NUM_CYCLES + 1):
            electives.extend(assignments_by_cycle[cid].get(cycle, []))
        prefs = [str(p).strip() for p in ballot_map[cid] if not pd.isna(p) and str(p).strip() != ""]
        ranks = []
        for e in electives:
            if e in prefs:
                ranks.append(prefs.index(e) + 1)
        best_rank = min(ranks) if ranks else None
        records.append({
            "camper_id": cid,
            "assigned_count": len(electives),
            "best_rank": best_rank,
            "ranks": ranks,
        })

    n = len(records)
    if n == 0:
        return {}, pd.DataFrame()

    got_rank1 = sum(1 for r in records if r["best_rank"] == 1)
    got_top3 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 3)
    got_top5 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 5)

    best_ranks_nonnull = [r["best_rank"] for r in records if r["best_rank"] is not None]
    avg_best = sum(best_ranks_nonnull) / len(best_ranks_nonnull) if best_ranks_nonnull else None
    avg_assigned = sum(r["assigned_count"] for r in records) / n

    stats = {
        "n_campers": n,
        "pct_got_rank1": 100.0 * got_rank1 / n,
        "pct_got_top3": 100.0 * got_top3 / n,
        "pct_got_top5": 100.0 * got_top5 / n,
        "avg_best_rank": avg_best,
        "avg_assigned": avg_assigned,
    }

    return stats

# -------------------------------------------------------------------
# File Uploads
# -------------------------------------------------------------------
st.header("1️⃣ Upload Data Files")

campers_file = st.file_uploader("Upload campers.csv", type=["csv"])
ballots_file = st.file_uploader("Upload ballots.csv (10 ranked choices)", type=["csv"])
electives_file = st.file_uploader("Upload electives.csv (cycle capacities)", type=["csv"])

have_all = campers_file is not None and ballots_file is not None and electives_file is not None

if not have_all:
    st.info(
        "Upload campers.csv, ballots.csv, and electives.csv to run the heavy solver. "
        "Or download templates below and adapt them to match the CampMinder export."
    )

    st.subheader("campers.csv template")
    tmpl_camp = template_campers_df()
    st.dataframe(tmpl_camp)
    st.download_button(
        "Download campers.csv template",
        tmpl_camp.to_csv(index=False).encode("utf-8"),
        file_name="campers_template.csv",
        mime="text/csv",
    )

    st.subheader("electives.csv template")
    tmpl_elec = template_electives_df()
    st.dataframe(tmpl_elec)
    st.download_button(
        "Download electives.csv template",
        tmpl_elec.to_csv(index=False).encode("utf-8"),
        file_name="electives_template.csv",
        mime="text/csv",
    )

    st.subheader("ballots.csv template")
    tmpl_ballots = template_ballots_df()
    st.dataframe(tmpl_ballots)
    st.download_button(
        "Download ballots.csv template",
        tmpl_ballots.to_csv(index=False).encode("utf-8"),
        file_name="ballots_template.csv",
        mime="text/csv",
    )

else:
    campers_df = pd.read_csv(campers_file)
    ballots_df = pd.read_csv(ballots_file)
    electives_df = pd.read_csv(electives_file)

    st.success("Files uploaded successfully. Preview below:")
    with st.expander("campers.csv"):
        st.dataframe(campers_df.head())
    with st.expander("ballots.csv"):
        st.dataframe(ballots_df.head())
    with st.expander("electives.csv"):
        st.dataframe(electives_df.head())

    # Validate campers.csv
    required_camper_cols = {"camper_id", "first_name", "last_name", "bunk", "age_group"}
    if not required_camper_cols.issubset(set(campers_df.columns)):
        st.error(f"campers.csv must contain columns: {sorted(required_camper_cols)}")
    else:
        # Rank columns
        rank_cols = [c for c in ballots_df.columns if c.startswith("rank_")]
        if len(rank_cols) < 10:
            st.warning("ballots.csv currently has fewer than 10 rank_* columns. The solver will use whatever exists.")
        rank_cols = sorted(rank_cols, key=lambda x: int(x.split("_")[1]))

        st.header("2️⃣ Solve 3-cycle elective plan (heavy)")

        if st.button("Run heavy solver"):
            # Merge campers + ballots
            merged = pd.merge(ballots_df, campers_df, on="camper_id", how="inner")
            camper_ids = merged["camper_id"].unique().tolist()
            num_campers = len(camper_ids)

            # Index maps
            camper_index = {cid: i for i, cid in enumerate(camper_ids)}

            # Electives
            electives_df = electives_df.copy()
            if "cycle_capacity" not in electives_df.columns:
                if "period_capacity" in electives_df.columns:
                    electives_df = electives_df.rename(columns={"period_capacity": "cycle_capacity"})
                else:
                    st.error("electives.csv must have either 'cycle_capacity' or 'period_capacity' column.")
                    st.stop()

            electives_df["cycle_capacity"] = electives_df["cycle_capacity"].astype(int)
            elective_ids = electives_df["elective_id"].tolist()
            elective_index = {eid: j for j, eid in enumerate(elective_ids)}
            num_electives = len(elective_ids)

            # Ballot map: camper -> list of preferences (elective_ids)
            ballot_map = {}
            for cid in camper_ids:
                row = merged[merged["camper_id"] == cid].iloc[0]
                prefs = []
                for col in rank_cols:
                    v = row[col]
                    if pd.isna(v):
                        continue
                    v = str(v).strip()
                    if v == "":
                        continue
                    prefs.append(v)
                ballot_map[cid] = prefs

            # Quick impossibility sanity check per camper:
            # if number of distinct electives on their ballot < 6, it's impossible.
            impossible_campers = [
                cid for cid in camper_ids if len(set(ballot_map[cid])) < TOTAL_ELECTIVES_PER_CAMPER
            ]
            if impossible_campers:
                st.error(
                    "Some campers listed fewer than 6 distinct electives in their 10 choices. "
                    "It is mathematically impossible to give them 6 different electives from their ballot. "
                    f"Example camper_id: {impossible_campers[0]}"
                )
                st.stop()

            # Build solver model
            model = cp_model.CpModel()

            # Decision variables: y[c, e, cycle] = 1 if camper c takes elective e in cycle
            y = {}
            # Only allow electives that appear on camper's ballot
            for cid in camper_ids:
                i = camper_index[cid]
                prefs_set = set(ballot_map[cid])
                for eid in elective_ids:
                    j = elective_index[eid]
                    if eid not in prefs_set:
                        continue  # never assign an elective not on their ballot
                    for cycle in range(1, NUM_CYCLES + 1):
                        y[(i, j, cycle)] = model.NewBoolVar(f"y_c{i}_e{j}_cy{cycle}")

            # Constraint 1: each camper gets exactly 6 electives total
            for cid in camper_ids:
                i = camper_index[cid]
                vars_for_camper = [
                    var
                    for (ci, j, cy), var in y.items()
                    if ci == i
                ]
                model.Add(sum(vars_for_camper) == TOTAL_ELECTIVES_PER_CAMPER)

            # Constraint 2: each camper can have at most 2 electives per cycle
            for cid in camper_ids:
                i = camper_index[cid]
                for cycle in range(1, NUM_CYCLES + 1):
                    vars_for_camper_cycle = [
                        var
                        for (ci, j, cy), var in y.items()
                        if ci == i and cy == cycle
                    ]
                    if vars_for_camper_cycle:
                        model.Add(sum(vars_for_camper_cycle) <= ELECTIVES_PER_DAY)

            # Constraint 3: no camper repeats the same elective in multiple cycles
            for cid in camper_ids:
                i = camper_index[cid]
                for eid in elective_ids:
                    j = elective_index[eid]
                    vars_same_elective = [
                        var
                        for (ci, ej, cy), var in y.items()
                        if ci == i and ej == j
                    ]
                    if vars_same_elective:
                        model.Add(sum(vars_same_elective) <= 1)

            # Constraint 4: capacity per elective per cycle
            for eid in elective_ids:
                j = elective_index[eid]
                cap = int(electives_df.loc[electives_df["elective_id"] == eid, "cycle_capacity"].iloc[0])
                for cycle in range(1, NUM_CYCLES + 1):
                    vars_for_elective_cycle = [
                        var
                        for (ci, ej, cy), var in y.items()
                        if ej == j and cy == cycle
                    ]
                    if vars_for_elective_cycle:
                        model.Add(sum(vars_for_elective_cycle) <= cap)

            # Objective: maximize preference quality
            # weight = 11 - rank (rank1=10, rank10=1)
            weights = {}  # (i, j) -> weight
            for cid in camper_ids:
                i = camper_index[cid]
                prefs = ballot_map[cid]
                for rank, eid in enumerate(prefs, start=1):
                    if eid not in elective_index:
                        continue
                    j = elective_index[eid]
                    w = max(1, 11 - rank)
                    # If multiple ranks of same elective somehow, keep best weight
                    if (i, j) not in weights or w > weights[(i, j)]:
                        weights[(i, j)] = w

            objective_terms = []
            for (i, j, cycle), var in y.items():
                w = weights.get((i, j), 1)
                objective_terms.append(w * var)

            model.Maximize(sum(objective_terms))

            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = float(time_limit)
            solver.parameters.random_seed = int(seed)

            st.info("Running heavy solver... this may take a little while.")
            status = solver.Solve(model)

            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                st.error(
                    "No feasible solution found. That means it is mathematically impossible to give "
                    "every camper 6 electives from their 10 choices with the current capacities. "
                    "Increase some elective capacities or adjust ballots and try again."
                )
                st.stop()

            st.success("Solver found a feasible 3-cycle plan with 6 electives per camper.")

            # Extract solution: assignments_by_cycle[cid][cycle] = [elective_ids]
            assignments_by_cycle = {
                cid: {cycle: [] for cycle in range(1, NUM_CYCLES + 1)}
                for cid in camper_ids
            }
            for (i, j, cycle), var in y.items():
                if solver.Value(var) == 1:
                    cid = camper_ids[i]
                    eid = elective_ids[j]
                    assignments_by_cycle[cid][cycle].append(eid)

            # Build camper_cycles_df
            rows = []
            for cid in camper_ids:
                row = {"camper_id": cid}
                for cycle in range(1, NUM_CYCLES + 1):
                    electives = assignments_by_cycle[cid][cycle]
                    # ensure stable length 2
                    for slot in range(ELECTIVES_PER_DAY):
                        key = f"cycle{cycle}_elective{slot+1}"
                        row[key] = electives[slot] if slot < len(electives) else ""
                rows.append(row)

            camper_cycles_df = pd.DataFrame(rows)
            camper_cycles_df = camper_cycles_df.merge(
                campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
                on="camper_id",
                how="left",
            )

            # Preference stats
            stats = compute_preference_stats(camper_ids, assignments_by_cycle, ballot_map)

            st.subheader("Preference Satisfaction Summary")
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Campers", stats["n_campers"])
                c2.metric("Got ≥1 #1 choice", f"{stats['pct_got_rank1']:.1f}%")
                c3.metric("Got ≥1 Top 3", f"{stats['pct_got_top3']:.1f}%")
                c4.metric("Got ≥1 Top 5", f"{stats['pct_got_top5']:.1f}%")
                st.caption(
                    f"Average best rank: {stats['avg_best_rank']:.2f} | "
                    f"Average electives assigned: {stats['avg_assigned']:.2f} (should be {TOTAL_ELECTIVES_PER_CAMPER})"
                )
            else:
                st.write("No stats available.")

            # Camper cycle overview
            st.subheader("Camper Cycle Overview")
            st.dataframe(camper_cycles_df)
            st.download_button(
                "Download camper cycle assignments CSV",
                camper_cycles_df.to_csv(index=False).encode("utf-8"),
                file_name="camper_cycle_assignments_heavy.csv",
                mime="text/csv",
            )

            # Cycle rosters
            roster_rows = []
            elec_name_map = dict(zip(electives_df["elective_id"], electives_df["elective_name"]))
            for _, row in camper_cycles_df.iterrows():
                cid = row["camper_id"]
                for cycle in range(1, NUM_CYCLES + 1):
                    for slot in range(ELECTIVES_PER_DAY):
                        col = f"cycle{cycle}_elective{slot+1}"
                        eid = row[col]
                        if isinstance(eid, str) and eid != "":
                            roster_rows.append({
                                "cycle": cycle,
                                "elective_id": eid,
                                "elective_name": elec_name_map.get(eid, eid),
                                "camper_id": cid,
                                "first_name": row["first_name"],
                                "last_name": row["last_name"],
                                "bunk": row["bunk"],
                                "age_group": row["age_group"],
                            })
            cycle_rosters_df = pd.DataFrame(roster_rows)
            if not cycle_rosters_df.empty:
                cycle_rosters_df = cycle_rosters_df.sort_values(
                    ["cycle", "elective_name", "bunk", "last_name"]
                )

            st.subheader("Rosters by Cycle & Elective")
            if cycle_rosters_df.empty:
                st.warning("No cycle rosters generated. Check capacities and ballots.")
            else:
                st.dataframe(cycle_rosters_df)
                st.download_button(
                    "Download cycle rosters CSV",
                    cycle_rosters_df.to_csv(index=False).encode("utf-8"),
                    file_name="cycle_rosters_heavy.csv",
                    mime="text/csv",
                )

            # Demand snapshot
            demand_counts = {}
            for _, row in ballots_df.iterrows():
                for col in rank_cols:
                    v = row[col]
                    if pd.isna(v):
                        continue
                    v = str(v).strip()
                    if v == "":
                        continue
                    demand_counts[v] = demand_counts.get(v, 0) + 1

            electives_df["demand"] = electives_df["elective_id"].apply(lambda e: demand_counts.get(e, 0))

            st.subheader("Elective Demand Snapshot")
            st.dataframe(
                electives_df[["elective_id", "elective_name", "cycle_capacity", "demand"]]
            )
