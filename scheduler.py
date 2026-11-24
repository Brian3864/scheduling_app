# === Streamlit App: Interleaved Desorption Scheduling ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import html as html_lib
import sqlalchemy
from google.cloud.sql.connector import Connector
from google.oauth2 import service_account

# Plant cycle log and stage-duration log now come from the live MySQL database
# (SCADA -> MySQL -> this app) instead of the two static CSVs this app used to
# ship with. The database is Google Cloud SQL, reached via the Cloud SQL Python
# Connector (not a plain host/port) using the instance connection name below --
# that name, the database name, and the username are identifiers, not secrets,
# so they're plain constants here. The two actual secrets -- the DB password
# and the GCP service account key used to authenticate to the Cloud SQL Admin
# API -- are never in this file; they live only in Streamlit secrets (locally:
# .streamlit/secrets.toml, gitignored; deployed: the app's Settings -> Secrets
# panel), under a [gcp_service_account] block (the exact fields from the
# downloaded service-account JSON key) and st.secrets["mysql"]["password"].
CLOUD_SQL_INSTANCE_CONNECTION_NAME = "scada-425409:us-central1:oc-public-db"
CLOUD_SQL_DATABASE = "oc_public"
CLOUD_SQL_USER = "brianstreamlit"


@st.cache_resource
def _mysql_engine():
    """A cached SQLAlchemy engine that tunnels through the Cloud SQL Python
    Connector using the GCP service account in secrets, re-used across reruns
    instead of re-authenticating to the Cloud SQL Admin API on every query."""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    connector = Connector(credentials=credentials)

    def _getconn():
        return connector.connect(
            CLOUD_SQL_INSTANCE_CONNECTION_NAME,
            "pymysql",
            user=CLOUD_SQL_USER,
            password=st.secrets["mysql"]["password"],
            db=CLOUD_SQL_DATABASE,
        )

    return sqlalchemy.create_engine("mysql+pymysql://", creator=_getconn)


# Table/column names below match the real schema (per studio_results export,
# 2026-11-12). Only the "rename(columns=...)" maps below and the WHERE clause
# on stage_durations are schema-specific -- nothing downstream of
# _plant_cycles_df()/_stage_timestamps_df() changed, since both still return a
# DataFrame with the same columns the CSVs used to.
#
# Critically, the "Module" column's VALUES must also use the same combo-style
# labels the rest of this file keys off of (e.g. "N1N2N3-M1n3", "N3-M2n4") --
# see PAIR_PLANT_MODULE, PAIR_YIELD_FORMULA, TAB3_8_8_MODULE and the
# REPRESENTS_*/PAIR_PHYSICAL_MODULES_* constants throughout this file. A
# mismatch here won't raise an error -- _module_average/_phase_averages_for_module
# just silently return 0.0/{} for a Module value that doesn't match any row.
PLANT_CYCLES_TABLE = "plant_cycles"
PLANT_CYCLES_COLUMNS = {
    # live DB column name: name the rest of this file expects
    "module": "Module",
    "etotal_kwh": "eTotal kWh",
    "bag_co2": "BAG CO2 (kg)",
}
PLANT_CYCLES_NUMERIC_COLUMNS = ["eTotal kWh", "BAG CO2 (kg)"]

# stage_durations has a "has_data" flag (1 = a real reading was logged for
# that cycle, 0 = placeholder/incomplete row) -- only has_data=1 rows are
# pulled, so incomplete cycles don't drag the phase-duration averages down.
STAGE_TIMESTAMPS_TABLE = "stage_durations"
STAGE_TIMESTAMPS_COLUMNS = {
    "module": "Module",
    "adsorption": "Adsorption (min)",
    "evacuation": "Evacuation (min)",
    "ncgs_purging": "NCGs Purging (min)",
    "ncgs_purging_heating": "NCGs Purging/Heating (min)",
    "heating_co2_purging": "Heating/CO2 Purging (min)",
    "co2_purging": "CO2 Purging (min)",
    "cooling": "Cooling (min)",
}
STAGE_TIMESTAMPS_NUMERIC_COLUMNS = [
    "Adsorption (min)", "Evacuation (min)", "NCGs Purging (min)",
    "NCGs Purging/Heating (min)", "Heating/CO2 Purging (min)",
    "CO2 Purging (min)", "Cooling (min)",
]

# Phase-duration sourcing (used by load_stage_duration_defaults_by_pair): Groups A
# and C are built from the same 3-Nelion (N1N2N3) module combination, while Group B
# is built from a 2-Nelion (N1N2) combination.
PAIR_PLANT_MODULE = {
    "Group A": "N1N2N3-M1n3",
    "Group B": "N1N2-M1n3",
    "Group C": "N1N2N3-M1n3",
}

# Yield per Cycle sourcing (used by load_plant_cycle_defaults_by_pair) is a
# deliberately different formula per pair from Energy per Cycle: a weighted sum of
# one or more Module averages. Group A uses N1N2N3-M1n3 alone; Group B uses
# N3-M2n4 doubled (it represents two of that combination); Group C sums
# N1N2-M1n3 + N3-M2n4. Energy per Cycle instead uses PAIR_PLANT_MODULE (the same
# single-Module sourcing as phase durations) — this formula does not apply to it.
PAIR_YIELD_FORMULA = {
    "Group A": [("N1N2N3-M1n3", 1)],
    "Group B": [("N3-M2n4", 2)],
    "Group C": [("N1N2-M1n3", 1), ("N3-M2n4", 1)],
}

# The actual physical desorption grouping for 6-6-4 — which of the 16 real
# Nelion/module-position modules are wired together into each pair. This is a
# full, non-overlapping partition of all 16 modules, distinct from
# PAIR_PLANT_MODULE/PAIR_YIELD_FORMULA above (those are only proxy data
# sources for Duration/Energy/Yield, not a claim about physical composition).
PAIR_PHYSICAL_MODULES = {
    "Group A": ["N1-M1", "N1-M3", "N2-M1", "N2-M3", "N3-M1", "N3-M3"],
    "Group B": ["N1-M2", "N1-M4", "N2-M2", "N2-M4", "N4-M2", "N4-M4"],
    "Group C": ["N3-M2", "N3-M4", "N4-M1", "N4-M3"],
}

# Alternative 8-4-4 pairing (vs. the 6-6-4 Group A/B/C above), for side-by-side
# comparison in Advanced Interleaved. Reuses the "Group A/B/C" pair names (this
# config's own dedicated tables, distinguished by section header, not by name) -
# Group A is the 8-module pair, Groups B and C are the two 4-module pairs. Phase
# durations AND Energy per Cycle both use one Module average per pair: Group A
# from N1N2N3-M1n3, Groups B and C from N1N2-M1n3.
PAIRS_8_4_4 = ["Group A", "Group B", "Group C"]
PAIR_PLANT_MODULE_8_4_4 = {
    "Group A": "N1N2N3-M1n3",
    "Group B": "N1N2-M1n3",
    "Group C": "N1N2-M1n3",
}

# Yield per Cycle for the 8-4-4 config uses its own different weighted formula
# (like PAIR_YIELD_FORMULA does for 6-6-4): Group A sums N1N2N3-M1n3 + N3-M2n4,
# Group B uses N1N2-M1n3 alone, Group C uses N3-M2n4 doubled.
PAIR_YIELD_FORMULA_8_4_4 = {
    "Group A": [("N1N2N3-M1n3", 1), ("N3-M2n4", 1)],
    "Group B": [("N1N2-M1n3", 1)],
    "Group C": [("N3-M2n4", 2)],
}

# The actual physical desorption grouping for 8-4-4 — a full, non-overlapping
# partition of all 16 modules (see PAIR_PHYSICAL_MODULES above for 6-6-4's).
PAIR_PHYSICAL_MODULES_8_4_4 = {
    "Group A": ["N1-M1", "N2-M1", "N3-M1", "N4-M1", "N1-M3", "N2-M3", "N3-M3", "N4-M3"],
    "Group B": ["N1-M2", "N2-M2", "N1-M4", "N2-M4"],
    "Group C": ["N3-M2", "N4-M2", "N3-M4", "N4-M4"],
}

# "4-4-4-4" configuration: 4 equal 4-module pairs, one per Nelion-pair x position
# combination — Group A = N1&N2's M1/M3, Group B = N1&N2's M2/M4, Group C =
# N3&N4's M1/M3, Group D = N3&N4's M2/M4. Only Group A's combination
# (N1N2-M1n3) is actually recorded; the other three have no real data of their
# own, so per instruction every group's phase durations AND Energy per Cycle
# come from N1N2-M1n3's own real average. Yield per Cycle instead splits: Group
# A and Group B keep N1N2-M1n3's own real Yield, while Group C and Group D use
# N3-M2n4's real Yield DOUBLED. Adsorption overlap is unrestricted for all 4
# pairs, since the original 3-pair rule doesn't generalize to a 4th pair
# without guessing at an unknown fan-sharing constraint.
PAIRS_4X4 = ["Group A", "Group B", "Group C", "Group D"]
PAIR_MODULE_4X4 = {
    "Group A": "N1N2-M1n3",
    "Group B": "N1N2-M1n3",
    "Group C": "N1N2-M1n3",
    "Group D": "N1N2-M1n3",
}
PAIR_YIELD_FORMULA_4X4 = {
    "Group A": [("N1N2-M1n3", 1)],
    "Group B": [("N1N2-M1n3", 1)],
    "Group C": [("N3-M2n4", 2)],
    "Group D": [("N3-M2n4", 2)],
}
# Group A's real modules match its source combo exactly; Groups B/C/D each
# represent a different physical 4-module combination than the one they
# borrow real averages from (none of the three has its own recorded data).
REPRESENTS_4X4 = {
    "Group B": "N1-M2, N1-M4, N2-M2, N2-M4 (not directly recorded)",
    "Group C": "N3-M1, N3-M3, N4-M1, N4-M3 (not directly recorded)",
    "Group D": "N3-M2, N3-M4, N4-M2, N4-M4 (not directly recorded)",
}
# The actual physical desorption grouping for 4-4-4-4 — a full, non-overlapping
# partition of all 16 modules (matches REPRESENTS_4X4's descriptions above).
PAIR_PHYSICAL_MODULES_4X4 = {
    "Group A": ["N1-M1", "N1-M3", "N2-M1", "N2-M3"],
    "Group B": ["N1-M2", "N1-M4", "N2-M2", "N2-M4"],
    "Group C": ["N3-M1", "N3-M3", "N4-M1", "N4-M3"],
    "Group D": ["N3-M2", "N3-M4", "N4-M2", "N4-M4"],
}

# "2-2-2-2 (N3/N4)" configuration: a second 4-equal-group config, this time
# covering only Nelion 3 and Nelion 4 (8 of 16 modules) instead of the full
# plant. Grouped by module TYPE rather than by Nelion: Group A = N3's M1&M3,
# Group B = N4's M1&M3, Group C = N3's M2&M4, Group D = N4's M2&M4.
# All four groups now have their own real recorded cycles (Nelion 4 was
# commissioned and both the plant-cycle and stage-duration data were updated
# with N4-M1n3/N4-M2n4 data), so
# each group sources Duration, Energy and Yield directly from its own matching
# Module — no mirroring across Nelions, and since every group is already a
# genuine 2-module pair (unlike 4-4-4-4's 4-module groups), no multiplier is
# needed anywhere. Adsorption overlap is unrestricted for all 4 pairs, same as
# 4-4-4-4 above.
PAIRS_N3N4 = ["Group A", "Group B", "Group C", "Group D"]
PAIR_MODULE_N3N4 = {
    "Group A": "N3-M1n3",
    "Group B": "N4-M1n3",
    "Group C": "N3-M2n4",
    "Group D": "N4-M2n4",
}
PAIR_YIELD_FORMULA_N3N4 = {
    "Group A": [("N3-M1n3", 1)],
    "Group B": [("N4-M1n3", 1)],
    "Group C": [("N3-M2n4", 1)],
    "Group D": [("N4-M2n4", 1)],
}
# Every group's physical modules now match its data source combo exactly — no
# REPRESENTS overrides needed (see module_mapping_table's represents_map arg).
# The actual physical desorption grouping for 2-2-2-2 (N3/N4) — a partition of
# only Nelion 3 and Nelion 4's 8 modules (N1 and N2 aren't part of this
# configuration at all).
PAIR_PHYSICAL_MODULES_N3N4 = {
    "Group A": ["N3-M1", "N3-M3"],
    "Group B": ["N4-M1", "N4-M3"],
    "Group C": ["N3-M2", "N3-M4"],
    "Group D": ["N4-M2", "N4-M4"],
}

# "6-5-5" configuration: 3 pairs (16 = 6+5+5), chosen for the best achievable
# Yield. Group A = N1N2N3-M1n3 (6 modules) — the only real combination that
# matches a 6-module pair exactly, and the highest real Yield per Cycle
# available (11.43 kg CO2). Group B physically = N3-M4 + all of Nelion 4 (5
# modules), Group C physically = N1&N2's M2/M4 + N3-M2 (5 modules) — see
# PAIR_PHYSICAL_MODULES_6_5_5. Neither has its own recorded cycles, so both
# take their phase durations from N1N2-M1n3 — the closest well-sampled real
# combination in size, so their cycle timing stays realistic rather than
# borrowing a 2-module combination's much shorter cycle — but their Yield per
# Cycle instead uses weighted sums of N2-M2n4/N3-M2n4: Group B = N2-M2n4 +
# N3-M2n4 + half of N3-M2n4 (i.e. N3-M2n4 x1.5), Group C = N3-M2n4 x2 + half
# of N3-M2n4 (i.e. N3-M2n4 x2.5). This is a 3-pair
# configuration, so it uses the ORIGINAL Adsorption-overlap rule (Group A can
# overlap with Group B/C; Group B and Group C cannot overlap each other) —
# no relaxation needed, unlike the 4-pair configurations above.
PAIRS_6_5_5 = ["Group A", "Group B", "Group C"]
PAIR_DURATION_MODULE_6_5_5 = {
    "Group A": "N1N2N3-M1n3",
    "Group B": "N1N2-M1n3",
    "Group C": "N1N2-M1n3",
}
PAIR_ENERGY_MODULE_6_5_5 = {
    "Group A": "N1N2N3-M1n3",
    "Group B": "N1N2-M1n3",
    "Group C": "N1N2-M1n3",
}
PAIR_YIELD_FORMULA_6_5_5 = {
    "Group A": [("N1N2N3-M1n3", 1)],
    "Group B": [("N2-M2n4", 1), ("N3-M2n4", 1), ("N3-M2n4", 0.5)],
    "Group C": [("N3-M2n4", 2), ("N3-M2n4", 0.5)],
}
# Group A's real modules match its source combo exactly (both are the same
# 6-module N1N2N3-M1n3 combination). Groups B and C are each "5 modules" only
# because that's the size needed to reach 16 — no real 5-module combination
# exists in the data, so their exact module composition is unspecified.
REPRESENTS_6_5_5 = {
    "Group B": "N3-M4, N4-M1, N4-M2, N4-M3, N4-M4 (no real 5-module combination in the data)",
    "Group C": "N1-M2, N1-M4, N2-M2, N2-M4, N3-M2 (no real 5-module combination in the data)",
}
# The actual physical desorption grouping for 6-5-5 — a full, non-overlapping
# partition of all 16 modules (matches REPRESENTS_6_5_5's descriptions above).
PAIR_PHYSICAL_MODULES_6_5_5 = {
    "Group A": ["N1-M1", "N1-M3", "N2-M1", "N2-M3", "N3-M1", "N3-M3"],
    "Group B": ["N3-M4", "N4-M1", "N4-M2", "N4-M3", "N4-M4"],
    "Group C": ["N1-M2", "N1-M4", "N2-M2", "N2-M4", "N3-M2"],
}

@st.cache_data(ttl=300)
def _plant_cycles_df():
    """Load the plant cycle log from MySQL, or None if the query/connection
    fails. Cached (5 min TTL) since every pair-default lookup across both tabs
    re-reads this same data — without caching, Streamlit would re-query the
    database on every single rerun (any widget interaction anywhere in the
    whole app); the TTL still lets fresh SCADA-sourced rows show up periodically
    instead of only once per server restart."""
    try:
        df = pd.read_sql(f"SELECT * FROM {PLANT_CYCLES_TABLE}", _mysql_engine())
        df = df.rename(columns=PLANT_CYCLES_COLUMNS)
        for col in PLANT_CYCLES_NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None

def _module_average(plant_cycles_df, module, since=None):
    """(avg eTotal kWh, avg BAG CO2 kg) for one Module value's rows, or (0.0, 0.0)
    if there are none (or the log failed to load). Yield is sourced from BAG CO2
    (kg) — the actual captured/bagged CO2 — not DES CO2 (kg), which is a
    desorption-side figure that can diverge from what's actually collected.
    Rows missing a reading (NaN) or logged as exactly 0 are excluded from that
    column's average rather than counted as a real 0-value cycle. If `since`
    is given, only rows with start_time on or after it are averaged -- e.g. to
    reflect a process change made partway through the log."""
    if plant_cycles_df is None:
        return 0.0, 0.0
    subset = plant_cycles_df[plant_cycles_df["Module"] == module]
    if since is not None:
        subset = subset[pd.to_datetime(subset["start_time"]) >= since]
    if len(subset) == 0:
        return 0.0, 0.0
    energy_vals = subset["eTotal kWh"]
    energy_vals = energy_vals[energy_vals.notna() & (energy_vals != 0)]
    co2_vals = subset["BAG CO2 (kg)"]
    co2_vals = co2_vals[co2_vals.notna() & (co2_vals != 0)]
    return (
        round(float(energy_vals.mean()), 2) if len(energy_vals) > 0 else 0.0,
        round(float(co2_vals.mean()), 2) if len(co2_vals) > 0 else 0.0,
    )

def _decode_module_combo(combo):
    """Expand a 'N1N2N3-M1n3' style combo name into its underlying module list,
    e.g. ['N1-M1', 'N1-M3', 'N2-M1', 'N2-M3', 'N3-M1', 'N3-M3']. Combo names are
    always '<Nelions>-M<a>n<b>', where <Nelions> is one or more 'N<digit>' runs."""
    nelions_part, positions_part = combo.split("-")
    nelions = [nelions_part[i:i + 2] for i in range(0, len(nelions_part), 2)]
    positions = [positions_part[0:2], "M" + positions_part[3]]
    return [f"{nelion}-{position}" for nelion in nelions for position in positions]

def _format_module_combo(combo):
    """'N1N2N3-M1n3' -> 'N1N2N3-M1n3 (N1-M1, N1-M3, N2-M1, N2-M3, N3-M1, N3-M3)'."""
    return f"{combo} ({', '.join(_decode_module_combo(combo))})"

def _format_yield_formula(formula):
    """[('N3-M2n4', 3)] -> 'N3-M2n4 × 3'; [('A', 1), ('B', 0.5)] -> 'A + B × 0.5'."""
    parts = []
    for module, weight in formula:
        parts.append(module if weight == 1 else f"{module} × {weight:g}")
    return " + ".join(parts)

def _format_source(value):
    """A Duration/Energy source is usually a single real combo string (rendered
    via _format_module_combo), but can also be a weighted list like Yield's
    formulas, for a pair with no single real combo of its own that instead
    averages/sums two or more other real Modules. Renders either form."""
    if isinstance(value, str):
        return _format_module_combo(value)
    return _format_yield_formula(value)

def module_mapping_table(pairs, duration_module_map, energy_module_map, yield_formula_map, represents_map=None):
    """Per-pair table showing which real Module combination sources that pair's
    phase Duration, Energy per Cycle, and Yield per Cycle — i.e. how modules
    from the CSV data are paired up to represent each configuration's pairs.
    When a pair's actual physical module composition differs from (or isn't
    fully covered by) its data source — e.g. an extrapolated pair, or a pair
    sized differently from any real combination — pass its description in
    represents_map so the table doesn't imply the source IS the pair."""
    rows = []
    for pair in pairs:
        row = {"Pair": pair}
        if represents_map is not None:
            row["Represents"] = represents_map.get(pair, "Same as Duration Source")
        row["Duration Source"] = _format_source(duration_module_map[pair])
        row["Energy Source"] = _format_source(energy_module_map[pair])
        row["Yield Source (weighted)"] = _format_yield_formula(yield_formula_map[pair])
        rows.append(row)
    return pd.DataFrame(rows)

PAIR_CIRCLE_COLORS = ["🟡", "🔴", "🟢", "🔵", "🟣", "🟠"]

def physical_pairing_table(pairs, physical_modules_map):
    """Position (M1-M4) x Nelion (1-4) grid — a colored-circle visual of the
    plant's real module grid, showing which pair each of the 16 modules is
    physically grouped into for desorption in this configuration. Visually
    distinct from module_mapping_table (the CSV data-sourcing table)."""
    pair_colors = {pair: PAIR_CIRCLE_COLORS[i % len(PAIR_CIRCLE_COLORS)] for i, pair in enumerate(pairs)}
    module_to_pair = {
        module: pair
        for pair, modules in physical_modules_map.items()
        for module in modules
    }
    rows = []
    for position in ["M1", "M2", "M3", "M4"]:
        row = {"Position": position}
        for nelion in range(1, 5):
            pair = module_to_pair.get(f"N{nelion}-{position}")
            row[f"Nelion {nelion}"] = f"{pair_colors[pair]} {pair}" if pair else "—"
        rows.append(row)
    return pd.DataFrame(rows)

def load_plant_cycle_defaults_weighted(pairs, energy_module_map, yield_formula_map, since=None):
    """Return {pair: (avg eTotal kWh, avg BAG CO2 kg)}. Energy per Cycle normally
    comes from each pair's single matching Module (energy_module_map), but a
    pair's value can also be a weighted list like [("N3-M2n4", 1), ("N4-M2n4", 1)]
    (same shape as a Yield formula) when no single real combo covers that pair —
    this sums the listed modules' own Energy averages, weighted, same as Yield.
    Yield per Cycle always comes from a weighted sum of one or more Module
    averages (yield_formula_map, e.g. {"Group A": [("N1N2N3-M1n3", 1),
    ("N3-M2n4", 1)]}). A pair missing from either mapping gets 0.0 for that value.
    `since` is passed straight through to _module_average -- see its docstring."""
    plant_cycles_df = _plant_cycles_df()
    defaults = {}
    for pair in pairs:
        energy_source = energy_module_map.get(pair)
        if isinstance(energy_source, str) or energy_source is None:
            energy, _ = _module_average(plant_cycles_df, energy_source, since=since)
        else:
            energy = sum(
                weight * _module_average(plant_cycles_df, module, since=since)[0]
                for module, weight in energy_source
            )
        yield_total = 0.0
        for module, weight in yield_formula_map.get(pair, []):
            _, co2 = _module_average(plant_cycles_df, module, since=since)
            yield_total += weight * co2
        defaults[pair] = (round(energy, 2), round(yield_total, 2))
    return defaults

def load_plant_cycle_defaults_by_pair(pairs):
    """load_plant_cycle_defaults_weighted for the 6-6-4 config's own sourcing
    (PAIR_PLANT_MODULE for Energy, PAIR_YIELD_FORMULA for Yield)."""
    return load_plant_cycle_defaults_weighted(pairs, PAIR_PLANT_MODULE, PAIR_YIELD_FORMULA)

# Full Schedule Analysis's two pairs (Group A, Group B; 8 modules each) are now a
# real recorded combination -- N3 & N4's full 8 modules (all 4 positions on each)
# -- logged as its own Module value after a week of cycles on that grouping. This
# replaces the previous estimate (the SUM of two unrelated module-type averages:
# N1N2N3-M1n3 + N3-M2n4, neither of which was actually an 8-module reading).
TAB3_8_8_MODULE = "N3N4-M1n3M2n4"

def load_plant_cycle_defaults_tab3():
    """Return (avg eTotal kWh, avg BAG CO2 kg), the shared default for both
    Group A and Group B, sourced directly from TAB3_8_8_MODULE's own real
    average (no summing/proxying)."""
    return _module_average(_plant_cycles_df(), TAB3_8_8_MODULE)

# The source log doesn't have a standalone "Heating" column or an app-equivalent
# "Preheating" phase. Evacuation uses only the "Evacuation (min)" column on its
# own (Preheating time is dropped entirely, not folded into any phase). Both
# "in-between" transition columns are folded entirely into Heating, since
# there's no other Heating figure to use.
STAGE_PHASE_COLUMNS = {
    "Adsorption": ["Adsorption (min)"],
    "Evacuation": ["Evacuation (min)"],
    "NCG Purging": ["NCGs Purging (min)"],
    "Heating": ["NCGs Purging/Heating (min)", "Heating/CO2 Purging (min)"],
    "CO2 Purging": ["CO2 Purging (min)"],
    "Cooling": ["Cooling (min)"],
}

# Original hand-picked phase durations, used as a fallback wherever the stage log
# is missing or has no matching rows (Advanced Interleaved adds a 7th,
# Repressurization, which the log has no equivalent column for at all).
CORE_PHASE_FALLBACK_DURATIONS = {
    "Adsorption": 25, "Evacuation": 7, "NCG Purging": 2,
    "Heating": 20, "CO2 Purging": 40, "Cooling": 30,
}

@st.cache_data(ttl=300)
def _stage_timestamps_df():
    """Load the stage-duration log from MySQL, or None if the query/connection
    fails. Cached (5 min TTL) for the same reason as _plant_cycles_df() — this
    data is re-read for every configuration's phase-duration defaults on every
    rerun otherwise."""
    try:
        df = pd.read_sql(
            f"SELECT * FROM {STAGE_TIMESTAMPS_TABLE} WHERE has_data = 1", _mysql_engine()
        )
        df = df.rename(columns=STAGE_TIMESTAMPS_COLUMNS)
        for col in STAGE_TIMESTAMPS_NUMERIC_COLUMNS:
            # The live table stores these in hours; every "(min)"-named column
            # downstream (phase durations throughout this file) expects minutes.
            df[col] = pd.to_numeric(df[col], errors="coerce") * 60
        return df
    except Exception:
        return None

# The log has no Repressurization column at all, so that phase always keeps
# whatever value is already in the table (originally defaulted to 5 minutes).
def _phase_averages_for_module(stage_df, module, since=None):
    """{phase: avg minutes} for one Module's rows, or {} if there are none.
    A row whose phase total is missing (NaN) or exactly 0 is excluded from
    that phase's average rather than counted as a real 0-minute cycle; a
    phase left with no valid rows simply has no entry (same as a module with
    no rows at all). If `since` is given, only rows with start_time on or
    after it are averaged -- see _module_average's docstring."""
    subset = stage_df[stage_df["Module"] == module] if module else stage_df.iloc[0:0]
    if since is not None:
        subset = subset[pd.to_datetime(subset["start_time"]) >= since]
    if len(subset) == 0:
        return {}
    result = {}
    for phase, cols in STAGE_PHASE_COLUMNS.items():
        totals = subset[cols].sum(axis=1)
        totals = totals[totals.notna() & (totals != 0)]
        if len(totals) > 0:
            result[phase] = float(totals.mean())
    return result

def load_stage_duration_defaults_by_pair(pairs, pair_module_map=None, since=None):
    """Return {pair: {phase: avg minutes}} for the 6 phases in STAGE_PHASE_COLUMNS,
    sourced from each pair's matching Module rows (pair_module_map, default
    PAIR_PLANT_MODULE) in the live stage-duration data. A pair with a missing
    data/module/phase simply has no entry for it — callers should fall back to
    their own default. A pair's map value can also be a weighted list like
    [("N3-M2n4", 1), ("N4-M2n4", 1)] (same shape as a Yield formula) when no
    single real combo covers that pair — this averages (not sums) each listed
    module's own phase durations, weighted, since Duration represents a typical
    cycle's timing rather than a combined total. `since` is passed straight
    through to _phase_averages_for_module -- see _module_average's docstring."""
    pair_module_map = pair_module_map if pair_module_map is not None else PAIR_PLANT_MODULE
    stage_df = _stage_timestamps_df()
    if stage_df is None:
        return {pair: {} for pair in pairs}

    defaults = {}
    for pair in pairs:
        source = pair_module_map.get(pair)
        if isinstance(source, str) or source is None:
            phase_avgs = _phase_averages_for_module(stage_df, source, since=since)
            defaults[pair] = {phase: round(v, 1) for phase, v in phase_avgs.items()}
            continue
        # Weighted average across multiple real modules.
        total_weight = sum(weight for _, weight in source)
        combined = {}
        missing_any = False
        for module, weight in source:
            phase_avgs = _phase_averages_for_module(stage_df, module, since=since)
            if not phase_avgs:
                missing_any = True
                break
            for phase, value in phase_avgs.items():
                combined[phase] = combined.get(phase, 0.0) + weight * value
        defaults[pair] = {} if missing_any or total_weight == 0 else {
            phase: round(value / total_weight, 1) for phase, value in combined.items()
        }
    return defaults

def load_stage_duration_defaults_tab3():
    """Return {phase: avg minutes}, the shared phase-duration default for both
    Group A and Group B in Full Schedule Analysis, sourced directly from
    TAB3_8_8_MODULE's own real per-phase averages (missing/zero rows excluded,
    same as every other configuration's sourcing). Falls back to
    CORE_PHASE_FALLBACK_DURATIONS if the log is missing or has no matching rows
    for that module, or for any individual phase."""
    stage_df = _stage_timestamps_df()
    if stage_df is None:
        return dict(CORE_PHASE_FALLBACK_DURATIONS)

    phase_avgs = _phase_averages_for_module(stage_df, TAB3_8_8_MODULE)
    return {
        phase: round(phase_avgs[phase], 1) if phase in phase_avgs else CORE_PHASE_FALLBACK_DURATIONS[phase]
        for phase in STAGE_PHASE_COLUMNS
    }

def get_versioned_default_table(state_key, expected_columns, version, build_fn):
    """Return st.session_state[state_key], rebuilding it via build_fn() if it's
    missing, has the wrong columns, or was last built by an older `version` of the
    default-sourcing logic (tracked in a companion "<state_key>__version" key).
    This is what actually prevents a stale browser session from getting stuck on
    outdated default values after the sourcing logic changes (e.g. a CSV mapping
    fix) — bump `version` at the call site whenever that logic changes. A table a
    user has genuinely edited is left alone as long as `version` hasn't moved."""
    version_key = f"{state_key}__version"
    needs_rebuild = (
        state_key not in st.session_state
        or list(st.session_state[state_key].columns) != expected_columns
        or st.session_state.get(version_key) != version
    )
    if needs_rebuild:
        st.session_state[state_key] = build_fn()
        st.session_state[version_key] = version
    return st.session_state[state_key]

st.sidebar.markdown("### ℹ️ Guide")
st.sidebar.markdown("""
**1. Full Schedule Analysis**
- Two groups (A and B) in four plant configuration: 8-8, 4-4, 2-2, 6-6.
- Compares **Concurrent** vs **Interleaved** scheduling.
- Each configuration has its own tab: edit durations and capacities, then click **Generate**

**2. Advanced Interleaved**
- Three or four pairs: 6-6-4, 8-4-4, 6-5-5, 4-4-4-4, 2-2-2-2 (N3/N4)
- Adds Repressurization after Cooling
- Each configuration has its own tab with its own **Generate** button

**3. Yield vs Cycles**
- Compares cycles, yield and energy across every configuration
- Shows results from the last time each **Generate** was clicked

**Scheduling rules**
- First come, first served; no group has priority
- NCG Purging → Heating → CO2 Purging runs unbroken, one group at a time; the other group can evacuate, then waits
- Evacuation pauses another group's Cooling
- Only cycles that finish within the operating period count

*Durations come from live plant data; edit the tables if real timings differ.*
""")

with st.sidebar.expander("🛠️ Database diagnostics"):
    st.caption(
        "Raw view of what the live MySQL connection is actually returning -- use "
        "this to check why a pair's Energy/Yield/Duration default might be "
        "showing 0.0 (empty table, a failed connection, or a Module value that "
        "doesn't match the combo strings in PAIR_PLANT_MODULE/PAIR_YIELD_FORMULA, "
        "e.g. 'N1N2N3-M1n3')."
    )
    st.markdown("**Connection test**")
    try:
        with _mysql_engine().connect() as _conn:
            _conn.exec_driver_sql("SELECT 1")
        st.success("Connected to MySQL successfully.")
    except Exception as _conn_exc:
        st.error("Connection failed -- exception below (secrets are never included in this message):")
        st.code(f"{type(_conn_exc).__name__}: {_conn_exc}")

    def _freshness_caption(df, time_col):
        """How long ago the most recent row's timestamp was -- the quickest way
        to tell whether SCADA is still actively logging into this table, vs. a
        connection that works fine but is just reading a stale/frozen table."""
        if time_col not in df.columns or df[time_col].dropna().empty:
            st.caption("No timestamp column to check freshness against.")
            return
        latest = pd.to_datetime(df[time_col]).max()
        age = pd.Timestamp.now() - latest
        st.write(f"Most recent `{time_col}`: {latest} ({age} ago)")
        if age > pd.Timedelta(days=1):
            st.warning("Most recent row is over a day old -- logging may have stalled.")
        else:
            st.success("Most recent row is within the last day -- still logging.")

    _diag_cycles = _plant_cycles_df()
    _diag_stages = _stage_timestamps_df()
    st.markdown("**plant_cycles**")
    if _diag_cycles is None:
        st.error("Query failed -- see the connection test above for the exception.")
    else:
        st.write(f"{len(_diag_cycles)} rows")
        _freshness_caption(_diag_cycles, "start_time")
        st.write("Distinct `Module` values:")
        st.dataframe(_diag_cycles["Module"].value_counts())
    st.markdown("**stage_durations** (has_data = 1 only)")
    if _diag_stages is None:
        st.error("Query failed -- see the connection test above for the exception.")
    else:
        st.write(f"{len(_diag_stages)} rows")
        _freshness_caption(_diag_stages, "start_time")
        st.write("Distinct `Module` values:")
        st.dataframe(_diag_stages["Module"].value_counts())

    try:
        _water_df = pd.read_sql(
            "SELECT reading_date, meter_code, volume_liters FROM water_meter_readings "
            "WHERE meter_code IN ('BOILER_WATER_TANK', 'COOLING_TOWER')",
            _mysql_engine(),
        )
        _water_df["volume_liters"] = pd.to_numeric(_water_df["volume_liters"], errors="coerce")
        _water_df["reading_date"] = pd.to_datetime(_water_df["reading_date"])
        st.markdown("**water_meter_readings**")
        st.write(f"{len(_water_df)} rows (BOILER_WATER_TANK + COOLING_TOWER)")
        _freshness_caption(_water_df, "reading_date")
        _water_df["reading_date"] = _water_df["reading_date"].dt.date
        _n3n4_modules = ["N3N4-M1n3", "N3N4-M2n4"]

        st.markdown("**Water consumption per cycle — single-day breakdown (N3N4-M1n3 + N3N4-M2n4)**")
        _available_dates = sorted(_water_df["reading_date"].dropna().unique())
        if not _available_dates:
            st.write("No water_meter_readings rows for BOILER_WATER_TANK/COOLING_TOWER.")
        elif _diag_cycles is None or "start_time" not in _diag_cycles.columns:
            st.write("plant_cycles has no `start_time` column to match cycles to a day.")
        else:
            _selected_date = st.selectbox(
                "Day", _available_dates, index=len(_available_dates) - 1, key="water_daily_date",
            )
            _day_total = float(
                _water_df.loc[_water_df["reading_date"] == _selected_date, "volume_liters"].sum()
            )
            _cycle_dates = pd.to_datetime(_diag_cycles["start_time"]).dt.date
            _n3n4_cycles_that_day = int(
                (_diag_cycles["Module"].isin(_n3n4_modules) & (_cycle_dates == _selected_date)).sum()
            )
            st.write(f"Total water (boiler + cooling tower) on {_selected_date}: {_day_total:,.0f} L")
            st.write(f"N3N4-M1n3 + N3N4-M2n4 cycles on {_selected_date}: {_n3n4_cycles_that_day}")
            if _n3n4_cycles_that_day > 0:
                st.write(f"Average: {_day_total / _n3n4_cycles_that_day:,.1f} L/cycle")
            else:
                st.write("No N3N4-M1n3 cycles that day -- can't compute a per-cycle average.")
    except Exception as _water_exc:
        st.error("Water query failed -- exception below:")
        st.code(f"{type(_water_exc).__name__}: {_water_exc}")

IDLE_CAUSE_LABELS = {
    ("Adsorption", "Evacuation"): "Waiting to start Evacuation (after Adsorption)",
    ("Evacuation", "NCG Purging"): "Waiting for the NCG → Heating → CO2 block (after Evacuation)",
    ("CO2 Purging", "Cooling"): "Waiting to start Cooling (after CO2 Purging)",
    ("Cooling", "Cooling"): "Cooling paused for another Evacuation",
    ("Cooling", "Adsorption"): "Waiting to start Adsorption (after Cooling)",
    ("Cooling", "Repressurization"): "Waiting to start Repressurization (after Cooling)",
    ("Repressurization", "Adsorption"): "Waiting to start Adsorption (after Repressurization)",
}


def idle_breakdown_table(schedule_df, unit_col, units_by_label, total_minutes):
    """Average idle minutes per unit (module or pair) in each column's group,
    split by what the unit was waiting for. Each column adds up to that group's
    average Idle (min) in the utilisation tables."""
    totals = {label: {} for label in units_by_label}
    for label, units in units_by_label.items():
        for unit in units:
            rows = schedule_df[schedule_df[unit_col] == unit].sort_values("Start")
            cursor, prev_phase = 0, None
            for start, end, phase in zip(rows["Start"], rows["End"], rows["Phase"]):
                start, end = min(start, total_minutes), min(end, total_minutes)
                if start > cursor:
                    cause = ("Start of day (before first phase)" if prev_phase is None
                             else IDLE_CAUSE_LABELS.get((prev_phase, phase), f"Waiting between {prev_phase} and {phase}"))
                    totals[label][cause] = totals[label].get(cause, 0) + (start - cursor) / len(units)
                cursor, prev_phase = max(cursor, end), phase
            if cursor < total_minutes:
                cause = "End of day (next phase couldn't start before the period ended)"
                totals[label][cause] = totals[label].get(cause, 0) + (total_minutes - cursor) / len(units)
    causes = sorted({c for t in totals.values() for c in t}, key=lambda c: -sum(t.get(c, 0) for t in totals.values()))
    table = pd.DataFrame({
        "Waiting for": causes + ["Total idle"],
        **{
            f"{label} (min)": [round(totals[label].get(c, 0), 1) for c in causes] + [round(sum(totals[label].values()), 1)]
            for label in units_by_label
        },
    })
    return table


st.markdown("<h1 style='text-align: center;'>Nelion Cycle Schedule</h1>", unsafe_allow_html=True)
# Spread the app's main tabs evenly across the full width instead of bunching
# them on the left. Tabs nested inside a tab (the configuration tabs) keep
# Streamlit's normal left-aligned layout.
st.markdown(
    """
    <style>
    [data-testid="stTabs"] [role="tablist"] {
        width: 100% !important;
        display: flex !important;
    }
    [data-testid="stTabs"] [role="tablist"] [role="tab"] {
        flex: 1 1 0 !important;
        justify-content: center !important;
    }
    [data-testid="stTabs"] [role="tabpanel"] [role="tablist"] [role="tab"] {
        flex: 0 0 auto !important;
        justify-content: flex-start !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Increments on every rerun; each result published to process_comparison is
# stamped with it, so Yield vs Cycles can tell which results were generated last.
st.session_state["app_run_id"] = st.session_state.get("app_run_id", 0) + 1
tab3, tab4, tab5 = st.tabs(["Full Schedule Analysis", "Advanced Interleaved", "Yield vs Cycles"])

if False:  # Module pair analysis removed (fan pairing fixed)
          # Everything inside this `if False:` block is DISABLED and never runs — it's
          # earlier fan-pairing scheduling code kept only for reference. It has been
          # superseded by the live tabs below (tab1-tab4). Skip this block if you're
          # trying to understand the app's current behavior.
          # ===============================
     # Serialized/Interleaved simulator WITH FAN-PAIRING (A→…→C order preserved)
     # CORRECTED: 2-slot D-chain lock in Serialized mode + per-module durations
     # ENHANCED: Performance metrics dashboard + advanced visualizations
     # FIXED: Each module reserves its own D-chain slot during tentative scheduling
     # ===============================     
     # ===============================
          # Concurrent/Interleaved simulator WITH FAN-PAIRING (A→…→C order preserved)
          # CORRECTED: 2-slot D-chain lock in Concurrent mode + per-module durations
          # ENHANCED: Performance metrics dashboard + advanced visualizations
          # FIXED: Each module reserves its own D-chain slot during tentative scheduling
          # ===============================

          def _build_fan_map(modules, fan_pairs):
              """Assign each module a shared-fan group id so paired modules can't adsorb at the same time."""
              fan_of = {}
              next_fan = 0
              pairs = fan_pairs or []
              for a, b in pairs:
                  if a not in fan_of and b not in fan_of:
                      fan_of[a] = fan_of[b] = next_fan; next_fan += 1
                  elif a in fan_of and b not in fan_of:
                      fan_of[b] = fan_of[a]
                  elif b in fan_of and a not in fan_of:
                      fan_of[a] = fan_of[b]
              for m in modules:
                  if m not in fan_of:
                      fan_of[m] = next_fan; next_fan += 1
              return fan_of
          
          def _dur_for(module_id, phase, PER_MODULE_DURATIONS, PHASE_DURATIONS):
              """Return per-module phase duration if provided, else default."""
              if PER_MODULE_DURATIONS and module_id in PER_MODULE_DURATIONS and phase in PER_MODULE_DURATIONS[module_id]:
                  return int(PER_MODULE_DURATIONS[module_id][phase])
              return int(PHASE_DURATIONS[phase])
          
          def simulate_schedule_pairing(
              MODULES,
              PHASES,
              DESORPTION_PHASES,
              PHASE_DURATIONS,
              TOTAL_MINUTES,
              *, desorption_mode="Interleaved",
              MODULE_DELAYS=None,
              PHASE_GAPS=None,
              steam_demand_per_phase=None,
              fan_pairs=None,
              ads_fan_exclusive=True,
              desorption_capacity=None,
              cooling_capacity=None,
              PER_MODULE_DURATIONS=None
          ):
              """Full multi-module scheduler with shared-fan exclusivity and a locked D-chain
              (Evacuation->NCG Purging->Heating->CO2 Purging) slot system. Superseded by the
              live tabs; not used."""
              import numpy as np
              import pandas as pd

              if MODULE_DELAYS is None:
                  MODULE_DELAYS = {m: 0 for m in MODULES}
              if PHASE_GAPS is None:
                  PHASE_GAPS = {}
              if PER_MODULE_DURATIONS is None:
                  PER_MODULE_DURATIONS = {}
          
              nmods = len(MODULES)
          
              # ---------------- Capacities ----------------
              if desorption_mode == "Concurrent":
                  d_cap = 2
                  c_cap = 2
              else:
                  d_cap = nmods if desorption_capacity is None else int(desorption_capacity)
                  c_cap = nmods if cooling_capacity    is None else int(cooling_capacity)
          
              RESOURCE_LIMITS = {}
              for p in PHASES:
                  if p == "Cooling":
                      RESOURCE_LIMITS[p] = c_cap
                  elif p in DESORPTION_PHASES:
                      RESOURCE_LIMITS[p] = d_cap
                  else:
                      RESOURCE_LIMITS[p] = nmods
          
              # ---------------- Timelines ----------------
              resource_usage = {p: np.zeros(TOTAL_MINUTES, dtype=int) for p in PHASES}
              module_timers  = {m: int(MODULE_DELAYS.get(m, 0)) for m in MODULES}
          
              # ---------------- Fan exclusivity ----------------
              fan_of  = _build_fan_map(MODULES, fan_pairs)
              fan_ids = sorted(set(fan_of.values()))
              fan_occ = {fid: np.zeros(TOTAL_MINUTES, dtype=int) for fid in fan_ids}
          
              # ---------------- D-chain control ----------------
              DCHAIN_PHASES = ["Evacuation", "NCG Purging", "Heating", "CO2 Purging"]
              PHASE_ID = {p: i for i, p in enumerate(DCHAIN_PHASES)}
          
              if desorption_mode == "Interleaved":
                  dphase_id    = np.full(TOTAL_MINUTES, -1, dtype=int)
                  dphase_count = np.zeros(TOTAL_MINUTES, dtype=int)
          
              if desorption_mode == "Concurrent":
                  serialized_locks = [0, 0]
                  # Track which modules have reserved which slots in THIS iteration
                  slots_reserved_by = {}  # {module_id: slot_index}
          
              # ---------------- Helpers ----------------
              def can_phase(phase, s, d, usage, lim):
                  """True if [s, s+d) fits within phase's capacity limit."""
                  e = s + d
                  return e <= TOTAL_MINUTES and np.all(usage[phase][s:e] < lim)

              def reserve_phase(phase, s, d, usage):
                  """Mark [s, s+d) as occupied for this phase."""
                  usage[phase][s:s+d] += 1

              def can_fan(mod, phase, s, d, _fans):
                  """True if this module's shared fan is free for [s, s+d) during Adsorption."""
                  if not ads_fan_exclusive or phase != "Adsorption":
                      return True
                  e = s + d
                  fid = fan_of[mod]
                  return e <= TOTAL_MINUTES and np.all(_fans[fid][s:e] == 0)

              def reserve_fan(mod, phase, s, d, _fans):
                  """Occupy this module's shared fan for [s, s+d) during Adsorption."""
                  if ads_fan_exclusive and phase == "Adsorption":
                      fid = fan_of[mod]
                      _fans[fid][s:s+d] += 1
          
              schedule_rows = []
              progressed = True
              while progressed:
                  progressed = False
                  
                  # Reset slot reservations for this iteration
                  if desorption_mode == "Concurrent":
                      slots_reserved_by = {}
                      iteration_serialized_locks = serialized_locks.copy()
                  
                  # Try modules in order of ready time for Concurrent mode (Sequential pairing)
                  modules_order = list(MODULES)
                  if desorption_mode == "Concurrent" and fan_pairs == [(1, 2), (3, 4)]:
                      # Sort by time, then prioritize odd modules (1, 3) over even (2, 4)
                      modules_order = sorted(MODULES, key=lambda m: (module_timers[m], m % 2 == 0, m))
                  
                  for mod in modules_order:
                      t = module_timers[mod]
                      tmp_usage = {p: np.copy(resource_usage[p]) for p in PHASES}
                      tmp_fans  = {fid: np.copy(arr) for fid, arr in fan_occ.items()}
          
                      if desorption_mode == "Interleaved":
                          tmp_dphase_id    = np.copy(dphase_id)
                          tmp_dphase_count = np.copy(dphase_count)
          
                      if desorption_mode == "Concurrent":
                          # Use the iteration-level locks that track all reservations so far
                          tmp_serialized_locks = iteration_serialized_locks.copy()
                          my_slot = None
          
                      cycle = []
                      ok = True
                      for idx, phase in enumerate(PHASES):
                          dur = int(PER_MODULE_DURATIONS.get(mod, {}).get(phase, PHASE_DURATIONS[phase]))
          
                          if desorption_mode == "Concurrent" and (phase in DESORPTION_PHASES or phase == "Cooling"):
                              # Find the earliest available slot using SHARED iteration locks
                              earliest_slot_time = min(iteration_serialized_locks)
                              earliest_slot_idx = iteration_serialized_locks.index(earliest_slot_time)
                              
                              # Reserve this slot for this module
                              if my_slot is None:
                                  my_slot = earliest_slot_idx
                              
                              t = max(t, earliest_slot_time)
          
                          s = t
                          while s + dur <= TOTAL_MINUTES:
                              # For Concurrent mode D-chain phases, skip capacity check (use locks only)
                              if desorption_mode == "Concurrent" and (phase in DESORPTION_PHASES or phase == "Cooling"):
                                  ok_caps = True  # Don't check capacity, locks handle it
                              else:
                                  ok_caps = can_phase(phase, s, dur, tmp_usage, RESOURCE_LIMITS[phase])
                              
                              ok_fan  = can_fan(mod, phase, s, dur, tmp_fans)
                              
                              if desorption_mode == "Interleaved":
                                  if phase in PHASE_ID:
                                      e = s + dur
                                      pid = PHASE_ID[phase]
                                      seg_id  = tmp_dphase_id[s:e]
                                      seg_cnt = tmp_dphase_count[s:e]
                                      same_or_empty = (seg_id == -1) | (seg_id == pid)
                                      ok_dchain = np.all(same_or_empty & (seg_cnt < d_cap))
                                  else:
                                      ok_dchain = True
                              else:
                                  ok_dchain = True
          
                              if ok_caps and ok_fan and ok_dchain:
                                  break
                              s += 1
          
                          if s + dur > TOTAL_MINUTES:
                              ok = False
                              break
          
                          reserve_phase(phase, s, dur, tmp_usage)
                          reserve_fan(mod, phase, s, dur, tmp_fans)
          
                          if desorption_mode == "Interleaved" and phase in PHASE_ID:
                              e = s + dur
                              pid = PHASE_ID[phase]
                              sl = slice(s, e)
                              empty_mask = (tmp_dphase_id[sl] == -1)
                              tmp_dphase_id[sl][empty_mask] = pid
                              tmp_dphase_count[sl] += 1
          
                          cycle.append((phase, s, s + dur))
                          t = s + dur
          
                          if idx + 1 < len(PHASES):
                              nxt = PHASES[idx + 1]
                              t += int(PHASE_GAPS.get((phase, nxt), 0))
          
                          if desorption_mode == "Concurrent" and phase == "Cooling":
                              # Update the slot this module is using
                              if my_slot is not None:
                                  tmp_serialized_locks[my_slot] = t
                                  # Update the iteration locks too
                                  iteration_serialized_locks[my_slot] = max(iteration_serialized_locks[my_slot], t)
          
                      if ok:
                          for p in PHASES:
                              resource_usage[p] = np.copy(tmp_usage[p])
                          for fid in fan_occ:
                              fan_occ[fid] = np.copy(tmp_fans[fid])
          
                          if desorption_mode == "Interleaved":
                              dphase_id[:]    = tmp_dphase_id[:]
                              dphase_count[:] = tmp_dphase_count[:]
          
                          if desorption_mode == "Concurrent":
                              # Commit the slot reservation
                              if my_slot is not None:
                                  serialized_locks[my_slot] = tmp_serialized_locks[my_slot]
                                  slots_reserved_by[mod] = my_slot
          
                          for phase, s, e in cycle:
                              schedule_rows.append({"Module": mod, "Phase": phase, "Start": s, "End": e})
                          module_timers[mod] = t
                          progressed = True
                          
                          # For Concurrent mode (Sequential pairing): commit one module at a time
                          # This allows M3 to grab a slot before M2 in the next iteration
                          if desorption_mode == "Concurrent" and fan_pairs == [(1, 2), (3, 4)]:
                              break  # Exit the for loop, start next iteration
          
              df_schedule = pd.DataFrame(schedule_rows).sort_values(["Module", "Start"]).reset_index(drop=True)
              steam_profile = pd.Series([0.0]*TOTAL_MINUTES, index=range(TOTAL_MINUTES))
              return df_schedule, steam_profile, resource_usage
          
          def count_complete_cycles(df_schedule, MODULES, PHASES):
              """Count, per module, how many times its scheduled phases form one full PHASES sequence in order."""
              out = []
              for mod in MODULES:
                  md = df_schedule[df_schedule["Module"] == mod].sort_values("Start").reset_index(drop=True)
                  i = 0; cnt = 0
                  while i <= len(md) - len(PHASES):
                      win = md.iloc[i:i+len(PHASES)]
                      if list(win["Phase"]) == PHASES:
                          cnt += 1; i += len(PHASES)
                      else:
                          i += 1
                  out.append({"Module": mod, "Complete Cycles": cnt})
              return pd.DataFrame(out)
          
          def calculate_throughput(cycles_df, module_co2_capture):
              """Calculate CO2 throughput based on complete cycles and per-module capture rates."""
              throughput_df = cycles_df.copy()
              throughput_df["CO₂/cycle (kg)"] = throughput_df["Module"].map(module_co2_capture)
              throughput_df["CO₂ Captured (kg)"] = throughput_df["Complete Cycles"] * throughput_df["CO₂/cycle (kg)"]
              return throughput_df
          
          def create_combined_metrics(df_schedule, cycles_df, modules, total_minutes, module_co2_capture):
              """Create a combined metrics table: cycles, CO2, utilization, and active minutes."""
              # Get utilization data
              util_data = []
              for mod in modules:
                  mod_data = df_schedule[df_schedule["Module"] == mod]
                  if len(mod_data) > 0:
                      total_active = (mod_data["End"] - mod_data["Start"]).sum()
                      utilization = (total_active / total_minutes) * 100
                  else:
                      utilization = 0.0
                      total_active = 0
                  util_data.append({
                      "Module": mod, 
                      "Utilization %": round(utilization, 1), 
                      "Active Minutes": int(total_active)
                  })
              util_df = pd.DataFrame(util_data)
              
              # Calculate throughput
              throughput_df = calculate_throughput(cycles_df, module_co2_capture)
              
              # Combine all metrics
              combined = pd.merge(throughput_df, util_df, on="Module")
              combined["Module"] = combined["Module"].apply(lambda x: f"M{x}")
              
              # Reorder columns
              combined = combined[["Module", "Complete Cycles", "CO₂/cycle (kg)", "CO₂ Captured (kg)", 
                                   "Utilization %", "Active Minutes"]]
              return combined
          
          def plot_phase_distribution(df_schedule, phases, title):
              """Create a pie chart showing time distribution across phases."""
              phase_times = {}
              for phase in phases:
                  phase_data = df_schedule[df_schedule["Phase"] == phase]
                  total_time = (phase_data["End"] - phase_data["Start"]).sum()
                  phase_times[phase] = total_time
              
              colors = {
                  'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
                  'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'
              }
              
              fig, ax = plt.subplots(figsize=(8, 6))
              wedges, texts, autotexts = ax.pie(
                  phase_times.values(), 
                  labels=phase_times.keys(),
                  autopct='%1.1f%%',
                  colors=[colors.get(p, '#888') for p in phase_times.keys()],
                  startangle=90
              )
              
              for autotext in autotexts:
                  autotext.set_color('white')
                  autotext.set_fontweight('bold')
              
              ax.set_title(title)
              plt.tight_layout()
              return fig
          
          def plot_resource_usage_timeline(resource_usage, phases, total_minutes, title):
              """Plot resource usage over time for each phase."""
              fig, ax = plt.subplots(figsize=(16, 6))
              
              colors = {
                  'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
                  'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'
              }
              
              time_axis = range(total_minutes)
              for phase in phases:
                  if phase in resource_usage:
                      ax.plot(time_axis, resource_usage[phase], label=phase, 
                             color=colors.get(phase, '#888'), linewidth=1.5, alpha=0.7)
              
              ax.set_xlabel("Time (min)")
              ax.set_ylabel("Number of Modules Active")
              ax.set_title(title)
              ax.legend(loc='upper right', fontsize=8)
              ax.grid(True, alpha=0.3)
              plt.tight_layout()
              return fig
          
          def plot_gantt(df, modules, module_labels, phases, total_minutes, title):
              """Draw a per-module horizontal bar (Gantt) chart, one row per module, colored by phase."""
              colors = {
                  'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
                  'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'
              }
              df_plot = df.copy()
              id_to_label = {m: f"M{m}" for m in modules}
              df_plot["ModuleLabel"] = df_plot["Module"].map(id_to_label)
          
              fig, ax = plt.subplots(figsize=(16, 6))
              df_plot["ModuleLabel"] = pd.Categorical(df_plot["ModuleLabel"], categories=module_labels, ordered=True)
              df_plot = df_plot.sort_values(["ModuleLabel", "Start"])
              for _, r in df_plot.iterrows():
                  ax.barh(r["ModuleLabel"], r["End"] - r["Start"], left=r["Start"],
                          color=colors.get(r["Phase"], "#888"), edgecolor="black", linewidth=0.8)
              ax.set_xlim(0, total_minutes)
              ax.set_title(title)
              ax.set_xlabel("Time (min)")
              ax.set_ylabel("Modules")
              ax.grid(True, axis='x', linestyle='--', alpha=0.4)
              ax.legend([plt.Rectangle((0,0),1,1,color=colors[p]) for p in phases], phases,
                        loc='upper right', fontsize=8)
              return fig
          
          def create_pairing_options(n):
              """Build pairing options for modules 1..n (Sequential and Alternate only)."""
              modules = list(range(1, n+1))
              opts = {}
          
              # Sequential (1-2, 3-4, ...)
              seq = []
              for i in range(0, n, 2):
                  if i + 1 < n:
                      seq.append((modules[i], modules[i+1]))
              if seq:
                  opts["Sequential (M1&M2, M3&M4)"] = seq
          
              # Alternate (1-3, 2-4) then sequential for the rest
              if n >= 4 and n % 2 == 0:
                  alt = [(1, 3), (2, 4)]
                  for i in range(4, n, 2):
                      if i + 1 < n:
                          alt.append((i+1, i+2))
                  opts["Alternate (M1&M3, M2&M4)"] = alt
          
              return opts
          
          # ========== STREAMLIT APP ==========
          st.title("Enhanced Analytics Per Module")
          
          n = 4
          modules = list(range(1, n+1))
          module_labels = [f"M{i}" for i in modules]
          
          PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
          DESORPTION_PHASES = {'Evacuation','NCG Purging','Heating','CO2 Purging','Cooling'}
          PHASE_DURATIONS = {'Adsorption':25,'Evacuation':7,'NCG Purging':2,'Heating':20,'CO2 Purging':40,'Cooling':30}
          
          st.markdown("#### Per-module phase durations (minutes)")
          df_durs = pd.DataFrame({
              "Module": module_labels,
              "Adsorption": [PHASE_DURATIONS['Adsorption']]*n,
              "Evacuation": [PHASE_DURATIONS['Evacuation']]*n,
              "NCG Purging": [PHASE_DURATIONS['NCG Purging']]*n,
              "Heating": [PHASE_DURATIONS['Heating']]*n,
              "CO2 Purging": [PHASE_DURATIONS['CO2 Purging']]*n,
              "Cooling": [PHASE_DURATIONS['Cooling']]*n,
              "CO₂/cycle (kg)": [0.7]*n,  # Default CO2 capture per cycle
          })
          df_durs = st.data_editor(df_durs, num_rows="fixed", use_container_width=True)
          
          PER_MODULE_DURATIONS = {}
          MODULE_CO2_CAPTURE = {}
          for _, r in df_durs.iterrows():
              mid = int(str(r["Module"]).replace("M",""))
              PER_MODULE_DURATIONS[mid] = {
                  "Adsorption": int(r["Adsorption"]),
                  "Evacuation": int(r["Evacuation"]),
                  "NCG Purging": int(r["NCG Purging"]),
                  "Heating": int(r["Heating"]),
                  "CO2 Purging": int(r["CO2 Purging"]),
                  "Cooling": int(r["Cooling"]),
              }
              MODULE_CO2_CAPTURE[mid] = float(r["CO₂/cycle (kg)"])
          
          horizon = st.number_input("Analysis period (min)", 60, 1440, 600, step=10)
          pairing_options = create_pairing_options(n)
          
          comparison_data = []
          
          for name, pairs in pairing_options.items():
              st.markdown(f"## 🔧 {name}")
              
              # SERIALIZED
              df_ser, _, res_ser = simulate_schedule_pairing(
                  MODULES=modules, PHASES=PHASES, DESORPTION_PHASES=DESORPTION_PHASES,
                  PHASE_DURATIONS=PHASE_DURATIONS, TOTAL_MINUTES=horizon,
                  desorption_mode="Concurrent", fan_pairs=pairs, ads_fan_exclusive=True,
                  PER_MODULE_DURATIONS=PER_MODULE_DURATIONS
              )
              
              st.markdown("### Concurrent Mode")
              fig_ser = plot_gantt(df_ser, modules, module_labels, PHASES, horizon, f"{name}: Concurrent")
              st.pyplot(fig_ser)
              
              st.markdown("#### Performance Metrics")
              cycles_ser = count_complete_cycles(df_ser, modules, PHASES)
              combined_ser = create_combined_metrics(df_ser, cycles_ser, modules, horizon, MODULE_CO2_CAPTURE)
              
              # Summary metrics
              total_cycles_ser = cycles_ser["Complete Cycles"].sum()
              total_co2_ser = combined_ser["CO₂ Captured (kg)"].sum()
              avg_util_ser = combined_ser["Utilization %"].mean()
              
              metric_col1, metric_col2, metric_col3 = st.columns(3)
              with metric_col1:
                  st.metric("Total Complete Cycles", total_cycles_ser)
              with metric_col2:
                  st.metric("Total CO₂ Captured", f"{total_co2_ser:.1f} kg")
              with metric_col3:
                  st.metric("Average Utilization", f"{avg_util_ser:.1f}%")
              
              st.dataframe(combined_ser, use_container_width=True, hide_index=True)
              
              with st.expander("Advanced Analytics - Concurrent"):
                  viz_col1, viz_col2 = st.columns(2)
                  with viz_col1:
                      fig_phase_ser = plot_phase_distribution(df_ser, PHASES, "Concurrent: Phase Distribution")
                      st.pyplot(fig_phase_ser)
                  with viz_col2:
                      fig_res_ser = plot_resource_usage_timeline(res_ser, PHASES, horizon, "Concurrent: Resource Usage")
                      st.pyplot(fig_res_ser)
              
              # INTERLEAVED
              df_int, _, res_int = simulate_schedule_pairing(
                  MODULES=modules, PHASES=PHASES, DESORPTION_PHASES=DESORPTION_PHASES,
                  PHASE_DURATIONS=PHASE_DURATIONS, TOTAL_MINUTES=horizon,
                  desorption_mode="Interleaved", fan_pairs=pairs, ads_fan_exclusive=True,
                  desorption_capacity=2, cooling_capacity=2,
                  PER_MODULE_DURATIONS=PER_MODULE_DURATIONS
              )
              
              st.markdown("### Interleaved Mode")
              fig_int = plot_gantt(df_int, modules, module_labels, PHASES, horizon, f"{name}: Interleaved")
              st.pyplot(fig_int)
              
              st.markdown("#### Performance Metrics")
              cycles_int = count_complete_cycles(df_int, modules, PHASES)
              combined_int = create_combined_metrics(df_int, cycles_int, modules, horizon, MODULE_CO2_CAPTURE)
              
              # Summary metrics
              total_cycles_int = cycles_int["Complete Cycles"].sum()
              total_co2_int = combined_int["CO₂ Captured (kg)"].sum()
              avg_util_int = combined_int["Utilization %"].mean()
              
              metric_col1, metric_col2, metric_col3 = st.columns(3)
              with metric_col1:
                  st.metric("Total Complete Cycles", total_cycles_int)
              with metric_col2:
                  st.metric("Total CO₂ Captured", f"{total_co2_int:.1f} kg")
              with metric_col3:
                  st.metric("Average Utilization", f"{avg_util_int:.1f}%")
              
              st.dataframe(combined_int, use_container_width=True, hide_index=True)
              
              with st.expander("Advanced Analytics - Interleaved"):
                  viz_col1, viz_col2 = st.columns(2)
                  with viz_col1:
                      fig_phase_int = plot_phase_distribution(df_int, PHASES, "Interleaved: Phase Distribution")
                      st.pyplot(fig_phase_int)
                  with viz_col2:
                      fig_res_int = plot_resource_usage_timeline(res_int, PHASES, horizon, "Interleaved: Resource Usage")
                      st.pyplot(fig_res_int)
              
              comparison_data.append({
                  "Configuration": name, "Mode": "Concurrent", "Total Cycles": total_cycles_ser,
                  "Cycles/Hour": round(total_cycles_ser / (horizon / 60), 2),
                  "CO₂ Captured (kg)": round(total_co2_ser, 1),
                  "kg/Hour": round(total_co2_ser / (horizon / 60), 2),
                  "Avg Utilization %": round(avg_util_ser, 1),
                  "Total Module-Hours": round(combined_ser["Active Minutes"].sum() / 60, 1)
              })
              comparison_data.append({
                  "Configuration": name, "Mode": "Interleaved", "Total Cycles": total_cycles_int,
                  "Cycles/Hour": round(total_cycles_int / (horizon / 60), 2),
                  "CO₂ Captured (kg)": round(total_co2_int, 1),
                  "kg/Hour": round(total_co2_int / (horizon / 60), 2),
                  "Avg Utilization %": round(avg_util_int, 1),
                  "Total Module-Hours": round(combined_int["Active Minutes"].sum() / 60, 1)
              })
              
              st.markdown("---")
          
          # DASHBOARD
          st.markdown("## Performance Comparison Dashboard")
          df_comparison = pd.DataFrame(comparison_data)
          
          best_cycles = df_comparison["Total Cycles"].max()
          best_util = df_comparison["Avg Utilization %"].max()
          best_throughput = df_comparison["Cycles/Hour"].max()
          best_co2 = df_comparison["CO₂ Captured (kg)"].max()
          best_co2_rate = df_comparison["kg/Hour"].max()
          
          st.markdown("### Key Metrics Summary")
          metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
          with metric_col1:
              st.metric("🥇 Best Total Cycles", int(best_cycles))
              best_config_cycles = df_comparison[df_comparison["Total Cycles"] == best_cycles].iloc[0]
              st.caption(f"{best_config_cycles['Configuration']} - {best_config_cycles['Mode']}")
          with metric_col2:
              st.metric("⚡ Best Throughput", f"{best_throughput} cycles/hr")
              best_config_throughput = df_comparison[df_comparison["Cycles/Hour"] == best_throughput].iloc[0]
              st.caption(f"{best_config_throughput['Configuration']} - {best_config_throughput['Mode']}")
          with metric_col3:
              st.metric("🌍 Best CO₂ Capture", f"{best_co2} kg")
              best_config_co2 = df_comparison[df_comparison["CO₂ Captured (kg)"] == best_co2].iloc[0]
              st.caption(f"{best_config_co2['Configuration']} - {best_config_co2['Mode']}")
          with metric_col4:
              st.metric("📊 Best Utilization", f"{best_util}%")
              best_config_util = df_comparison[df_comparison["Avg Utilization %"] == best_util].iloc[0]
              st.caption(f"{best_config_util['Configuration']} - {best_config_util['Mode']}")
          
          st.markdown("### Detailed Comparison Table")
          st.dataframe(
              df_comparison.style.highlight_max(subset=["Total Cycles", "Cycles/Hour", "CO₂ Captured (kg)", "kg/Hour", "Avg Utilization %"], color='lightgreen'),
              use_container_width=True, hide_index=True
          )
          
          st.markdown("### Visual Comparisons")
          comp_col1, comp_col2 = st.columns(2)
          with comp_col1:
              fig_bar, ax = plt.subplots(figsize=(10, 6))
              df_pivot = df_comparison.pivot(index="Configuration", columns="Mode", values="Total Cycles")
              df_pivot.plot(kind='bar', ax=ax, color=['#4B9CD3', '#E97451'])
              ax.set_title("Total Cycles by Configuration & Mode")
              ax.set_ylabel("Total Cycles")
              ax.set_xlabel("Configuration")
              ax.legend(title="Mode")
              plt.xticks(rotation=45, ha='right')
              plt.tight_layout()
              st.pyplot(fig_bar)
          
          with comp_col2:
              fig_bar2, ax = plt.subplots(figsize=(10, 6))
              df_pivot2 = df_comparison.pivot(index="Configuration", columns="Mode", values="CO₂ Captured (kg)")
              df_pivot2.plot(kind='bar', ax=ax, color=['#90EE90', '#FFB347'])
              ax.set_title("CO₂ Captured by Configuration & Mode")
              ax.set_ylabel("CO₂ Captured (kg)")
              ax.set_xlabel("Configuration")
              ax.legend(title="Mode")
              plt.xticks(rotation=45, ha='right')
              plt.tight_layout()
              st.pyplot(fig_bar2)
          
          comp_col3, comp_col4 = st.columns(2)
          with comp_col3:
              fig_bar3, ax = plt.subplots(figsize=(10, 6))
              df_pivot3 = df_comparison.pivot(index="Configuration", columns="Mode", values="kg/Hour")
              df_pivot3.plot(kind='bar', ax=ax, color=['#9370DB', '#FFD700'])
              ax.set_title("CO₂ Capture Rate by Configuration & Mode")
              ax.set_ylabel("kg/Hour")
              ax.set_xlabel("Configuration")
              ax.legend(title="Mode")
              plt.xticks(rotation=45, ha='right')
              plt.tight_layout()
              st.pyplot(fig_bar3)
          
          with comp_col4:
              fig_bar4, ax = plt.subplots(figsize=(10, 6))
              df_pivot4 = df_comparison.pivot(index="Configuration", columns="Mode", values="Avg Utilization %")
              df_pivot4.plot(kind='bar', ax=ax, color=['#E97451', '#4B9CD3'])
              ax.set_title("Average Utilization by Configuration & Mode")
              ax.set_ylabel("Utilization %")
              ax.set_xlabel("Configuration")
              ax.legend(title="Mode")
              plt.xticks(rotation=45, ha='right')
              plt.tight_layout()
              st.pyplot(fig_bar4)
          
          st.markdown("### 💡 Recommendations")
          best_config_cycles = df_comparison[df_comparison["Total Cycles"] == best_cycles].iloc[0]
          best_config_co2 = df_comparison[df_comparison["CO₂ Captured (kg)"] == best_co2].iloc[0]
          
          if best_config_cycles["Mode"] == "Interleaved":
              st.success(f"✅ **Interleaved mode with {best_config_cycles['Configuration']}** achieves the highest throughput ({int(best_cycles)} cycles, {best_co2:.1f} kg CO₂).")
          else:
              st.info(f"✅ **Concurrent mode with {best_config_cycles['Configuration']}** achieves the highest throughput ({int(best_cycles)} cycles, {best_co2:.1f} kg CO₂).")
          
          # Show top 3 configurations
          st.markdown("#### 📈 Top 3 Configurations by CO₂ Capture")
          top_3 = df_comparison.nlargest(3, "CO₂ Captured (kg)")[["Configuration", "Mode", "Total Cycles", "CO₂ Captured (kg)", "kg/Hour", "Avg Utilization %"]]
          st.dataframe(top_3, use_container_width=True, hide_index=True)
     
with tab3:
    # "Full Schedule Analysis": the app's main scheduler. Every module runs the full
    # 6-phase cycle (Adsorption -> Evacuation -> NCG Purging -> Heating -> CO2 Purging
    # -> Cooling), split into Group A (odd module numbers) and Group B (even), with
    # shared capacity pools between groups. The tab runs three variants side by side
    # for comparison: Baseline (Group A alone, no sharing), Concurrent (a single
    # combined desorption+cooling cap), and Interleaved (separate Evacuation/Cooling
    # and NCG+Heating+CO2 shared pools, with Evacuation able to pause a pair's
    # Cooling). See run_simulation() below for the actual scheduling logic.
    # === PHASES ===
    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
    DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling'}

    # Backend constants (edit in code to change)
    OPT_MAX_DELAY = 120
    OPT_STEP = 1
    DELAY_METHOD = "Automatic"

    # === Input Configuration ===
    st.markdown("#### Settings for all configurations")
    set_col1, set_col2 = st.columns(2)
    with set_col1:
        target_time = st.number_input("Operating Period (Mins)", 0, 1440, 1440)
        generate_all_tab3 = st.button(
            "Generate all four configurations", key="generate_all_tab3",
            help="Runs 8-8, 4-4, 2-2 and 6-6 at once; each one's results appear on its own tab.",
        )
    with set_col2:
        enforce_evac_cool = st.checkbox("Evacuation Overrides Cooling", value=True)
        allow_cooling_pause = st.checkbox("Allow Cooling Pause During Evacuation", value=True)
        preload_group_b = st.checkbox(
            "Start Group B pre-loaded (begins in Evacuation, no startup gap)",
            value=True,
            help="Group B's very first cycle skips Adsorption and starts directly in "
                 "Evacuation, so it begins desorbing at the same instant Group A begins its "
                 "first Adsorption instead of idling while Group A holds the shared Adsorption "
                 "capacity. Every cycle after the first runs the normal full Adsorption→Cooling "
                 "sequence. Applies to all four configurations below (8-8, 4-4, 2-2, 6-6).",
        )

    st.caption(
        "Each tab below holds one configuration: its inputs and duration tables, "
        "its own Generate button, and its results, so there's nothing to scroll between."
    )
    cfg_tab_88, cfg_tab_44, cfg_tab_22, cfg_tab_66 = st.tabs(
        ["8-8 Configuration", "4-4 Configuration", "2-2 Configuration", "6-6 Configuration"]
    )
    with cfg_tab_88:
        st.caption("The main analysis: a 16-module plant split into two 8-module pairs.")
        in_col1, in_col2 = st.columns(2)
        with in_col1:
            total_modules = st.number_input("Total Modules", 2, 32, 16, step=2)
            adsorption_capacity = st.number_input(
                "Adsorption capacity",
                1, 32, 8,
                help="Max modules in Adsorption at once."
            )
        with in_col2:
            concurrent_desorption_cap = st.number_input(
                "Concurrent: Max modules in desorption",
                1, 32, 8,
                help="Max modules in desorption phases + Cooling at the same time."
            )
            shared_evac_cooling_cap = st.number_input(
                "Max modules: Evacuation / Cooling (Interleaved)",
                1, 32, 8,
                # help="One shared resource for both phases. Max modules in Evacuation or Cooling combined at once."
                help ="One shared resource for both phases. Max modules in Evacuation or Cooling combined at once."
            )
            shared_purge_heat_co2_cap = st.number_input(
                "Max modules: NCG + Heating + CO2 (Interleaved)",
                1, 32, 8,
                help="One shared resource for all three phases. Max modules in NCG Purging, Heating, or CO2 Purging combined at once. In Interleaved, only one group can be in these phases at a time; the other group can evacuate, then waits."
            )
        st.caption("Phase durations (minutes) — edit directly in the table")
        default_phase_durations_tab3 = load_stage_duration_defaults_tab3()
        st.caption(
            "Defaults for both pairs are seeded from the live stage-duration data using the "
            f"N3N4-M1n3M2n4 cycle average (now a real recorded 8-module combination): "
            f"{default_phase_durations_tab3}. Edit per pair if a pair's real durations differ."
        )
        PHASE_DURATIONS_TAB3_VERSION = 5  # bump whenever load_stage_duration_defaults_tab3()'s sourcing changes
        get_versioned_default_table(
            "phase_durations_tab3", ["Phase", "Group A (min)", "Group B (min)"], PHASE_DURATIONS_TAB3_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": [default_phase_durations_tab3[phase] for phase in PHASES],
                "Group B (min)": [default_phase_durations_tab3[phase] for phase in PHASES],
            }),
        )
        phase_edited = st.data_editor(
            st.session_state.phase_durations_tab3,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, default=25, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, default=25, required=True),
            },
            key="phase_durations_editor_tab3",
        )
        st.session_state.phase_durations_tab3 = phase_edited
        grp_a = {phase: int(phase_edited.loc[phase_edited["Phase"] == phase, "Group A (min)"].iloc[0]) for phase in PHASES}
        grp_b = {phase: int(phase_edited.loc[phase_edited["Phase"] == phase, "Group B (min)"].iloc[0]) for phase in PHASES}
        PHASE_DURATIONS = grp_a
        PHASE_DURATIONS_BY_GROUP = {"A": grp_a, "B": grp_b}

        # --- Fixed Constants ---
        TOTAL_MINUTES = target_time # 24 hours * 60 minutes
        TOTAL_MODULES = int(total_modules)
        MODULES = list(range(1, TOTAL_MODULES + 1))

        GROUP_IDS = ["A", "B"]
        GROUP_OF = {m: ("A" if m % 2 == 1 else "B") for m in MODULES}

        default_energy_tab3, default_yield_tab3 = load_plant_cycle_defaults_tab3()
        st.caption("Energy used & plant yield per pair, per completed cycle — edit directly in the table")
        st.caption(
            f"Defaults for both pairs are seeded from the live plant-cycle data using the "
            f"N3N4-M1n3M2n4 cycle average (now a real recorded 8-module combination, replacing "
            f"the previous N1N2N3-M1n3 + N3-M2n4 estimate): {default_energy_tab3} kWh/cycle, "
            f"{default_yield_tab3} kg CO2/cycle. Edit per pair if a pair's real output differs."
        )
        pair_labels_tab3 = [f"Group {gid}" for gid in GROUP_IDS]
        ENERGY_YIELD_TAB3_DEFAULTS_VERSION = 2  # bump whenever load_plant_cycle_defaults_tab3()'s sourcing changes

        energy_yield_tab3_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab3", energy_yield_tab3_cols, ENERGY_YIELD_TAB3_DEFAULTS_VERSION,
            lambda: pd.DataFrame({
                "Pair": pair_labels_tab3,
                "Energy per Cycle (kWh)": [default_energy_tab3] * len(pair_labels_tab3),
                "Yield per Cycle (kg CO2)": [default_yield_tab3] * len(pair_labels_tab3),
            }),
        )
        energy_yield_tab3 = st.data_editor(
            st.session_state.energy_yield_tab3,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_tab3",
        )
        st.session_state.energy_yield_tab3 = energy_yield_tab3
        PAIR_ENERGY_PER_CYCLE_TAB3 = {
            gid: float(energy_yield_tab3.loc[energy_yield_tab3["Pair"] == f"Group {gid}", "Energy per Cycle (kWh)"].iloc[0])
            for gid in GROUP_IDS
        }
        PAIR_YIELD_PER_CYCLE_TAB3 = {
            gid: float(energy_yield_tab3.loc[energy_yield_tab3["Pair"] == f"Group {gid}", "Yield per Cycle (kg CO2)"].iloc[0])
            for gid in GROUP_IDS
        }

        def adjusted_duration(phase, base_duration):
            """Pass-through hook for per-phase duration tweaks; currently just returns the base duration unchanged."""
            return int(base_duration)

        # Idle time required between specific phase transitions (in minutes)
        PHASE_GAPS = {
            ('NCG Purging', 'Heating'): 0,
            ('Heating', 'CO2 Purging'): 0
        }

        # --- Resource Limits: Adsorption from input; others for display (shared caps for bottleneck report) ---
        adsorption_cap = min(int(adsorption_capacity), TOTAL_MODULES)
        RESOURCE_LIMITS = {
            "Adsorption": adsorption_cap,
            "Evacuation": shared_evac_cooling_cap,
            "NCG Purging": shared_purge_heat_co2_cap,
            "Heating": shared_purge_heat_co2_cap,
            "CO2 Purging": shared_purge_heat_co2_cap,
            "Cooling": shared_evac_cooling_cap,
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        results_slot_88 = st.container()

    # === "4-4" Configuration: a second, separate analysis ===
    # An 8-module plant split into two 4-module pairs (vs. the 16-module 8-8 split
    # above), run through the same Concurrent/Interleaved engine for side-by-side
    # comparison. Group A = N3&N4's M1&M3, Group B = N3&N4's M2&M4. Both are now
    # directly recorded 4-module combinations in their own right (N3N4-M1n3 for
    # Group A, N3N4-M2n4 for Group B — Nelion 4 brought both position types
    # online), so each group's Duration, Energy AND Yield all come straight from
    # its own exact real average -- no more proxying Group B off Group A's
    # number, and no more type-matched sum for Yield either, now that there's a
    # real combined reading to use directly. Shares the Operating Period and
    # Evacuation/Cooling behavior checkboxes with the 8-8 config above, since
    # those are plant-wide settings.
    with cfg_tab_44:
        st.caption(
            "A second, smaller full-schedule analysis: an 8-module plant split into two "
            "4-module pairs — Group A = N3 & N4's M1/M3, Group B = N3 & N4's M2/M4 — "
            "run through the same Concurrent/Interleaved engine for side-by-side "
            "comparison. Both pairs are now directly recorded combinations in their own "
            "right (N3N4-M1n3 for Group A, N3N4-M2n4 for Group B), so each group's "
            "Duration, Energy and Yield all come straight from its own exact real "
            "average — no more borrowing Group A's number for Group B, and no more "
            "type-matched sum for Yield either."
        )
        TAB3_44_TOTAL_MODULES = 8
        MODULES_44 = list(range(1, TAB3_44_TOTAL_MODULES + 1))
        GROUP_OF_44 = {m: ("A" if m % 2 == 1 else "B") for m in MODULES_44}
        TAB3_44_MODULE_MAP = {"A": "N3N4-M1n3", "B": "N3N4-M2n4"}
        # Only cycles/stage-duration rows logged on or after this date feed the 4-4
        # config's Duration/Energy/Yield defaults, to reflect a process change made
        # on the plant around then rather than being diluted by older behavior.
        TAB3_44_DATA_SINCE = pd.Timestamp("2026-11-18")
        TAB3_44_YIELD_FORMULA = {"A": [("N3N4-M1n3", 1)], "B": [("N3N4-M2n4", 1)]}
        # Both groups' real physical modules now match their Duration/Energy/Yield
        # source exactly (Group A = N3N4-M1n3's own decode, Group B = N3N4-M2n4's
        # own decode) -- no represents_map needed below, same as the 2-2 config.
        PAIR_PHYSICAL_MODULES_TAB3_44 = {
            "A": ["N3-M1", "N3-M3", "N4-M1", "N4-M3"],
            "B": ["N3-M2", "N3-M4", "N4-M2", "N4-M4"],
        }
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of this configuration's 8 real modules are grouped together for desorption.")
        st.dataframe(physical_pairing_table(["A", "B"], PAIR_PHYSICAL_MODULES_TAB3_44), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(
                ["A", "B"], TAB3_44_MODULE_MAP, TAB3_44_MODULE_MAP, TAB3_44_YIELD_FORMULA,
            ),
            use_container_width=True, hide_index=True,
        )

        cap44_col1, cap44_col2 = st.columns(2)
        with cap44_col1:
            adsorption_capacity_44 = st.number_input(
                "Adsorption capacity (4-4)", 1, TAB3_44_TOTAL_MODULES, 4,
                help="Max modules in Adsorption at once.", key="adsorption_capacity_44",
            )
        with cap44_col2:
            concurrent_desorption_cap_44 = st.number_input(
                "Concurrent: Max modules in desorption (4-4)", 1, TAB3_44_TOTAL_MODULES, 4,
                help="Max modules in desorption phases + Cooling at the same time.", key="concurrent_desorption_cap_44",
            )
            shared_evac_cooling_cap_44 = st.number_input(
                "Max modules: Evacuation / Cooling (4-4, Interleaved)", 1, TAB3_44_TOTAL_MODULES, 4,
                help="One shared resource for both phases. Max modules in Evacuation or Cooling combined at once.",
                key="shared_evac_cooling_cap_44",
            )
            shared_purge_heat_co2_cap_44 = st.number_input(
                "Max modules: NCG + Heating + CO2 (4-4, Interleaved)", 1, TAB3_44_TOTAL_MODULES, 4,
                help="One shared resource for all three phases. Max modules in NCG Purging, Heating, or CO2 Purging combined at once. In Interleaved, only one group can be in these phases at a time; the other group can evacuate, then waits.",
                key="shared_purge_heat_co2_cap_44",
            )

        default_phase_durations_44 = load_stage_duration_defaults_by_pair(
            ["A", "B"], TAB3_44_MODULE_MAP, since=TAB3_44_DATA_SINCE,
        )
        st.caption(
            f"Phase durations are seeded from each group's own real average -- Group A from "
            f"N3N4-M1n3, Group B from N3N4-M2n4 -- rows logged on or after "
            f"{TAB3_44_DATA_SINCE.date()} only."
        )
        PHASE_DURATIONS_TAB3_44_VERSION = 3
        get_versioned_default_table(
            "phase_durations_tab3_44", ["Phase", "Group A (min)", "Group B (min)"], PHASE_DURATIONS_TAB3_44_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": [default_phase_durations_44.get("A", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
                "Group B (min)": [default_phase_durations_44.get("B", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
            }),
        )
        phase_edited_44 = st.data_editor(
            st.session_state.phase_durations_tab3_44,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
            },
            key="phase_durations_editor_tab3_44",
        )
        st.session_state.phase_durations_tab3_44 = phase_edited_44
        grp_a_44 = {phase: int(phase_edited_44.loc[phase_edited_44["Phase"] == phase, "Group A (min)"].iloc[0]) for phase in PHASES}
        grp_b_44 = {phase: int(phase_edited_44.loc[phase_edited_44["Phase"] == phase, "Group B (min)"].iloc[0]) for phase in PHASES}
        PHASE_DURATIONS_44 = grp_a_44
        PHASE_DURATIONS_BY_GROUP_44 = {"A": grp_a_44, "B": grp_b_44}

        default_energy_yield_44 = load_plant_cycle_defaults_weighted(
            ["A", "B"], TAB3_44_MODULE_MAP, TAB3_44_YIELD_FORMULA, since=TAB3_44_DATA_SINCE,
        )
        st.caption(
            f"Both Energy and Yield now come from each group's own exact real average -- "
            f"Group A from N3N4-M1n3, Group B from N3N4-M2n4: {default_energy_yield_44}. "
            f"Only cycles on or after {TAB3_44_DATA_SINCE.date()} are averaged. Edit per pair "
            f"if a pair's real output differs."
        )
        ENERGY_YIELD_TAB3_44_VERSION = 4
        get_versioned_default_table(
            "energy_yield_tab3_44", ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"], ENERGY_YIELD_TAB3_44_VERSION,
            lambda: pd.DataFrame({
                "Pair": ["Group A", "Group B"],
                "Energy per Cycle (kWh)": [default_energy_yield_44["A"][0], default_energy_yield_44["B"][0]],
                "Yield per Cycle (kg CO2)": [default_energy_yield_44["A"][1], default_energy_yield_44["B"][1]],
            }),
        )
        energy_yield_tab3_44 = st.data_editor(
            st.session_state.energy_yield_tab3_44,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_tab3_44",
        )
        st.session_state.energy_yield_tab3_44 = energy_yield_tab3_44
        PAIR_ENERGY_PER_CYCLE_TAB3_44 = {
            gid: float(energy_yield_tab3_44.loc[energy_yield_tab3_44["Pair"] == f"Group {gid}", "Energy per Cycle (kWh)"].iloc[0])
            for gid in ["A", "B"]
        }
        PAIR_YIELD_PER_CYCLE_TAB3_44 = {
            gid: float(energy_yield_tab3_44.loc[energy_yield_tab3_44["Pair"] == f"Group {gid}", "Yield per Cycle (kg CO2)"].iloc[0])
            for gid in ["A", "B"]
        }
        RESOURCE_LIMITS_44 = {
            "Adsorption": min(int(adsorption_capacity_44), TAB3_44_TOTAL_MODULES),
            "Evacuation": shared_evac_cooling_cap_44,
            "NCG Purging": shared_purge_heat_co2_cap_44,
            "Heating": shared_purge_heat_co2_cap_44,
            "CO2 Purging": shared_purge_heat_co2_cap_44,
            "Cooling": shared_evac_cooling_cap_44,
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        results_slot_44 = st.container()

    # === "2-2" Configuration: a third, separate analysis ===
    # A 4-module plant split into two 2-module pairs — one size smaller than the
    # 4-4 config above. N3-M2n4 IS an actual 2-module combination, so unlike 4-4
    # (which had to borrow a differently-sized combination and double the Yield),
    # here Duration, Energy per Cycle, AND Yield per Cycle all come directly from
    # N3-M2n4's own real average, with no scaling, for both pairs. Shares the
    # Operating Period and Evacuation/Cooling behavior checkboxes with the 8-8
    # config above, since those are plant-wide settings.
    with cfg_tab_22:
        st.caption(
            "A third, even smaller full-schedule analysis: a 4-module plant split into "
            "two 2-module pairs, run through the same Concurrent/Interleaved engine for "
            "side-by-side comparison. N3-M2n4 is an actual 2-module combination, so "
            "Duration, Energy per Cycle, and Yield per Cycle all come directly from its "
            "own real average — no scaling — for both pairs."
        )
        TAB3_22_TOTAL_MODULES = 4
        MODULES_22 = list(range(1, TAB3_22_TOTAL_MODULES + 1))
        GROUP_OF_22 = {m: ("A" if m % 2 == 1 else "B") for m in MODULES_22}
        TAB3_22_MODULE_MAP = {"A": "N3-M2n4", "B": "N3-M2n4"}
        TAB3_22_YIELD_FORMULA = {"A": [("N3-M2n4", 1)], "B": [("N3-M2n4", 1)]}

        cap22_col1, cap22_col2 = st.columns(2)
        with cap22_col1:
            adsorption_capacity_22 = st.number_input(
                "Adsorption capacity (2-2)", 1, TAB3_22_TOTAL_MODULES, 2,
                help="Max modules in Adsorption at once.", key="adsorption_capacity_22",
            )
        with cap22_col2:
            concurrent_desorption_cap_22 = st.number_input(
                "Concurrent: Max modules in desorption (2-2)", 1, TAB3_22_TOTAL_MODULES, 2,
                help="Max modules in desorption phases + Cooling at the same time.", key="concurrent_desorption_cap_22",
            )
            shared_evac_cooling_cap_22 = st.number_input(
                "Max modules: Evacuation / Cooling (2-2, Interleaved)", 1, TAB3_22_TOTAL_MODULES, 2,
                help="One shared resource for both phases. Max modules in Evacuation or Cooling combined at once.",
                key="shared_evac_cooling_cap_22",
            )
            shared_purge_heat_co2_cap_22 = st.number_input(
                "Max modules: NCG + Heating + CO2 (2-2, Interleaved)", 1, TAB3_22_TOTAL_MODULES, 2,
                help="One shared resource for all three phases. Max modules in NCG Purging, Heating, or CO2 Purging combined at once. In Interleaved, only one group can be in these phases at a time; the other group can evacuate, then waits.",
                key="shared_purge_heat_co2_cap_22",
            )

        default_phase_durations_22 = load_stage_duration_defaults_by_pair(["A", "B"], TAB3_22_MODULE_MAP)
        st.caption("Phase durations for both pairs are seeded from the live stage-duration data using the N3-M2n4 cycle average.")
        PHASE_DURATIONS_TAB3_22_VERSION = 1
        get_versioned_default_table(
            "phase_durations_tab3_22", ["Phase", "Group A (min)", "Group B (min)"], PHASE_DURATIONS_TAB3_22_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": [default_phase_durations_22.get("A", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
                "Group B (min)": [default_phase_durations_22.get("B", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
            }),
        )
        phase_edited_22 = st.data_editor(
            st.session_state.phase_durations_tab3_22,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
            },
            key="phase_durations_editor_tab3_22",
        )
        st.session_state.phase_durations_tab3_22 = phase_edited_22
        grp_a_22 = {phase: int(phase_edited_22.loc[phase_edited_22["Phase"] == phase, "Group A (min)"].iloc[0]) for phase in PHASES}
        grp_b_22 = {phase: int(phase_edited_22.loc[phase_edited_22["Phase"] == phase, "Group B (min)"].iloc[0]) for phase in PHASES}
        PHASE_DURATIONS_22 = grp_a_22
        PHASE_DURATIONS_BY_GROUP_22 = {"A": grp_a_22, "B": grp_b_22}

        default_energy_yield_22 = load_plant_cycle_defaults_weighted(["A", "B"], TAB3_22_MODULE_MAP, TAB3_22_YIELD_FORMULA)
        st.caption(
            f"Energy per Cycle and Yield per Cycle both use N3-M2n4's own real average: {default_energy_yield_22}. "
            f"Edit per pair if a pair's real output differs."
        )
        ENERGY_YIELD_TAB3_22_VERSION = 1
        get_versioned_default_table(
            "energy_yield_tab3_22", ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"], ENERGY_YIELD_TAB3_22_VERSION,
            lambda: pd.DataFrame({
                "Pair": ["Group A", "Group B"],
                "Energy per Cycle (kWh)": [default_energy_yield_22["A"][0], default_energy_yield_22["B"][0]],
                "Yield per Cycle (kg CO2)": [default_energy_yield_22["A"][1], default_energy_yield_22["B"][1]],
            }),
        )
        energy_yield_tab3_22 = st.data_editor(
            st.session_state.energy_yield_tab3_22,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_tab3_22",
        )
        st.session_state.energy_yield_tab3_22 = energy_yield_tab3_22
        PAIR_ENERGY_PER_CYCLE_TAB3_22 = {
            gid: float(energy_yield_tab3_22.loc[energy_yield_tab3_22["Pair"] == f"Group {gid}", "Energy per Cycle (kWh)"].iloc[0])
            for gid in ["A", "B"]
        }
        PAIR_YIELD_PER_CYCLE_TAB3_22 = {
            gid: float(energy_yield_tab3_22.loc[energy_yield_tab3_22["Pair"] == f"Group {gid}", "Yield per Cycle (kg CO2)"].iloc[0])
            for gid in ["A", "B"]
        }
        RESOURCE_LIMITS_22 = {
            "Adsorption": min(int(adsorption_capacity_22), TAB3_22_TOTAL_MODULES),
            "Evacuation": shared_evac_cooling_cap_22,
            "NCG Purging": shared_purge_heat_co2_cap_22,
            "Heating": shared_purge_heat_co2_cap_22,
            "CO2 Purging": shared_purge_heat_co2_cap_22,
            "Cooling": shared_evac_cooling_cap_22,
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        results_slot_22 = st.container()

    # === "6-6" Configuration: a fourth, separate analysis ===
    # A 12-module plant split into two 6-module pairs — Group A = N2&N3&N4's
    # M1&M3, Group B = N2&N3&N4's M2&M4 (no N1). Neither 6-module combination is
    # itself recorded, so each group sources from its own module TYPE: Group A
    # (M1n3-type) uses N3N4-M1n3 (Duration alone; Energy/Yield add N4-M1n3).
    # Group B (M2n4-type) has no real N3N4-M2n4 combo, so its Duration borrows
    # Group A's N3N4-M1n3 (same real 4-module N3+N4 timing — same precedent as
    # the 4-4 config's Group B), while its Energy/Yield stay type-matched:
    # N4-M2n4 DOUBLED plus N3-M2n4 (one real unit each, matching the 6-module
    # total: N4-M2n4 x2 + N3-M2n4 x1). Shares the Operating Period and
    # Evacuation/Cooling behavior checkboxes with the 8-8 config above, since
    # those are plant-wide settings.
    with cfg_tab_66:
        st.caption(
            "A fourth full-schedule analysis: a 12-module plant split into two "
            "6-module pairs — Group A = N2, N3 & N4's M1/M3, Group B = N2, N3 & N4's "
            "M2/M4 (no N1) — run through the same Concurrent/Interleaved engine for "
            "side-by-side comparison. Neither 6-module combination is itself "
            "recorded. Both groups' Duration comes from N3N4-M1n3's own real "
            "average (Group B borrows Group A's, since there's no real N3N4-M2n4 "
            "combo). Energy and Yield per Cycle instead stay type-matched: Group "
            "A sums N3N4-M1n3 + N4-M1n3, Group B sums N4-M2n4 DOUBLED + N3-M2n4."
        )
        TAB3_66_TOTAL_MODULES = 12
        MODULES_66 = list(range(1, TAB3_66_TOTAL_MODULES + 1))
        GROUP_OF_66 = {m: ("A" if m % 2 == 1 else "B") for m in MODULES_66}
        TAB3_66_DURATION_MODULE_MAP = {
            "A": "N3N4-M1n3",
            "B": "N3N4-M1n3",
        }
        # Energy and Yield per Cycle share the same type-matched formula per group.
        TAB3_66_ENERGY_YIELD_FORMULA = {
            "A": [("N3N4-M1n3", 1), ("N4-M1n3", 1)],
            "B": [("N4-M2n4", 2), ("N3-M2n4", 1)],
        }
        TAB3_66_YIELD_FORMULA = TAB3_66_ENERGY_YIELD_FORMULA
        # Neither group's real physical modules (on N2/N3/N4) match the sources
        # they're built from (no N2 modules in either; Group B's Duration borrows
        # Group A's M1n3-type source even though Group B is M2n4-type).
        REPRESENTS_TAB3_66 = {
            "A": "N2-M1, N2-M3, N3-M1, N3-M3, N4-M1, N4-M3 (sourced from N3N4-M1n3 + N4-M1n3)",
            "B": "N2-M2, N2-M4, N3-M2, N3-M4, N4-M2, N4-M4 (Duration mirrors Group A's N3N4-M1n3; Energy/Yield from N4-M2n4 x2 + N3-M2n4)",
        }
        PAIR_PHYSICAL_MODULES_TAB3_66 = {
            "A": ["N2-M1", "N2-M3", "N3-M1", "N3-M3", "N4-M1", "N4-M3"],
            "B": ["N2-M2", "N2-M4", "N3-M2", "N3-M4", "N4-M2", "N4-M4"],
        }
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of this configuration's 12 real modules are grouped together for desorption.")
        st.dataframe(physical_pairing_table(["A", "B"], PAIR_PHYSICAL_MODULES_TAB3_66), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(
                ["A", "B"], TAB3_66_DURATION_MODULE_MAP, TAB3_66_ENERGY_YIELD_FORMULA, TAB3_66_YIELD_FORMULA,
                represents_map=REPRESENTS_TAB3_66,
            ),
            use_container_width=True, hide_index=True,
        )

        cap66_col1, cap66_col2 = st.columns(2)
        with cap66_col1:
            adsorption_capacity_66 = st.number_input(
                "Adsorption capacity (6-6)", 1, TAB3_66_TOTAL_MODULES, 6,
                help="Max modules in Adsorption at once.", key="adsorption_capacity_66",
            )
        with cap66_col2:
            concurrent_desorption_cap_66 = st.number_input(
                "Concurrent: Max modules in desorption (6-6)", 1, TAB3_66_TOTAL_MODULES, 6,
                help="Max modules in desorption phases + Cooling at the same time.", key="concurrent_desorption_cap_66",
            )
            shared_evac_cooling_cap_66 = st.number_input(
                "Max modules: Evacuation / Cooling (6-6, Interleaved)", 1, TAB3_66_TOTAL_MODULES, 6,
                help="One shared resource for both phases. Max modules in Evacuation or Cooling combined at once.",
                key="shared_evac_cooling_cap_66",
            )
            shared_purge_heat_co2_cap_66 = st.number_input(
                "Max modules: NCG + Heating + CO2 (6-6, Interleaved)", 1, TAB3_66_TOTAL_MODULES, 6,
                help="One shared resource for all three phases. Max modules in NCG Purging, Heating, or CO2 Purging combined at once. In Interleaved, only one group can be in these phases at a time; the other group can evacuate, then waits.",
                key="shared_purge_heat_co2_cap_66",
            )

        default_phase_durations_66 = load_stage_duration_defaults_by_pair(["A", "B"], TAB3_66_DURATION_MODULE_MAP)
        st.caption("Phase durations for both pairs are seeded from the live stage-duration data using the N3N4-M1n3 cycle average (Group B borrows Group A's, since there's no real N3N4-M2n4 combo).")
        PHASE_DURATIONS_TAB3_66_VERSION = 1
        get_versioned_default_table(
            "phase_durations_tab3_66", ["Phase", "Group A (min)", "Group B (min)"], PHASE_DURATIONS_TAB3_66_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": [default_phase_durations_66.get("A", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
                "Group B (min)": [default_phase_durations_66.get("B", {}).get(phase, CORE_PHASE_FALLBACK_DURATIONS[phase]) for phase in PHASES],
            }),
        )
        adv_phase_table_66 = st.data_editor(
            st.session_state.phase_durations_tab3_66,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
            },
            key="phase_durations_editor_tab3_66",
        )
        st.session_state.phase_durations_tab3_66 = adv_phase_table_66
        grp_a_66 = {phase: int(adv_phase_table_66.loc[adv_phase_table_66["Phase"] == phase, "Group A (min)"].iloc[0]) for phase in PHASES}
        grp_b_66 = {phase: int(adv_phase_table_66.loc[adv_phase_table_66["Phase"] == phase, "Group B (min)"].iloc[0]) for phase in PHASES}
        PHASE_DURATIONS_66 = grp_a_66
        PHASE_DURATIONS_BY_GROUP_66 = {"A": grp_a_66, "B": grp_b_66}

        default_energy_yield_66 = load_plant_cycle_defaults_weighted(["A", "B"], TAB3_66_ENERGY_YIELD_FORMULA, TAB3_66_YIELD_FORMULA)
        st.caption(
            f"Group A's Energy and Yield sum N3N4-M1n3 + N4-M1n3's own real averages; Group B's sum "
            f"N4-M2n4 doubled + N3-M2n4's own real average: {default_energy_yield_66}. Edit per pair if "
            f"a pair's real output differs."
        )
        ENERGY_YIELD_TAB3_66_VERSION = 1
        get_versioned_default_table(
            "energy_yield_tab3_66", ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"], ENERGY_YIELD_TAB3_66_VERSION,
            lambda: pd.DataFrame({
                "Pair": ["Group A", "Group B"],
                "Energy per Cycle (kWh)": [default_energy_yield_66["A"][0], default_energy_yield_66["B"][0]],
                "Yield per Cycle (kg CO2)": [default_energy_yield_66["A"][1], default_energy_yield_66["B"][1]],
            }),
        )
        energy_yield_table_66 = st.data_editor(
            st.session_state.energy_yield_tab3_66,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_tab3_66",
        )
        st.session_state.energy_yield_tab3_66 = energy_yield_table_66
        PAIR_ENERGY_PER_CYCLE_TAB3_66 = {
            gid: float(energy_yield_table_66.loc[energy_yield_table_66["Pair"] == f"Group {gid}", "Energy per Cycle (kWh)"].iloc[0])
            for gid in ["A", "B"]
        }
        PAIR_YIELD_PER_CYCLE_TAB3_66 = {
            gid: float(energy_yield_table_66.loc[energy_yield_table_66["Pair"] == f"Group {gid}", "Yield per Cycle (kg CO2)"].iloc[0])
            for gid in ["A", "B"]
        }
        RESOURCE_LIMITS_66 = {
            "Adsorption": min(int(adsorption_capacity_66), TAB3_66_TOTAL_MODULES),
            "Evacuation": shared_evac_cooling_cap_66,
            "NCG Purging": shared_purge_heat_co2_cap_66,
            "Heating": shared_purge_heat_co2_cap_66,
            "CO2 Purging": shared_purge_heat_co2_cap_66,
            "Cooling": shared_evac_cooling_cap_66,
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        results_slot_66 = st.container()

    selected_delay = 0 # Default value, will be updated based on user choice

    def build_module_delays(offset, group_ids, group_of, modules):
        """Stagger each group's start time by `offset` minutes (Group A starts at 0,
        Group B at 1x offset, etc.) so their phases interleave instead of colliding."""
        group_index = {g: i for i, g in enumerate(group_ids)}
        return {m: group_index[group_of[m]] * offset for m in modules}



    # --- Core Simulation Function ---
    def run_simulation(module_delays_config, phase_durations_config, resource_limits_config, current_desorption_mode, group_ids, group_of, phase_durations_by_group=None, concurrent_desorption_cap=1, shared_evac_cooling_cap=1, shared_purge_heat_co2_cap=1, modules_to_run=None, baseline_mode=False, preloaded_groups=None):
        """Runs scheduling simulation; returns schedule, resource_usage, evac/cool stats.
        modules_to_run: optional list of module IDs (default: all MODULES).
        baseline_mode: if True, no shared resource constraints—each phase has its own capacity.
        preloaded_groups: optional set of group ids (e.g. {"B"}) whose modules are treated as
        already charged with CO2 at t=0 — their very first cycle skips Adsorption and starts
        directly at Evacuation, so that group doesn't idle waiting on shared Adsorption capacity
        while another group's first Adsorption is in progress. Every cycle after the first runs
        the full Adsorption→Cooling sequence as normal."""
        modules = modules_to_run if modules_to_run is not None else list(MODULES)
        # Arrays/fit checks run past the operating period so a cycle can start
        # before the end of the day and finish after it.
        horizon = 2 * TOTAL_MINUTES + 1
        preloaded_groups = preloaded_groups or set()
        module_first_cycle_done = {mod: False for mod in modules}
        resource_usage = {phase: np.zeros(horizon, dtype=int) for phase in PHASES}
        module_timers = {mod: 0 for mod in modules}
        evac_cool_stats = {
            "cooling_delay_events": 0,
            "cooling_delay_minutes": 0,
            "evac_delay_events": 0,
            "evac_delay_minutes": 0
        }

        for mod in modules:
            module_timers[mod] = module_delays_config.get(mod, 0)
        
        schedule = []
        # When per-group durations are off, fix Adsorption to same value for all modules
        adsorption_dur_uniform = int(phase_durations_config["Adsorption"]) if not phase_durations_by_group else None

        def can_allocate_internal(phase, start, duration, current_resource_state):
            """True if this phase has spare shared-resource capacity for the whole
            [start, start+duration) window, under the given desorption mode's rules
            (Adsorption's own cap, Baseline's per-phase caps, Interleaved's shared
            Evacuation/Cooling and NCG+Heating+CO2 pools, or Concurrent's single
            desorption-wide cap)."""
            end_time = start + duration
            if end_time > horizon or start < 0:
                return False
            end_safe = min(end_time, horizon)  # clamp for array bounds (indices 0..horizon-1)
            # Adsorption: always per-phase capacity from table
            if phase == "Adsorption":
                return bool(np.all(current_resource_state[phase][start:end_safe] < resource_limits_config[phase]))
            # Baseline mode: no sharing—each phase has its own capacity
            if baseline_mode and current_desorption_mode == "Interleaved":
                if phase in ("Evacuation", "Cooling"):
                    return bool(np.all(current_resource_state[phase][start:end_safe] < shared_evac_cooling_cap))
                if phase in ("NCG Purging", "Heating", "CO2 Purging"):
                    return bool(np.all(current_resource_state[phase][start:end_safe] < shared_purge_heat_co2_cap))
                return True
            # Interleaved: shared resource groups
            if current_desorption_mode == "Interleaved":
                  # Evacuation and Cooling are now independent.
                  # They can run at the same time, as long as each phase stays
                 # within its own allowed capacity.
                if phase == "Evacuation":
                   return bool(np.all(current_resource_state["Evacuation"][start:end_safe] < shared_evac_cooling_cap))

                    # if phase == "Cooling":
                    # return bool(np.all(current_resource_state["Cooling"][start:end_safe] < shared_evac_cooling_cap))

                if phase in ("NCG Purging", "Heating", "CO2 Purging"):
                    purge_heat_co2_phases = ("NCG Purging", "Heating", "CO2 Purging")
                    combined = sum(current_resource_state[p][start:end_safe] for p in purge_heat_co2_phases)
                    return bool(np.all(combined < shared_purge_heat_co2_cap))
            # Concurrent: per-phase caps ignored for desorption phases; concurrent_desorption_cap gates Evacuation start
            if current_desorption_mode == "Concurrent" and phase == "Evacuation":
                desorption_phases_plus_cooling = list(DESORPTION_PHASES) + ["Cooling"]
                total_in_desorption = sum(current_resource_state[p][start:end_safe] for p in desorption_phases_plus_cooling)
                if np.any(total_in_desorption >= concurrent_desorption_cap):
                    return False
            return True

        def reserve_internal(phase, start, duration, current_resource_state):
            """Occupy one capacity slot of this phase for every minute in [start, start+duration)."""
            end_time = start + duration
            end_safe = min(end_time, horizon)
            current_resource_state[phase][start:end_safe] += 1

        def evac_cool_conflict_internal(phase, start, duration, current_resource_state, mod):
            """True only for a Cooling attempt that overlaps another module's Evacuation
            (Interleaved mode, with the 'Evacuation Overrides Cooling' checkbox on).
            The caller uses this to make Cooling pause rather than block outright."""
            if baseline_mode or current_desorption_mode != "Interleaved":
                return False

             
            if not enforce_evac_cool:
                return False   
            # end_time = start + duration
            # end_safe = min(end_time, horizon)
            # enforce_evac_cool: Evacuation overrides Cooling (Cooling pauses). Only Cooling waits.
            # When off: No interruption—both block each other; whoever started first runs to completion.
            if phase != "Cooling":
                return False
             
            end_time = start + duration
            end_safe = min(end_time, horizon)

            this_group = group_of[mod]
            other_group = "B" if this_group == "A" else "A"
            
            return np.any(current_resource_state["Evacuation"][start:end_safe] > 0)
            # if phase == "Evacuation":
            #     return False if enforce_evac_cool else np.any(current_resource_state["Cooling"][start:end_safe] > 0)
            # return False
        
        # Event-driven scheduling with no group priority: always advance whichever
        # module is ready earliest, one phase at a time, so every shared resource is
        # first-come-first-served in real time. NCG Purging -> Heating -> CO2 Purging
        # is booked as one unbroken block, and (Interleaved) only one group may hold
        # that block at a time; the other group can still evacuate, then waits.
        # A phase may start any time before the end of the operating period and run
        # past it (the plant keeps going into the next day); only cycles that finish
        # inside the period count as complete.
        CHAIN = ["NCG Purging", "Heating", "CO2 Purging"]
        chain_by_group = {g: np.zeros(horizon, dtype=int) for g in {group_of[m] for m in modules}}
        group_exclusive_chain = current_desorption_mode == "Interleaved" and not baseline_mode

        module_cycles = {mod: [] for mod in modules}  # each cycle: {"phases": [[phase, start, end], ...], "done", "full"}
        module_phase_idx = {mod: 0 for mod in modules}
        module_last_chain_end = {mod: -1 for mod in modules}
        open_cooling = {}  # mod -> (cooling start, duration) while a later Evacuation can still pause it
        active = set(modules)

        def phase_duration(group_id, phase):
            if phase == "Adsorption" and adsorption_dur_uniform is not None:
                return adsorption_dur_uniform
            base_dur = phase_durations_by_group[group_id][phase] if phase_durations_by_group else phase_durations_config[phase]
            return adjusted_duration(phase, base_dur)

        def gap_after(phase):
            nxt = PHASES.index(phase) + 1
            return PHASE_GAPS.get((phase, PHASES[nxt]), 0) if nxt < len(PHASES) else 0

        def chain_layout(start, group_id):
            layout, t0 = [], start
            for p in CHAIN:
                d = phase_duration(group_id, p)
                layout.append((p, t0, t0 + d))
                t0 += d + gap_after(p)
            return layout

        def chain_fits(layout, group_id):
            for p, s, e in layout:
                if not can_allocate_internal(p, s, e - s, resource_usage):
                    return False
                if group_exclusive_chain and any(
                    np.any(usage[s:e] > 0) for g, usage in chain_by_group.items() if g != group_id
                ):
                    return False
            return True

        def cooling_segments_from(start, duration):
            """Split a Cooling run around any Evacuation already booked during it, so
            Cooling stops for the Evacuation and resumes for its remaining time."""
            if not allow_cooling_pause:
                return [(start, start + duration)]
            evac_busy = resource_usage["Evacuation"]
            segments, remaining, cur = [], duration, start
            while remaining > 0:
                window_end = min(horizon, cur + remaining)
                busy = np.flatnonzero(evac_busy[cur:window_end] > 0) if cur < window_end else []
                if len(busy) == 0:
                    segments.append((cur, cur + remaining))
                    remaining = 0
                else:
                    next_conflict = cur + int(busy[0])
                    if next_conflict > cur:
                        segments.append((cur, next_conflict))
                        remaining -= next_conflict - cur
                    cur = next_conflict
                    while cur < horizon and evac_busy[cur] > 0:
                        cur += 1
            return segments

        def pause_overlapping_cooling(evac_mod, s, e):
            """A newly booked Evacuation overrides any other module's Cooling that is
            still running across it: re-split that Cooling and push its end later.
            Safe because that module's next phase hasn't been scheduled yet (its
            Cooling ends after `s`, and modules are advanced in time order)."""
            for other, (c_start, c_dur) in list(open_cooling.items()):
                if other == evac_mod:
                    continue
                phases = module_cycles[other][-1]["phases"]
                old = [seg for seg in phases if seg[0] == "Cooling"]
                if not any(seg[1] < e and seg[2] > s for seg in old):
                    continue
                old_end = old[-1][2]
                for seg in old:
                    resource_usage["Cooling"][seg[1]:min(seg[2], horizon)] -= 1
                    phases.remove(seg)
                new_segments = cooling_segments_from(c_start, c_dur)
                for ns, ne in new_segments:
                    resource_usage["Cooling"][ns:min(ne, horizon)] += 1
                    phases.append(["Cooling", ns, ne])
                evac_cool_stats["cooling_delay_events"] += 1
                evac_cool_stats["cooling_delay_minutes"] += new_segments[-1][1] - old_end
                module_timers[other] = new_segments[-1][1]

        while active:
            # Earliest-ready module first; on a tie, whichever last held the
            # NCG/Heating/CO2 block longest ago, so neither group is favored.
            mod = min(active, key=lambda m: (module_timers[m], module_last_chain_end[m], m))
            open_cooling.pop(mod, None)
            t = module_timers[mod]
            if t >= TOTAL_MINUTES:
                active.discard(mod)
                continue
            group_id = group_of[mod]
            phases_this_cycle = (
                PHASES[1:] if group_id in preloaded_groups and not module_first_cycle_done[mod] else PHASES
            )
            idx = module_phase_idx[mod]
            phase = phases_this_cycle[idx]

            if phase == CHAIN[0]:
                layout, start = None, t
                while start < TOTAL_MINUTES:
                    candidate = chain_layout(start, group_id)
                    if candidate[-1][2] > horizon:
                        break
                    if chain_fits(candidate, group_id):
                        layout = candidate
                        break
                    start += 1
                if layout is None:
                    active.discard(mod)
                    continue
                booked = [[p, s, e] for p, s, e in layout]
                for p, s, e in layout:
                    reserve_internal(p, s, e - s, resource_usage)
                    chain_by_group[group_id][s:min(e, horizon)] += 1
                module_last_chain_end[mod] = layout[-1][2]
                next_timer = layout[-1][2] + gap_after(CHAIN[-1])
                steps = len(CHAIN)

            elif phase == "Cooling":
                duration = phase_duration(group_id, "Cooling")
                attempt_start, conflict_hit, placed = t, False, False
                while attempt_start < TOTAL_MINUTES and attempt_start + duration <= horizon:
                    if evac_cool_conflict_internal("Cooling", attempt_start, duration, resource_usage, mod):
                        # Cooling can't start during another module's Evacuation; jump
                        # ahead by the other group's Evacuation duration and retry.
                        other_group = "B" if group_id == "A" else "A"
                        pause_minutes = int(
                            phase_durations_by_group[other_group]["Evacuation"] if phase_durations_by_group
                            else phase_durations_config["Evacuation"]
                        )
                        if not conflict_hit:
                            evac_cool_stats["cooling_delay_events"] += 1
                            conflict_hit = True
                        evac_cool_stats["cooling_delay_minutes"] += pause_minutes
                        attempt_start += pause_minutes
                        continue
                    if can_allocate_internal("Cooling", attempt_start, duration, resource_usage):
                        placed = True
                        break
                    attempt_start += 1
                if not placed:
                    active.discard(mod)
                    continue
                segments = cooling_segments_from(attempt_start, duration)
                booked = [["Cooling", s, e] for s, e in segments]
                for s, e in segments:
                    reserve_internal("Cooling", s, e - s, resource_usage)
                open_cooling[mod] = (attempt_start, duration)
                next_timer = segments[-1][1]
                steps = 1

            else:
                duration = phase_duration(group_id, phase)
                attempt_start = t
                while (attempt_start < TOTAL_MINUTES and attempt_start + duration <= horizon
                       and not can_allocate_internal(phase, attempt_start, duration, resource_usage)):
                    attempt_start += 1
                if attempt_start >= TOTAL_MINUTES or attempt_start + duration > horizon:
                    active.discard(mod)
                    continue
                reserve_internal(phase, attempt_start, duration, resource_usage)
                booked = [[phase, attempt_start, attempt_start + duration]]
                next_timer = attempt_start + duration + gap_after(phase)
                steps = 1

            if idx == 0:
                module_cycles[mod].append({"phases": [], "done": False, "full": len(phases_this_cycle) == len(PHASES)})
            module_cycles[mod][-1]["phases"].extend(booked)
            module_timers[mod] = next_timer
            module_phase_idx[mod] = idx + steps
            if phase == "Evacuation" and allow_cooling_pause:
                pause_overlapping_cooling(mod, booked[0][1], booked[0][2])

            if module_phase_idx[mod] >= len(phases_this_cycle):
                module_cycles[mod][-1]["done"] = True
                module_phase_idx[mod] = 0
                module_first_cycle_done[mod] = True

        # Everything that started inside the operating period is shown (clipped at
        # its end); "Complete" marks the full cycles that also finished inside it,
        # which are the only ones counted toward cycles, yield and energy.
        schedule = []
        for mod in modules:
            for cycle_no, c in enumerate(module_cycles[mod]):
                complete = c["done"] and c["full"] and all(end <= TOTAL_MINUTES for _, _, end in c["phases"])
                for phase, start, end in c["phases"]:
                    if start < TOTAL_MINUTES:
                        schedule.append({
                            "Module": mod, "Phase": phase, "Start": start, "End": min(end, TOTAL_MINUTES),
                            "Cycle": cycle_no, "Complete": complete,
                        })

        if not schedule:
            empty = pd.DataFrame(columns=["Module", "Phase", "Start", "End", "Cycle", "Complete"])
            return empty, resource_usage, evac_cool_stats
        df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start']).reset_index(drop=True)
        return df_schedule, resource_usage, evac_cool_stats


    # --- Main Run Button ---
    def run_and_display_config(
        title, key_suffix, comparison_suffix,
        MODULES, GROUP_IDS, GROUP_OF,
        PHASE_DURATIONS, PHASE_DURATIONS_BY_GROUP, RESOURCE_LIMITS,
        PAIR_ENERGY_PER_CYCLE, PAIR_YIELD_PER_CYCLE,
        concurrent_desorption_cap, shared_evac_cooling_cap, shared_purge_heat_co2_cap,
    ):
        """Runs the optimizer + both simulations for one Full Schedule Analysis
        configuration (its own module count/grouping/durations/rates/capacities)
        and renders its full results section. Called once per configuration so
        the 8-8 and 4-4 setups can be compared side by side from one button."""
        st.markdown(f"### Results: {title}")
        final_delay_to_use = 0
        optimization_summary_df = None

        def count_complete_cycles_df(df_schedule):
            """Count each module's complete cycles (full Adsorption→Cooling cycles that
            finished inside the operating period, as flagged by run_simulation); returns
            one row per module with a "M<n>"-formatted label."""
            cycle_counts = []
            for mod in MODULES:
                mod_df = df_schedule[(df_schedule["Module"] == mod) & df_schedule["Complete"]]
                cycle_counts.append({"Module": mod, "Complete Cycles": int(mod_df["Cycle"].nunique())})
            df_cycles = pd.DataFrame(cycle_counts)
            df_cycles["Module"] = df_cycles["Module"].apply(lambda m: f"M{m}")
            return df_cycles

        if DELAY_METHOD == 'Automatic':
            st.info("Running optimization to find the best delay (both modes)...")

            def optimize_offsets(group_ids, group_of):
                """Brute-force search over Group B start offsets (0..OPT_MAX_DELAY minutes)
                to find the one that yields the most total Interleaved-mode cycles; also
                records the Concurrent-mode cycle count at each offset for comparison."""
                best_delay = 0
                best_cycles_int = -1
                delay_search_range = range(0, OPT_MAX_DELAY + 1, OPT_STEP)
                optimization_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                opt_preloaded_groups = {"B"} if preload_group_b else None
                for i, delay in enumerate(delay_search_range):
                    current_module_delays_for_opt = build_module_delays(delay, group_ids, group_of, MODULES)
                    schedule_int, _, _ = run_simulation(
                        current_module_delays_for_opt, PHASE_DURATIONS, RESOURCE_LIMITS, "Interleaved", group_ids, group_of, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap, modules_to_run=MODULES, preloaded_groups=opt_preloaded_groups
                    )
                    schedule_conc, _, _ = run_simulation(
                        current_module_delays_for_opt, PHASE_DURATIONS, RESOURCE_LIMITS, "Concurrent", group_ids, group_of, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap, modules_to_run=MODULES, preloaded_groups=opt_preloaded_groups
                    )
                    cycles_int = int(count_complete_cycles_df(schedule_int)["Complete Cycles"].sum())
                    cycles_conc = int(count_complete_cycles_df(schedule_conc)["Complete Cycles"].sum())
                    optimization_results.append({'Delay_Group_B': delay, 'Interleaved_Cycles': cycles_int, 'Concurrent_Cycles': cycles_conc})
                    if cycles_int > best_cycles_int:
                        best_cycles_int = cycles_int
                        best_delay = delay
                    progress_bar.progress((i + 1) / len(delay_search_range))
                    status_text.text(f"Testing offset: {delay} min. Interleaved: {cycles_int}, Concurrent: {cycles_conc}")

                return best_delay, pd.DataFrame(optimization_results)

            best_delay, optimization_summary_df = optimize_offsets(GROUP_IDS, GROUP_OF)
            st.success("Optimization Complete!")
            st.write(f"**Optimal Group Offset: {best_delay} minutes** (optimized for Interleaved)")
            final_delay_to_use = best_delay

        else:
            # Manual: set MANUAL_DELAY in code
            MANUAL_DELAY = 62
            final_delay_to_use = MANUAL_DELAY

        # --- Run both simulations with the determined delay ---
        st.markdown("---")
        actual_module_delays = build_module_delays(final_delay_to_use, GROUP_IDS, GROUP_OF, MODULES)
        actual_preloaded_groups = {"B"} if preload_group_b else None

        schedule_int, resource_usage_int, conflicts_int = run_simulation(
            actual_module_delays, PHASE_DURATIONS, RESOURCE_LIMITS, "Interleaved", GROUP_IDS, GROUP_OF, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap, modules_to_run=MODULES, preloaded_groups=actual_preloaded_groups
        )
        schedule_conc, resource_usage_conc, _ = run_simulation(
            actual_module_delays, PHASE_DURATIONS, RESOURCE_LIMITS, "Concurrent", GROUP_IDS, GROUP_OF, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap, modules_to_run=MODULES, preloaded_groups=actual_preloaded_groups
        )

        cycle_counts_int = count_complete_cycles_df(schedule_int)
        cycle_counts_conc = count_complete_cycles_df(schedule_conc)
        cycles_int = int(cycle_counts_int["Complete Cycles"].sum())
        cycles_conc = int(cycle_counts_conc["Complete Cycles"].sum())

        # Baseline run (Group A only, no resource sharing)
        modules_A = [m for m in MODULES if GROUP_OF[m] == "A"]
        baseline_delays = {m: 0 for m in modules_A}
        schedule_baseline, _, _ = run_simulation(
            baseline_delays, PHASE_DURATIONS, RESOURCE_LIMITS, "Interleaved", GROUP_IDS, GROUP_OF,
            phase_durations_by_group=PHASE_DURATIONS_BY_GROUP,
            shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap,
            modules_to_run=modules_A, baseline_mode=True
        )
        cycle_counts_baseline = count_complete_cycles_df(schedule_baseline)
        cycles_baseline = int(cycle_counts_baseline["Complete Cycles"].sum())

        # Cycle count per pair: a pair's cycle = one Adsorption-to-next-Adsorption span of
        # its combined (merged) timeline — matches what's visually countable on the Gantt
        # chart below, unlike summing each of the 8 modules' own cycle counts (which
        # overcounts, since several modules adsorb at once within the same pair).
        def merge_intervals(intervals):
            """Union of overlapping/touching (start, end) intervals -> [(start, width), ...]."""
            if not intervals:
                return []
            intervals = sorted(intervals)
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            return [(s, e - s) for s, e in merged]

        def pair_complete_cycles(schedule_df, group_ids):
            """For each group, merge all its modules' Adsorption windows into one
            timeline and count the distinct (non-overlapping) spans — this is a pair's
            "Total Cycles", matching what's visually countable on the merged Gantt
            chart rather than summing each module's own cycle count."""
            rows = []
            for gid in group_ids:
                mods_in_group = [m for m in MODULES if GROUP_OF[m] == gid]
                ads = schedule_df[
                    schedule_df["Module"].isin(mods_in_group) & (schedule_df["Phase"] == "Adsorption")
                    & schedule_df["Complete"]
                ]
                intervals = [
                    (max(0, s), min(TOTAL_MINUTES, e))
                    for s, e in zip(ads["Start"], ads["End"]) if s < e
                ]
                merged = merge_intervals(intervals)
                rows.append({"Pair": f"Group {gid}", "Total Cycles": len(merged)})
            return pd.DataFrame(rows)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.subheader("Baseline")
            pair_totals_baseline = pair_complete_cycles(schedule_baseline, ["A"])
            st.dataframe(pair_totals_baseline, use_container_width=True, hide_index=True)
            st.metric("Total Cycles", int(pair_totals_baseline["Total Cycles"].sum()))
        with res_col2:
            st.subheader("Concurrent")
            pair_totals_conc = pair_complete_cycles(schedule_conc, GROUP_IDS)
            st.dataframe(pair_totals_conc, use_container_width=True, hide_index=True)
            st.metric("Total Cycles", int(pair_totals_conc["Total Cycles"].sum()))
        with res_col3:
            st.subheader("Interleaved")
            pair_totals_int = pair_complete_cycles(schedule_int, GROUP_IDS)
            st.dataframe(pair_totals_int, use_container_width=True, hide_index=True)
            st.metric("Total Cycles", int(pair_totals_int["Total Cycles"].sum()))

        # === Yield & Energy Analysis ===
        st.markdown("### Yield & Energy Analysis")
        st.caption("Total Yield/Energy = a pair's Total Cycles (above) × the per-cycle rate entered for that pair.")

        def pair_yield_energy(pair_totals_df):
            """Multiply each pair's Total Cycles by the per-cycle Energy/Yield rate
            entered for that pair to get plant output totals, plus a derived kWh
            per kg CO2 efficiency figure."""
            df = pair_totals_df.copy()
            gids = df["Pair"].str.replace("Group ", "", regex=False)
            df["Total Yield (kg CO2)"] = [
                round(cycles * PAIR_YIELD_PER_CYCLE[gid], 1) for cycles, gid in zip(df["Total Cycles"], gids)
            ]
            df["Total Energy (kWh)"] = [
                round(cycles * PAIR_ENERGY_PER_CYCLE[gid], 1) for cycles, gid in zip(df["Total Cycles"], gids)
            ]
            df["kWh per kg CO2"] = [
                round(e / y, 3) if y > 0 else "—"
                for y, e in zip(df["Total Yield (kg CO2)"], df["Total Energy (kWh)"])
            ]
            return df

        yield_energy_baseline = pair_yield_energy(pair_totals_baseline)
        yield_energy_conc = pair_yield_energy(pair_totals_conc)
        yield_energy_int = pair_yield_energy(pair_totals_int)

        # Publish Concurrent/Interleaved totals for the Yield vs Cycles comparison
        # tab. Only refreshes when this "Generate" button is (re)clicked, not on
        # every app interaction — see that tab's caption for why. Uses the
        # pair-merged "Total Cycles" (same figure shown in the pair cycle-count
        # table above), NOT cycles_conc/cycles_int — those sum each module's own
        # cycle counts per pair and overcount for the same reason fixed earlier
        # in "Cycle count per pair".
        st.session_state.setdefault("process_comparison", {})
        st.session_state["process_comparison"][f"Concurrent{comparison_suffix}"] = {
            "Run": st.session_state["app_run_id"],
            "Total Cycles": int(yield_energy_conc["Total Cycles"].sum()),
            "Total Yield (kg CO2)": float(yield_energy_conc["Total Yield (kg CO2)"].sum()),
            "Total Energy (kWh)": float(yield_energy_conc["Total Energy (kWh)"].sum()),
        }
        st.session_state["process_comparison"][f"Interleaved{comparison_suffix}"] = {
            "Run": st.session_state["app_run_id"],
            "Total Cycles": int(yield_energy_int["Total Cycles"].sum()),
            "Total Yield (kg CO2)": float(yield_energy_int["Total Yield (kg CO2)"].sum()),
            "Total Energy (kWh)": float(yield_energy_int["Total Energy (kWh)"].sum()),
        }

        ye_col1, ye_col2, ye_col3 = st.columns(3)
        with ye_col1:
            st.subheader("Baseline")
            st.dataframe(yield_energy_baseline, use_container_width=True, hide_index=True)
            st.metric("Total Yield", f"{yield_energy_baseline['Total Yield (kg CO2)'].sum():.1f} kg CO2")
            st.metric("Total Energy", f"{yield_energy_baseline['Total Energy (kWh)'].sum():.1f} kWh")
        with ye_col2:
            st.subheader("Concurrent")
            st.dataframe(yield_energy_conc, use_container_width=True, hide_index=True)
            st.metric("Total Yield", f"{yield_energy_conc['Total Yield (kg CO2)'].sum():.1f} kg CO2")
            st.metric("Total Energy", f"{yield_energy_conc['Total Energy (kWh)'].sum():.1f} kWh")
        with ye_col3:
            st.subheader("Interleaved")
            st.dataframe(yield_energy_int, use_container_width=True, hide_index=True)
            st.metric("Total Yield", f"{yield_energy_int['Total Yield (kg CO2)'].sum():.1f} kg CO2")
            st.metric("Total Energy", f"{yield_energy_int['Total Energy (kWh)'].sum():.1f} kWh")

        # === Gantt Charts (both modes) ===
        st.markdown("### Schedule Gantt Charts")
        colors_interleaved = {
            'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
            'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'
        }
        colors_concurrent = {
            'Adsorption': '#4B9CD3',
            'Evacuation': '#90EE90', 'NCG Purging': '#90EE90', 'Heating': '#90EE90',
            'CO2 Purging': '#90EE90', 'Cooling': '#90EE90'
        }
        group_to_y = {gid: i for i, gid in enumerate(GROUP_IDS)}
        bar_height = 0.6
        desorption_set = DESORPTION_PHASES | {'Cooling'}

        def draw_gantt(ax, plot_df, colors, is_concurrent, title):
            """Draw a 2-row (Group A / Group B) Gantt chart from a per-module schedule.
            Since several modules in the same group can be in the same phase at once,
            each group's activity is merged (via merge_intervals) into single bars per
            phase, so overlapping module activity shows as one continuous block instead
            of stacked/duplicate bars. Concurrent mode only distinguishes Adsorption vs.
            "locked" desorption; Interleaved mode shows each phase in its own color."""
            plot_df = plot_df.copy()
            plot_df['Group'] = plot_df['Module'].apply(lambda m: GROUP_OF.get(int(m)))
            if is_concurrent:
                for gid in GROUP_IDS:
                    grp_df = plot_df[plot_df['Group'] == gid]
                    y_pos = group_to_y[gid]
                    yrange = (y_pos - bar_height / 2, bar_height)
                    des_intervals = []
                    ads_intervals = []
                    for _, row in grp_df.iterrows():
                        start = max(0, row['Start'])
                        end = min(TOTAL_MINUTES, row['End'])
                        if start >= end:
                            continue
                        if row['Phase'] == 'Adsorption':
                            ads_intervals.append((start, end))
                        elif row['Phase'] in desorption_set:
                            des_intervals.append((start, end))
                    des_segments = merge_intervals(des_intervals)
                    ads_segments = merge_intervals(ads_intervals)
                    if des_segments:
                        ax.broken_barh(des_segments, yrange, facecolors=colors['Evacuation'],
                                       edgecolors=colors['Evacuation'], linewidths=0.5, zorder=1)
                    if ads_segments:
                        ax.broken_barh(ads_segments, yrange, facecolors=colors['Adsorption'],
                                       edgecolors=colors['Adsorption'], linewidths=0.5, zorder=10)
            else:
                for gid in GROUP_IDS:
                    grp_df = plot_df[plot_df['Group'] == gid]
                    y_pos = group_to_y[gid]
                    yrange = (y_pos - bar_height / 2, bar_height)
                    for phase in PHASES:
                        intervals = []
                        for _, row in grp_df[grp_df['Phase'] == phase].iterrows():
                            start = max(0, row['Start'])
                            end = min(TOTAL_MINUTES, row['End'])
                            if start >= end:
                                continue
                            intervals.append((start, end))
                        segments = merge_intervals(intervals)
                        if segments:
                            c = colors.get(phase, colors['Adsorption'])
                            ax.broken_barh(segments, yrange, facecolors=c, edgecolors='black', linewidths=0.5)
            ax.set_yticks(range(len(GROUP_IDS)))
            ax.set_yticklabels([f"Group {gid} ({len([m for m in MODULES if GROUP_OF[m] == gid])} modules)" for gid in GROUP_IDS])
            ax.set_ylabel('Pair')
            ax.set_xlabel('Time (minutes)')
            ax.set_title(title)
            ax.set_xlim(0, TOTAL_MINUTES)
            ax.set_ylim(len(GROUP_IDS) - 0.5, -0.5)
            if is_concurrent:
                legend_handles = [plt.Rectangle((0,0),1,1,color=colors['Adsorption']), plt.Rectangle((0,0),1,1,color=colors['Evacuation'])]
                legend_labels = ['Adsorption', 'Desorption (locked)']
            else:
                legend_handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors.values()]
                legend_labels = list(colors.keys())
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8)

        gantt_col1, gantt_col2 = st.columns(2)
        fig_height = max(3, len(GROUP_IDS) * 1.5)
        with gantt_col1:
            fig1, ax1 = plt.subplots(figsize=(12, fig_height))
            draw_gantt(ax1, schedule_conc, colors_concurrent, True, f'Concurrent (Delay: {final_delay_to_use} min)')
            plt.tight_layout()
            st.pyplot(fig1)
        with gantt_col2:
            fig2, ax2 = plt.subplots(figsize=(12, fig_height))
            draw_gantt(ax2, schedule_int, colors_interleaved, False, f'Interleaved (Delay: {final_delay_to_use} min)')
            plt.tight_layout()
            st.pyplot(fig2)

        # === Module Utilisation & Idle Time ===
        def compute_module_metrics(schedule_df):
            """Returns DataFrame with Module, Active (min), Idle (min), Adsorption (min), Utilisation %, Adsorption Utilisation %."""
            rows = []
            for mod in MODULES:
                mod_df = schedule_df[schedule_df["Module"] == mod]
                if len(mod_df) == 0:
                    rows.append({"Module": f"M{mod}", "Active (min)": 0, "Idle (min)": TOTAL_MINUTES, "Adsorption (min)": 0, "Utilisation %": 0, "Adsorption Utilisation %": 0})
                    continue
                active = int((mod_df["End"] - mod_df["Start"]).sum())
                ads_df = mod_df[mod_df["Phase"] == "Adsorption"]
                ads_time = int((ads_df["End"] - ads_df["Start"]).sum()) if len(ads_df) > 0 else 0
                idle = max(0, TOTAL_MINUTES - active)
                rows.append({
                    "Module": f"M{mod}",
                    "Active (min)": active,
                    "Idle (min)": idle,
                    "Adsorption (min)": ads_time,
                    "Utilisation %": round(100 * active / TOTAL_MINUTES, 1),
                    "Adsorption Utilisation %": round(100 * ads_time / TOTAL_MINUTES, 1),
                })
            return pd.DataFrame(rows)

        st.markdown("### Module Utilisation & Idle Time")
        util_int = compute_module_metrics(schedule_int)
        util_conc = compute_module_metrics(schedule_conc)
        with st.expander("Module utilisation tables (Active, Idle, Adsorption %)"):
            util_col1, util_col2 = st.columns(2)
            with util_col1:
                st.caption("Concurrent")
                st.dataframe(util_conc, use_container_width=True, hide_index=True)
            with util_col2:
                st.caption("Interleaved")
                st.dataframe(util_int, use_container_width=True, hide_index=True)
            st.caption("Active = time in any phase. Idle = waiting/unused time.")

        st.markdown("#### What the idle time is spent waiting for")
        st.caption(
            "Average idle minutes per module in each group, split by cause. Each column adds up "
            "to that group's average Idle (min) in the utilisation tables."
        )
        idle_units = {f"Group {gid}": [m for m in MODULES if GROUP_OF[m] == gid] for gid in GROUP_IDS}
        idle_col1, idle_col2 = st.columns(2)
        with idle_col1:
            st.caption("Concurrent")
            st.dataframe(idle_breakdown_table(schedule_conc, "Module", idle_units, TOTAL_MINUTES),
                         use_container_width=True, hide_index=True)
        with idle_col2:
            st.caption("Interleaved")
            st.dataframe(idle_breakdown_table(schedule_int, "Module", idle_units, TOTAL_MINUTES),
                         use_container_width=True, hide_index=True)

        # Add Avg Idle/Utilisation to the Concurrent/Interleaved entries already
        # published above, for the Yield vs Cycles tab's idle-time summary.
        st.session_state["process_comparison"][f"Concurrent{comparison_suffix}"]["Avg Idle (min)"] = round(float(util_conc["Idle (min)"].mean()), 1)
        st.session_state["process_comparison"][f"Concurrent{comparison_suffix}"]["Avg Utilisation %"] = round(float(util_conc["Utilisation %"].mean()), 1)
        st.session_state["process_comparison"][f"Interleaved{comparison_suffix}"]["Avg Idle (min)"] = round(float(util_int["Idle (min)"].mean()), 1)
        st.session_state["process_comparison"][f"Interleaved{comparison_suffix}"]["Avg Utilisation %"] = round(float(util_int["Utilisation %"].mean()), 1)

        # Distinct colors for Interleaved vs Concurrent (not Adsorption/Desorption)
        color_interleaved = "#E07C5E"
        color_concurrent = "#6C5CE7"
        # Utilisation bar chart (side by side) with value labels
        fig_util, ax_util = plt.subplots(figsize=(12, 4))
        x = np.arange(len(MODULES))
        w = 0.35
        bars_conc = ax_util.bar(x - w/2, util_conc["Utilisation %"], w, label="Concurrent", color=color_concurrent, alpha=0.85)
        bars_int = ax_util.bar(x + w/2, util_int["Utilisation %"], w, label="Interleaved", color=color_interleaved, alpha=0.85)
        for bar in bars_int:
            h = bar.get_height()
            if h > 5:
                ax_util.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom", fontsize=7, color=color_interleaved)
        for bar in bars_conc:
            h = bar.get_height()
            if h > 5:
                ax_util.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom", fontsize=7, color=color_concurrent)
        ax_util.set_xlabel("Module")
        ax_util.set_ylabel("Utilisation %")
        ax_util.set_title("Module Utilisation (Active Time / Total Time)")
        ax_util.set_xticks(x)
        ax_util.set_xticklabels([f"M{m}" for m in MODULES])
        ax_util.legend()
        ax_util.set_ylim(0, 115)
        plt.tight_layout()
        st.pyplot(fig_util)
        plt.close(fig_util)

        # Idle time comparison with value labels
        fig_idle, ax_idle = plt.subplots(figsize=(12, 4))
        bars_idle_conc = ax_idle.bar(x - w/2, util_conc["Idle (min)"], w, label="Concurrent", color=color_concurrent, alpha=0.85)
        bars_idle_int = ax_idle.bar(x + w/2, util_int["Idle (min)"], w, label="Interleaved", color=color_interleaved, alpha=0.85)
        for bar in bars_idle_int:
            h = bar.get_height()
            if h > 2:
                ax_idle.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom", fontsize=7, color=color_interleaved)
        for bar in bars_idle_conc:
            h = bar.get_height()
            if h > 2:
                ax_idle.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom", fontsize=7, color=color_concurrent)
        ax_idle.set_xlabel("Module")
        ax_idle.set_ylabel("Idle (minutes)")
        ax_idle.set_title("Idle Time per Module (waiting / unused)")
        ax_idle.set_xticks(x)
        ax_idle.set_xticklabels([f"M{m}" for m in MODULES])
        ax_idle.legend()
        plt.tight_layout()
        st.pyplot(fig_idle)
        plt.close(fig_idle)

        # Phase breakdown (time per phase)
        st.markdown("### Phase Breakdown (Total Minutes per Phase)")
        st.caption("Total minutes spent in each phase, all modules combined")
        phase_totals_int = schedule_int.groupby("Phase").apply(lambda g: (g["End"] - g["Start"]).sum()).reindex(PHASES, fill_value=0)
        phase_totals_conc = schedule_conc.groupby("Phase").apply(lambda g: (g["End"] - g["Start"]).sum()).reindex(PHASES, fill_value=0)
        colors_phase = ['#4B9CD3', '#FFB347', '#FFD700', '#E97451', '#90EE90', '#9370DB']
        pbrk_col1, pbrk_col2 = st.columns(2)
        total_conc = phase_totals_conc.sum() or 1
        total_int = phase_totals_int.sum() or 1
        pct_conc = (phase_totals_conc.values / total_conc) * 100
        pct_int = (phase_totals_int.values / total_int) * 100
        with pbrk_col1:
            st.caption("Concurrent")
            fig_phase, ax_phase = plt.subplots(figsize=(6, 3.5))
            bars_conc = ax_phase.barh(PHASES, phase_totals_conc.values, color=colors_phase[:len(PHASES)])
            for bar, pct in zip(bars_conc, pct_conc):
                ax_phase.annotate(f"{pct:.1f}%", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                 xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
            ax_phase.set_xlabel("Total Minutes")
            ax_phase.set_title("Concurrent")
            plt.tight_layout()
            st.pyplot(fig_phase)
            plt.close(fig_phase)
        with pbrk_col2:
            st.caption("Interleaved")
            fig_phase2, ax_phase2 = plt.subplots(figsize=(6, 3.5))
            bars_int = ax_phase2.barh(PHASES, phase_totals_int.values, color=colors_phase[:len(PHASES)])
            for bar, pct in zip(bars_int, pct_int):
                ax_phase2.annotate(f"{pct:.1f}%", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                  xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
            ax_phase2.set_xlabel("Total Minutes")
            ax_phase2.set_title("Interleaved")
            plt.tight_layout()
            st.pyplot(fig_phase2)
            plt.close(fig_phase2)

    # Each configuration's Generate button sits on its own tab, right under its
    # inputs; the button at the top runs all four.
    TAB3_RUNS = [
        ("8-8", results_slot_88, (
            "8-8 Configuration", "_88", " (8-8)",
            MODULES, GROUP_IDS, GROUP_OF,
            PHASE_DURATIONS, PHASE_DURATIONS_BY_GROUP, RESOURCE_LIMITS,
            PAIR_ENERGY_PER_CYCLE_TAB3, PAIR_YIELD_PER_CYCLE_TAB3,
            concurrent_desorption_cap, shared_evac_cooling_cap, shared_purge_heat_co2_cap,
        )),
        ("4-4", results_slot_44, (
            "4-4 Configuration", "_44", " (4-4)",
            MODULES_44, GROUP_IDS, GROUP_OF_44,
            PHASE_DURATIONS_44, PHASE_DURATIONS_BY_GROUP_44, RESOURCE_LIMITS_44,
            PAIR_ENERGY_PER_CYCLE_TAB3_44, PAIR_YIELD_PER_CYCLE_TAB3_44,
            concurrent_desorption_cap_44, shared_evac_cooling_cap_44, shared_purge_heat_co2_cap_44,
        )),
        ("2-2", results_slot_22, (
            "2-2 Configuration", "_22", " (2-2)",
            MODULES_22, GROUP_IDS, GROUP_OF_22,
            PHASE_DURATIONS_22, PHASE_DURATIONS_BY_GROUP_22, RESOURCE_LIMITS_22,
            PAIR_ENERGY_PER_CYCLE_TAB3_22, PAIR_YIELD_PER_CYCLE_TAB3_22,
            concurrent_desorption_cap_22, shared_evac_cooling_cap_22, shared_purge_heat_co2_cap_22,
        )),
        ("6-6", results_slot_66, (
            "6-6 Configuration", "_66", " (6-6)",
            MODULES_66, GROUP_IDS, GROUP_OF_66,
            PHASE_DURATIONS_66, PHASE_DURATIONS_BY_GROUP_66, RESOURCE_LIMITS_66,
            PAIR_ENERGY_PER_CYCLE_TAB3_66, PAIR_YIELD_PER_CYCLE_TAB3_66,
            concurrent_desorption_cap_66, shared_evac_cooling_cap_66, shared_purge_heat_co2_cap_66,
        )),
    ]
    for cfg_label, cfg_slot, cfg_args in TAB3_RUNS:
        with cfg_slot:
            clicked = st.button(
                f"Generate {cfg_label} Schedule and Analyze", key=f"generate_tab3_{cfg_label}", type="primary",
            )
            if clicked or generate_all_tab3:
                run_and_display_config(*cfg_args)

with tab4:
    # "Advanced Interleaved": a 3-pair (Group A/B/C) plant-level scheduler with a
    # 7-phase cycle that adds Repressurization after Cooling. Unlike tab3, this
    # schedules whole pairs as single units (not individual modules) — see
    # run_advanced_interleaved() below for the actual scheduling rules.
    st.subheader("Advanced Interleaved - 3 Pair Scheduling")

    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling','Repressurization' ]

    st.markdown("### Inputs")
    col1, col2 = st.columns(2)

    with col1:
        TOTAL_MINUTES_ADV = st.number_input(
            "Operating Period (min)",
            0, 1440, 1440,
            key="adv_total_minutes"
        )

    with col2:
        adv_enforce_evac_cool = st.checkbox(
            "Evacuation Overrides Cooling (Advanced Interleaved)",
            value=True,
            key="adv_enforce_evac_cool",
            help="When another pair's Evacuation overlaps a pair's Cooling, Cooling pauses and resumes once that Evacuation ends."
        )

    adv_generate_all = st.button(
        "Generate all five configurations", key="adv_generate_all",
        help="Runs 6-6-4, 8-4-4, 6-5-5, 4-4-4-4 and 2-2-2-2 N3/N4 at once; each one's results appear on its own tab.",
    )
    st.caption(
        "Each tab below holds one configuration: its inputs and duration tables, its own "
        "Generate button, and its results. 6-6-4, 8-4-4 and 6-5-5 are 3-pair configurations; "
        "4-4-4-4 and 2-2-2-2 N3/N4 are 4-pair configurations."
    )
    adv_tab_664, adv_tab_844, adv_tab_655, adv_tab_4x4, adv_tab_n3n4 = st.tabs(
        ["6-6-4 Configuration", "8-4-4 Configuration", "6-5-5 Configuration", "4-4-4-4 Configuration", "2-2-2-2 (N3/N4) Configuration"]
    )
    with adv_tab_664:
        st.caption("Adsorption rule (3-pair): one privileged pair can overlap its Adsorption with the other two, but those two cannot overlap each other.")
        st.caption("Phase durations per pair - edit directly in the table")

        PAIRS = ["Group A", "Group B", "Group C"]
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of the plant's 16 real modules are grouped together for desorption in this configuration.")
        st.dataframe(physical_pairing_table(PAIRS, PAIR_PHYSICAL_MODULES), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(PAIRS, PAIR_PLANT_MODULE, PAIR_PLANT_MODULE, PAIR_YIELD_FORMULA),
            use_container_width=True, hide_index=True,
        )
        adv_phase_columns = ["Phase"] + [f"{p} (min)" for p in PAIRS]

        # Fallback for any phase/pair the stage log doesn't cover (e.g. Repressurization,
        # which has no equivalent log column, or if the CSV is missing).
        FALLBACK_PHASE_DURATIONS = {
            **CORE_PHASE_FALLBACK_DURATIONS, "Repressurization": 5,
        }
        stage_defaults_by_pair = load_stage_duration_defaults_by_pair(PAIRS)
        # Bump whenever STAGE_PHASE_COLUMNS' mapping (which source columns feed which
        # phase) or FALLBACK_PHASE_DURATIONS changes, so stale sessions refresh.
        ADV_PHASE_DURATIONS_DEFAULTS_VERSION = 2

        def _pair_phase_durations(pair):
            pair_defaults = stage_defaults_by_pair.get(pair, {})
            return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

        get_versioned_default_table(
            "adv_phase_durations", adv_phase_columns, ADV_PHASE_DURATIONS_DEFAULTS_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": _pair_phase_durations("Group A"),
                "Group B (min)": _pair_phase_durations("Group B"),
                "Group C (min)": _pair_phase_durations("Group C"),
            }),
        )

        st.caption(
            "Defaults for Adsorption, Evacuation, NCG Purging, Heating, CO2 Purging, and Cooling are "
            "seeded from the live stage-duration data: Group A and Group C from the N1N2N3-M1n3 "
            "cycle average, Group B from the N1N2-M1n3 cycle average. Repressurization has no "
            "equivalent in that data, so it keeps its original default. Edit any cell if a pair's real "
            "durations differ."
        )

        adv_phase_table = st.data_editor(
            st.session_state.adv_phase_durations,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
                "Group C (min)": st.column_config.NumberColumn("Group C (min)", min_value=0, max_value=240, required=True),
            },
            key="adv_phase_editor"
        )

        st.session_state.adv_phase_durations = adv_phase_table

        PHASE_DURATIONS_BY_PAIR = {
            pair: {
                phase: int(
                    adv_phase_table.loc[
                        adv_phase_table["Phase"] == phase,
                        f"{pair} (min)"
                    ].iloc[0]
                )
                for phase in PHASES
            }
            for pair in PAIRS
        }

        pair_plant_defaults = load_plant_cycle_defaults_by_pair(PAIRS)
        st.caption("Energy used & plant yield per pair, per completed cycle — edit directly in the table")
        st.caption(
            "Defaults are seeded from the live plant-cycle data. Energy per Cycle uses each pair's "
            "single Module average (PAIR_PLANT_MODULE, same source as phase durations): Group A/C from "
            f"N1N2N3-M1n3 ({pair_plant_defaults['Group A'][0]} kWh), Group B from N1N2-M1n3 "
            f"({pair_plant_defaults['Group B'][0]} kWh). Yield per Cycle instead uses a different formula "
            "per pair (PAIR_YIELD_FORMULA): Group A from N1N2N3-M1n3 "
            f"({pair_plant_defaults['Group A'][1]} kg CO2), Group B from N3-M2n4 doubled "
            f"({pair_plant_defaults['Group B'][1]} kg CO2), Group C from N1N2-M1n3 + N3-M2n4 summed "
            f"({pair_plant_defaults['Group C'][1]} kg CO2). Edit per pair if a pair's real output differs."
        )
        # Bump whenever PAIR_PLANT_MODULE, PAIR_YIELD_FORMULA, or
        # load_plant_cycle_defaults_by_pair()'s sourcing changes.
        ENERGY_YIELD_TAB4_DEFAULTS_VERSION = 3
        energy_yield_tab4_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab4", energy_yield_tab4_cols, ENERGY_YIELD_TAB4_DEFAULTS_VERSION,
            lambda: pd.DataFrame({
                "Pair": PAIRS,
                "Energy per Cycle (kWh)": [pair_plant_defaults[p][0] for p in PAIRS],
                "Yield per Cycle (kg CO2)": [pair_plant_defaults[p][1] for p in PAIRS],
            }),
        )
        energy_yield_tab4 = st.data_editor(
            st.session_state.energy_yield_tab4,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_tab4",
        )
        st.session_state.energy_yield_tab4 = energy_yield_tab4

        PAIR_ENERGY_PER_CYCLE = {
            pair: float(energy_yield_tab4.loc[energy_yield_tab4["Pair"] == pair, "Energy per Cycle (kWh)"].iloc[0])
            for pair in PAIRS
        }
        PAIR_YIELD_PER_CYCLE = {
            pair: float(energy_yield_tab4.loc[energy_yield_tab4["Pair"] == pair, "Yield per Cycle (kg CO2)"].iloc[0])
            for pair in PAIRS
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        adv_results_slot_664 = st.container()

    # === 8-4-4 comparison configuration ===
    # Same scheduling engine, phase list, operating period, and Evacuation-overrides-
    # Cooling setting as the 6-6-4 config above — only the pairing/module sourcing
    # differs. See PAIRS_8_4_4 / PAIR_PLANT_MODULE_8_4_4 near the top of the file.
    with adv_tab_844:
        st.caption("Adsorption rule (3-pair): one privileged pair can overlap its Adsorption with the other two, but those two cannot overlap each other.")
        st.caption("Same scheduling rules as 6-6-4, applied to an 8-4-4 pairing instead, for side-by-side comparison.")
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of the plant's 16 real modules are grouped together for desorption in this configuration.")
        st.dataframe(physical_pairing_table(PAIRS_8_4_4, PAIR_PHYSICAL_MODULES_8_4_4), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(PAIRS_8_4_4, PAIR_PLANT_MODULE_8_4_4, PAIR_PLANT_MODULE_8_4_4, PAIR_YIELD_FORMULA_8_4_4),
            use_container_width=True, hide_index=True,
        )

        adv_phase_columns_844 = ["Phase"] + [f"{p} (min)" for p in PAIRS_8_4_4]
        stage_defaults_by_pair_844 = load_stage_duration_defaults_by_pair(PAIRS_8_4_4, PAIR_PLANT_MODULE_8_4_4)
        ADV_PHASE_DURATIONS_844_VERSION = 2

        def _pair_phase_durations_844(pair):
            pair_defaults = stage_defaults_by_pair_844.get(pair, {})
            return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

        get_versioned_default_table(
            "adv_phase_durations_844", adv_phase_columns_844, ADV_PHASE_DURATIONS_844_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": _pair_phase_durations_844("Group A"),
                "Group B (min)": _pair_phase_durations_844("Group B"),
                "Group C (min)": _pair_phase_durations_844("Group C"),
            }),
        )
        st.caption(
            "Phase durations seeded from the live stage-duration data: Group A (8-module pair) from "
            "the N1N2N3-M1n3 cycle average, Groups B and C (4-module pairs) from the N1N2-M1n3 cycle average."
        )
        adv_phase_table_844 = st.data_editor(
            st.session_state.adv_phase_durations_844,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
                "Group C (min)": st.column_config.NumberColumn("Group C (min)", min_value=0, max_value=240, required=True),
            },
            key="adv_phase_editor_844",
        )
        st.session_state.adv_phase_durations_844 = adv_phase_table_844

        PHASE_DURATIONS_BY_PAIR_844 = {
            pair: {
                phase: int(
                    adv_phase_table_844.loc[
                        adv_phase_table_844["Phase"] == phase,
                        f"{pair} (min)"
                    ].iloc[0]
                )
                for phase in PHASES
            }
            for pair in PAIRS_8_4_4
        }

        pair_plant_defaults_844 = load_plant_cycle_defaults_weighted(
            PAIRS_8_4_4, PAIR_PLANT_MODULE_8_4_4, PAIR_YIELD_FORMULA_8_4_4
        )
        # Bumped because Yield per Cycle sourcing changed from the simple single-Module
        # average (matching Energy) to PAIR_YIELD_FORMULA_8_4_4's weighted sum.
        ENERGY_YIELD_844_VERSION = 3
        energy_yield_844_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab4_844", energy_yield_844_cols, ENERGY_YIELD_844_VERSION,
            lambda: pd.DataFrame({
                "Pair": PAIRS_8_4_4,
                "Energy per Cycle (kWh)": [pair_plant_defaults_844[p][0] for p in PAIRS_8_4_4],
                "Yield per Cycle (kg CO2)": [pair_plant_defaults_844[p][1] for p in PAIRS_8_4_4],
            }),
        )
        st.caption(
            "Energy per Cycle uses the same sourcing as phase durations: Group A (8-module pair) from "
            "N1N2N3-M1n3, Groups B and C (4-module pairs) from N1N2-M1n3. Yield per Cycle instead uses "
            "a different formula per pair (PAIR_YIELD_FORMULA_8_4_4): Group A from N1N2N3-M1n3 + N3-M2n4 "
            "summed, Group B from N1N2-M1n3 alone, Group C from N3-M2n4 doubled."
        )
        energy_yield_table_844 = st.data_editor(
            st.session_state.energy_yield_tab4_844,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_844",
        )
        st.session_state.energy_yield_tab4_844 = energy_yield_table_844

        PAIR_ENERGY_PER_CYCLE_844 = {
            pair: float(energy_yield_table_844.loc[energy_yield_table_844["Pair"] == pair, "Energy per Cycle (kWh)"].iloc[0])
            for pair in PAIRS_8_4_4
        }
        PAIR_YIELD_PER_CYCLE_844 = {
            pair: float(energy_yield_table_844.loc[energy_yield_table_844["Pair"] == pair, "Yield per Cycle (kg CO2)"].iloc[0])
            for pair in PAIRS_8_4_4
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        adv_results_slot_844 = st.container()

    # === "6-5-5" configuration ===
    with adv_tab_655:
        st.caption("Adsorption rule (3-pair): one privileged pair can overlap its Adsorption with the other two, but those two cannot overlap each other.")
        st.caption(
            "3 pairs (16 = 6+5+5), sized for the best achievable Yield: Group A = N1N2N3-M1n3 "
            "(6 modules, the highest real Yield per Cycle available). Group B and Group C (5 modules "
            "each) take their phase durations from N1N2-M1n3 — the closest well-sampled real "
            "combination in size, keeping cycle timing realistic — but their Yield per Cycle instead "
            "uses weighted sums of N2-M2n4/N3-M2n4 (see the table below). This is a 3-pair "
            "configuration, so it uses the original Adsorption-overlap rule: Group A "
            "can overlap with Group B/C, but Group B and Group C cannot overlap each other."
        )
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of the plant's 16 real modules are grouped together for desorption in this configuration.")
        st.dataframe(physical_pairing_table(PAIRS_6_5_5, PAIR_PHYSICAL_MODULES_6_5_5), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(
                PAIRS_6_5_5, PAIR_DURATION_MODULE_6_5_5, PAIR_ENERGY_MODULE_6_5_5, PAIR_YIELD_FORMULA_6_5_5,
                represents_map=REPRESENTS_6_5_5,
            ),
            use_container_width=True, hide_index=True,
        )

        adv_phase_columns_655 = ["Phase"] + [f"{p} (min)" for p in PAIRS_6_5_5]
        stage_defaults_655 = load_stage_duration_defaults_by_pair(PAIRS_6_5_5, PAIR_DURATION_MODULE_6_5_5)
        ADV_PHASE_DURATIONS_655_VERSION = 1

        def _pair_phase_durations_655(pair):
            pair_defaults = stage_defaults_655.get(pair, {})
            return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

        get_versioned_default_table(
            "adv_phase_durations_655", adv_phase_columns_655, ADV_PHASE_DURATIONS_655_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": _pair_phase_durations_655("Group A"),
                "Group B (min)": _pair_phase_durations_655("Group B"),
                "Group C (min)": _pair_phase_durations_655("Group C"),
            }),
        )
        adv_phase_table_655 = st.data_editor(
            st.session_state.adv_phase_durations_655,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
                "Group C (min)": st.column_config.NumberColumn("Group C (min)", min_value=0, max_value=240, required=True),
            },
            key="adv_phase_editor_655",
        )
        st.session_state.adv_phase_durations_655 = adv_phase_table_655

        PHASE_DURATIONS_BY_PAIR_655 = {
            pair: {
                phase: int(
                    adv_phase_table_655.loc[
                        adv_phase_table_655["Phase"] == phase,
                        f"{pair} (min)"
                    ].iloc[0]
                )
                for phase in PHASES
            }
            for pair in PAIRS_6_5_5
        }

        pair_plant_defaults_655 = load_plant_cycle_defaults_weighted(
            PAIRS_6_5_5, PAIR_ENERGY_MODULE_6_5_5, PAIR_YIELD_FORMULA_6_5_5,
        )
        ENERGY_YIELD_655_VERSION = 3
        energy_yield_655_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab4_655", energy_yield_655_cols, ENERGY_YIELD_655_VERSION,
            lambda: pd.DataFrame({
                "Pair": PAIRS_6_5_5,
                "Energy per Cycle (kWh)": [pair_plant_defaults_655[p][0] for p in PAIRS_6_5_5],
                "Yield per Cycle (kg CO2)": [pair_plant_defaults_655[p][1] for p in PAIRS_6_5_5],
            }),
        )
        energy_yield_table_655 = st.data_editor(
            st.session_state.energy_yield_tab4_655,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_655",
        )
        st.session_state.energy_yield_tab4_655 = energy_yield_table_655

        PAIR_ENERGY_PER_CYCLE_655 = {
            pair: float(energy_yield_table_655.loc[energy_yield_table_655["Pair"] == pair, "Energy per Cycle (kWh)"].iloc[0])
            for pair in PAIRS_6_5_5
        }
        PAIR_YIELD_PER_CYCLE_655 = {
            pair: float(energy_yield_table_655.loc[energy_yield_table_655["Pair"] == pair, "Yield per Cycle (kg CO2)"].iloc[0])
            for pair in PAIRS_6_5_5
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        adv_results_slot_655 = st.container()

    with adv_tab_4x4:
        st.caption("Adsorption rule (4-pair): every pair's Adsorption can overlap freely with every other pair's.")
        st.caption(
            "4 equal 4-module pairs: Group A = N1&N2's M1/M3, Group B = N1&N2's M2/M4, Group C = "
            "N3&N4's M1/M3, Group D = N3&N4's M2/M4. Only Group A's combination (N1N2-M1n3) is "
            "actually recorded, so every pair's phase durations and Energy per Cycle come from "
            "N1N2-M1n3's own real average. Yield per Cycle differs: Group A and Group B keep "
            "N1N2-M1n3's own real Yield, while Group C and Group D use N3-M2n4's real Yield "
            "doubled. Adsorption overlap is unrestricted for all 4 pairs."
        )
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of the plant's 16 real modules are grouped together for desorption in this configuration.")
        st.dataframe(physical_pairing_table(PAIRS_4X4, PAIR_PHYSICAL_MODULES_4X4), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield proxy.")
        st.dataframe(
            module_mapping_table(
                PAIRS_4X4, PAIR_MODULE_4X4, PAIR_MODULE_4X4, PAIR_YIELD_FORMULA_4X4,
                represents_map=REPRESENTS_4X4,
            ),
            use_container_width=True, hide_index=True,
        )

        adv_phase_columns_4x4 = ["Phase"] + [f"{p} (min)" for p in PAIRS_4X4]
        stage_defaults_4x4 = load_stage_duration_defaults_by_pair(PAIRS_4X4, PAIR_MODULE_4X4)
        ADV_PHASE_DURATIONS_4X4_VERSION = 1

        def _pair_phase_durations_4x4(pair):
            pair_defaults = stage_defaults_4x4.get(pair, {})
            return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

        get_versioned_default_table(
            "adv_phase_durations_4x4", adv_phase_columns_4x4, ADV_PHASE_DURATIONS_4X4_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": _pair_phase_durations_4x4("Group A"),
                "Group B (min)": _pair_phase_durations_4x4("Group B"),
                "Group C (min)": _pair_phase_durations_4x4("Group C"),
                "Group D (min)": _pair_phase_durations_4x4("Group D"),
            }),
        )
        adv_phase_table_4x4 = st.data_editor(
            st.session_state.adv_phase_durations_4x4,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
                "Group C (min)": st.column_config.NumberColumn("Group C (min)", min_value=0, max_value=240, required=True),
                "Group D (min)": st.column_config.NumberColumn("Group D (min)", min_value=0, max_value=240, required=True),
            },
            key="adv_phase_editor_4x4",
        )
        st.session_state.adv_phase_durations_4x4 = adv_phase_table_4x4

        PHASE_DURATIONS_BY_PAIR_4X4 = {
            pair: {
                phase: int(
                    adv_phase_table_4x4.loc[
                        adv_phase_table_4x4["Phase"] == phase,
                        f"{pair} (min)"
                    ].iloc[0]
                )
                for phase in PHASES
            }
            for pair in PAIRS_4X4
        }

        pair_plant_defaults_4x4 = load_plant_cycle_defaults_weighted(
            PAIRS_4X4, PAIR_MODULE_4X4, PAIR_YIELD_FORMULA_4X4,
        )
        ENERGY_YIELD_4X4_VERSION = 2
        energy_yield_4x4_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab4_4x4", energy_yield_4x4_cols, ENERGY_YIELD_4X4_VERSION,
            lambda: pd.DataFrame({
                "Pair": PAIRS_4X4,
                "Energy per Cycle (kWh)": [pair_plant_defaults_4x4[p][0] for p in PAIRS_4X4],
                "Yield per Cycle (kg CO2)": [pair_plant_defaults_4x4[p][1] for p in PAIRS_4X4],
            }),
        )
        energy_yield_table_4x4 = st.data_editor(
            st.session_state.energy_yield_tab4_4x4,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_4x4",
        )
        st.session_state.energy_yield_tab4_4x4 = energy_yield_table_4x4

        PAIR_ENERGY_PER_CYCLE_4X4 = {
            pair: float(energy_yield_table_4x4.loc[energy_yield_table_4x4["Pair"] == pair, "Energy per Cycle (kWh)"].iloc[0])
            for pair in PAIRS_4X4
        }
        PAIR_YIELD_PER_CYCLE_4X4 = {
            pair: float(energy_yield_table_4x4.loc[energy_yield_table_4x4["Pair"] == pair, "Yield per Cycle (kg CO2)"].iloc[0])
            for pair in PAIRS_4X4
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        adv_results_slot_4x4 = st.container()

    with adv_tab_n3n4:
        st.caption("Adsorption rule (4-pair): every pair's Adsorption can overlap freely with every other pair's.")
        st.caption(
            "A second 4-equal-group configuration, covering only Nelion 3 and Nelion 4 (8 of 16 "
            "modules) instead of the full plant, grouped by module type rather than by Nelion: "
            "Group A = N3's M1/M3, Group B = N4's M1/M3, Group C = N3's M2/M4, Group D = N4's "
            "M2/M4. Nelion 4 has since been commissioned and now has its own recorded cycles, so "
            "every group sources Duration, Energy and Yield directly from its own matching Module "
            "average — no proxy or mirroring needed. Since every group here is already a genuine "
            "2-module pair, no multiplier is applied anywhere. Adsorption overlap is unrestricted "
            "for all 4 pairs."
        )
        st.markdown("#### Physical Module Pairing")
        st.caption("Which of Nelion 3 and Nelion 4's 8 real modules are grouped together for desorption in this configuration.")
        st.dataframe(physical_pairing_table(PAIRS_N3N4, PAIR_PHYSICAL_MODULES_N3N4), use_container_width=True, hide_index=True)

        st.markdown("#### Data Sourcing")
        st.caption("Which real Module average from the CSV logs is used as each pair's Duration/Energy/Yield.")
        st.dataframe(
            module_mapping_table(
                PAIRS_N3N4, PAIR_MODULE_N3N4, PAIR_MODULE_N3N4, PAIR_YIELD_FORMULA_N3N4,
            ),
            use_container_width=True, hide_index=True,
        )

        adv_phase_columns_n3n4 = ["Phase"] + [f"{p} (min)" for p in PAIRS_N3N4]
        stage_defaults_n3n4 = load_stage_duration_defaults_by_pair(PAIRS_N3N4, PAIR_MODULE_N3N4)
        ADV_PHASE_DURATIONS_N3N4_VERSION = 1

        def _pair_phase_durations_n3n4(pair):
            pair_defaults = stage_defaults_n3n4.get(pair, {})
            return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

        get_versioned_default_table(
            "adv_phase_durations_n3n4", adv_phase_columns_n3n4, ADV_PHASE_DURATIONS_N3N4_VERSION,
            lambda: pd.DataFrame({
                "Phase": PHASES,
                "Group A (min)": _pair_phase_durations_n3n4("Group A"),
                "Group B (min)": _pair_phase_durations_n3n4("Group B"),
                "Group C (min)": _pair_phase_durations_n3n4("Group C"),
                "Group D (min)": _pair_phase_durations_n3n4("Group D"),
            }),
        )
        adv_phase_table_n3n4 = st.data_editor(
            st.session_state.adv_phase_durations_n3n4,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Phase": st.column_config.TextColumn("Phase", disabled=True),
                "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=240, required=True),
                "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=240, required=True),
                "Group C (min)": st.column_config.NumberColumn("Group C (min)", min_value=0, max_value=240, required=True),
                "Group D (min)": st.column_config.NumberColumn("Group D (min)", min_value=0, max_value=240, required=True),
            },
            key="adv_phase_editor_n3n4",
        )
        st.session_state.adv_phase_durations_n3n4 = adv_phase_table_n3n4

        PHASE_DURATIONS_BY_PAIR_N3N4 = {
            pair: {
                phase: int(
                    adv_phase_table_n3n4.loc[
                        adv_phase_table_n3n4["Phase"] == phase,
                        f"{pair} (min)"
                    ].iloc[0]
                )
                for phase in PHASES
            }
            for pair in PAIRS_N3N4
        }

        pair_plant_defaults_n3n4 = load_plant_cycle_defaults_weighted(
            PAIRS_N3N4, PAIR_MODULE_N3N4, PAIR_YIELD_FORMULA_N3N4,
        )
        ENERGY_YIELD_N3N4_VERSION = 1
        energy_yield_n3n4_cols = ["Pair", "Energy per Cycle (kWh)", "Yield per Cycle (kg CO2)"]
        get_versioned_default_table(
            "energy_yield_tab4_n3n4", energy_yield_n3n4_cols, ENERGY_YIELD_N3N4_VERSION,
            lambda: pd.DataFrame({
                "Pair": PAIRS_N3N4,
                "Energy per Cycle (kWh)": [pair_plant_defaults_n3n4[p][0] for p in PAIRS_N3N4],
                "Yield per Cycle (kg CO2)": [pair_plant_defaults_n3n4[p][1] for p in PAIRS_N3N4],
            }),
        )
        energy_yield_table_n3n4 = st.data_editor(
            st.session_state.energy_yield_tab4_n3n4,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pair": st.column_config.TextColumn("Pair", disabled=True),
                "Energy per Cycle (kWh)": st.column_config.NumberColumn("Energy per Cycle (kWh)", min_value=0.0, step=0.1, required=True),
                "Yield per Cycle (kg CO2)": st.column_config.NumberColumn("Yield per Cycle (kg CO2)", min_value=0.0, step=0.1, required=True),
            },
            key="energy_yield_editor_n3n4",
        )
        st.session_state.energy_yield_tab4_n3n4 = energy_yield_table_n3n4

        PAIR_ENERGY_PER_CYCLE_N3N4 = {
            pair: float(energy_yield_table_n3n4.loc[energy_yield_table_n3n4["Pair"] == pair, "Energy per Cycle (kWh)"].iloc[0])
            for pair in PAIRS_N3N4
        }
        PAIR_YIELD_PER_CYCLE_N3N4 = {
            pair: float(energy_yield_table_n3n4.loc[energy_yield_table_n3n4["Pair"] == pair, "Yield per Cycle (kg CO2)"].iloc[0])
            for pair in PAIRS_N3N4
        }
        # This configuration's Generate button and results are rendered here
        # (filled in further down, once the scheduler is defined).
        adv_results_slot_n3n4 = st.container()

    def run_advanced_interleaved(pairs, phase_durations_by_pair, total_minutes, enforce_evac_cool, free_adsorption_overlap=False):
        """
        Advanced interleaved scheduler, with whole pairs scheduled as single units.

        Sequence per pair:
        Adsorption -> Evacuation -> NCG Purging -> Heating -> CO2 Purging -> Cooling -> Repressurization

        Same approach as the Full Schedule Analysis tab: pairs are advanced in time
        order, one phase at a time, so every shared resource is first-come-first-
        served and no pair is favored (on a tie, whichever pair held the NCG ->
        Heating -> CO2 block longest ago goes first). Rules:
        - Adsorption overlap: if free_adsorption_overlap is False (the original
          3-pair rule, requires exactly 3 pairs), pairs[0] can overlap Adsorption
          with pairs[1]/pairs[2], but pairs[1] and pairs[2] cannot overlap each
          other's Adsorption. If True, every pair's Adsorption overlaps freely.
        - Evacuation: one pair at a time.
        - NCG Purging -> Heating -> CO2 Purging: booked as one unbroken block, one
          pair at a time. A pair can evacuate while another holds the block, then waits.
        - Cooling: one pair at a time. If enforce_evac_cool, another pair's
          Evacuation pauses it (including an Evacuation booked after the Cooling).
        - Repressurization: one pair at a time.

        Pairs start staggered into different phase groups (see next_step) so they
        don't all compete for the same resource at once. A phase may start any time
        before total_minutes and run past it (the plant keeps going into the next
        day): rows are clipped at total_minutes, and "Complete" marks the full
        cycles (Adsorption through Repressurization) that finished inside the period.
        """
        horizon = 2 * total_minutes + 1
        if not free_adsorption_overlap:
            pair_a, pair_b, pair_c = pairs
        CHAIN = ["NCG Purging", "Heating", "CO2 Purging"]
        STAGGER_STEPS = ["Adsorption", "Evacuation", "Cooling", "Repressurization"]
        next_step = {pair: STAGGER_STEPS[i % len(STAGGER_STEPS)] for i, pair in enumerate(pairs)}
        order = {pair: i for i, pair in enumerate(pairs)}
        ready = {pair: 0 for pair in pairs}
        last_chain_end = {pair: -1 for pair in pairs}

        cycles = {pair: [] for pair in pairs}  # each: {"phases": [[phase, start, end], ...], "full", "done"}
        ads_busy = {pair: [] for pair in pairs}
        evac_busy = []  # (start, end, pair)
        chain_busy, repress_busy = [], []
        cooling_entries = []  # {"pair", "seq", "request", "segments", "cycle"}
        open_cooling = {}  # pair -> its latest cooling entry, while a later Evacuation can still pause it

        def earliest_fit(intervals, start, duration):
            while True:
                conflict = [b for a, b in intervals if a < start + duration and start < b]
                if not conflict:
                    return start
                start = max(conflict)

        def cooling_segments(pair, start, duration):
            evacs = sorted((s, e) for s, e, p in evac_busy if p != pair) if enforce_evac_cool else []
            segments, remaining, cur = [], duration, start
            while remaining > 0:
                blocking = next((e for s, e in evacs if s <= cur < e), None)
                if blocking is not None:
                    cur = blocking
                    continue
                upcoming = next((s for s, e in evacs if s > cur), None)
                run_end = cur + remaining
                if upcoming is not None and upcoming < run_end:
                    run_end = upcoming
                segments.append((cur, run_end))
                remaining -= run_end - cur
                cur = run_end
            return segments

        def place_cooling(pair, start, others, limit):
            duration = phase_durations_by_pair[pair]["Cooling"]
            while start < limit:
                segs = cooling_segments(pair, start, duration)
                a, b = segs[0][0], segs[-1][1]
                if not any(x < b and a < y for x, y in others):
                    return segs
                start += 1
            return None

        def repack_coolings(evac_pair, s, e):
            """A newly booked Evacuation pauses any other pair's Cooling running
            across it. Re-place every still-open Cooling that ends after `s` (in the
            order they were booked) so they stay one-at-a-time around the longer
            pauses. Safe because those pairs' next phases aren't scheduled yet."""
            affected = [en for p, en in open_cooling.items() if p != evac_pair and en["segments"][-1][1] > s]
            if not any(a < e and s < b for en in affected for a, b in en["segments"]):
                return
            affected_ids = {id(en) for en in affected}
            fixed = [(en["segments"][0][0], en["segments"][-1][1]) for en in cooling_entries if id(en) not in affected_ids]
            for en in sorted(affected, key=lambda en: en["seq"]):
                segs = place_cooling(en["pair"], max(en["request"], en["segments"][0][0]), fixed, horizon) or en["segments"]
                cyc = en["cycle"]
                cyc["phases"] = [r for r in cyc["phases"] if r[0] != "Cooling"] + [["Cooling", a, b] for a, b in segs]
                en["segments"] = segs
                fixed.append((segs[0][0], segs[-1][1]))
                ready[en["pair"]] = segs[-1][1]

        active = set(pairs)
        seq = 0
        while active:
            pair = min(active, key=lambda p: (ready[p], last_chain_end[p], order[p]))
            open_cooling.pop(pair, None)
            t = ready[pair]
            if t >= total_minutes:
                active.discard(pair)
                continue
            step = next_step[pair]
            d = phase_durations_by_pair[pair]
            cooling_entry = None

            if step == "Adsorption":
                start = t
                if not free_adsorption_overlap and pair in (pair_b, pair_c):
                    blocking_pair = pair_c if pair == pair_b else pair_b
                    start = earliest_fit(ads_busy[blocking_pair], start, d["Adsorption"])
                end = start + d["Adsorption"]
                booked = [["Adsorption", start, end]]
                after = "Evacuation"

            elif step == "Evacuation":
                start = earliest_fit([(s, e) for s, e, _ in evac_busy], t, d["Evacuation"])
                end = start + d["Evacuation"]
                booked = [["Evacuation", start, end]]
                after = "Chain"

            elif step == "Chain":
                total_chain = sum(d[p] for p in CHAIN)
                start = earliest_fit(chain_busy, t, total_chain)
                end = start + total_chain
                booked, cursor = [], start
                for p in CHAIN:
                    if d[p] > 0:
                        booked.append([p, cursor, cursor + d[p]])
                    cursor += d[p]
                after = "Cooling"

            elif step == "Cooling":
                if d["Cooling"] > 0:
                    others = [(en["segments"][0][0], en["segments"][-1][1]) for en in cooling_entries]
                    segs = place_cooling(pair, t, others, total_minutes)
                    if segs is None:
                        active.discard(pair)
                        continue
                    start, end = segs[0][0], segs[-1][1]
                    booked = [["Cooling", a, b] for a, b in segs]
                    cooling_entry = {"pair": pair, "seq": seq, "request": t, "segments": segs}
                else:
                    start = end = t
                    booked = []
                after = "Repressurization"

            else:  # Repressurization
                start = earliest_fit(repress_busy, t, d["Repressurization"])
                end = start + d["Repressurization"]
                booked = [["Repressurization", start, end]]
                after = "Adsorption"

            if start >= total_minutes:
                active.discard(pair)
                continue

            if step == "Adsorption" or not cycles[pair]:
                cycles[pair].append({"phases": [], "full": step == "Adsorption", "done": False})
            cycle = cycles[pair][-1]
            cycle["phases"].extend(booked)
            seq += 1

            if step == "Adsorption":
                ads_busy[pair].append((start, end))
            elif step == "Evacuation":
                evac_busy.append((start, end, pair))
                if enforce_evac_cool and end > start:
                    repack_coolings(pair, start, end)
            elif step == "Chain":
                chain_busy.append((start, end))
                last_chain_end[pair] = end
            elif step == "Cooling" and cooling_entry is not None:
                cooling_entry["cycle"] = cycle
                cooling_entries.append(cooling_entry)
                open_cooling[pair] = cooling_entry
            elif step == "Repressurization":
                repress_busy.append((start, end))
                cycle["done"] = True

            ready[pair] = end
            next_step[pair] = after

        rows = []
        for pair in pairs:
            for cycle_no, c in enumerate(cycles[pair]):
                complete = c["done"] and c["full"] and all(e <= total_minutes for _, _, e in c["phases"])
                for phase, s, e in sorted(c["phases"], key=lambda r: r[1]):
                    if s < total_minutes:
                        rows.append({
                            "Pair": pair, "Phase": phase, "Start": s, "End": min(e, total_minutes),
                            "Cycle": cycle_no, "Complete": complete,
                        })
        return pd.DataFrame(rows, columns=["Pair", "Phase", "Start", "End", "Cycle", "Complete"])

    def _complete_cycles_and_yield_energy(schedule_df, pairs, yield_per_cycle, energy_per_cycle):
        """Shared logic for both configurations: count each pair's complete cycles
        from its scheduled rows, then multiply by its per-cycle rates."""
        rows = []
        for pair in pairs:
            pair_df = schedule_df[(schedule_df["Pair"] == pair) & schedule_df["Complete"]]
            cycles = int(pair_df["Cycle"].nunique())
            total_yield = cycles * yield_per_cycle.get(pair, 0.0)
            total_energy = cycles * energy_per_cycle.get(pair, 0.0)
            rows.append({
                "Pair": pair,
                "Complete Cycles": cycles,
                "Total Yield (kg CO2)": round(total_yield, 1),
                "Total Energy (kWh)": round(total_energy, 1),
                "kWh per kg CO2": round(total_energy / total_yield, 3) if total_yield > 0 else "—",
            })
        return pd.DataFrame(rows)

    def _pair_utilization(schedule_df, pairs, total_minutes):
        """Per-pair Active/Idle time and Utilisation %, same idea as tab3's
        compute_module_metrics but per Pair instead of per Module — this scheduler
        doesn't track individual modules within a pair."""
        rows = []
        for pair in pairs:
            pair_df = schedule_df[schedule_df["Pair"] == pair]
            if len(pair_df) == 0:
                rows.append({
                    "Pair": pair, "Active (min)": 0, "Idle (min)": total_minutes,
                    "Adsorption (min)": 0, "Utilisation %": 0, "Adsorption Utilisation %": 0,
                })
                continue
            active = int((pair_df["End"] - pair_df["Start"]).sum())
            ads_df = pair_df[pair_df["Phase"] == "Adsorption"]
            ads_time = int((ads_df["End"] - ads_df["Start"]).sum()) if len(ads_df) > 0 else 0
            idle = max(0, total_minutes - active)
            rows.append({
                "Pair": pair,
                "Active (min)": active,
                "Idle (min)": idle,
                "Adsorption (min)": ads_time,
                "Utilisation %": round(100 * active / total_minutes, 1) if total_minutes > 0 else 0,
                "Adsorption Utilisation %": round(100 * ads_time / total_minutes, 1) if total_minutes > 0 else 0,
            })
        return pd.DataFrame(rows)

    def _show_idle_breakdown(schedule_df, pairs):
        st.caption("What each pair's idle time is spent waiting for (adds up to its Idle (min) above).")
        st.dataframe(
            idle_breakdown_table(schedule_df, "Pair", {pair: [pair] for pair in pairs}, TOTAL_MINUTES_ADV),
            use_container_width=True, hide_index=True,
        )

    def _draw_idle_chart(utilization_df, title):
        """Bar chart of each pair's Idle (min) — the empty stretches where that
        pair is scheduled in no phase at all (neither Adsorption nor any step of
        the Desorption chain, Cooling, or Repressurization)."""
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(utilization_df["Pair"], utilization_df["Idle (min)"], color="#9AA5B1", edgecolor="black", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Pair")
        ax.set_ylabel("Idle (minutes)")
        ax.set_title(title)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    gantt_colors = {
        'Adsorption': '#4B9CD3',
        'Evacuation': '#FFB347',
        'NCG Purging': '#FFD700',
        'Heating': '#E97451',
        'CO2 Purging': '#90EE90',
        'Cooling': '#9370DB',
        'Repressurization': '#B0B0B0'
    }
    IDLE_GANTT_COLOR = '#D9D9D9'

    def _draw_advanced_gantt(schedule_df, total_minutes, title):
        fig, ax = plt.subplots(figsize=(14, 5))
        for _, row in schedule_df.iterrows():
            ax.barh(
                row["Pair"],
                row["End"] - row["Start"],
                left=row["Start"],
                color=gantt_colors.get(row["Phase"], "#888"),
                edgecolor="black"
            )
        # Idle segments: the gaps in each pair's own timeline where it isn't
        # scheduled in any phase at all (e.g. waiting on a shared resource) —
        # drawn on top of the same bars rather than replacing the separate
        # "Pair Utilisation & Idle Time" table/chart above, which still shows
        # the totals.
        for pair in schedule_df["Pair"].unique():
            pair_rows = schedule_df[schedule_df["Pair"] == pair].sort_values("Start")
            cursor = 0
            for _, row in pair_rows.iterrows():
                if row["Start"] > cursor:
                    ax.barh(pair, row["Start"] - cursor, left=cursor, color=IDLE_GANTT_COLOR, edgecolor="black", hatch="//")
                cursor = max(cursor, row["End"])
            if cursor < total_minutes:
                ax.barh(pair, total_minutes - cursor, left=cursor, color=IDLE_GANTT_COLOR, edgecolor="black", hatch="//")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Pairs")
        ax.set_title(title)
        ax.set_xlim(0, total_minutes)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        legend_labels = PHASES + ["Idle"]
        legend_colors = [gantt_colors[p] for p in PHASES] + [IDLE_GANTT_COLOR]
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend_colors],
            legend_labels,
            loc="upper right",
            fontsize=8
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def _run_and_display_adv_config(title, comparison_key, pairs, durations_by_pair,
                                    yield_per_cycle, energy_per_cycle, free_adsorption_overlap, rate_caption):
        """Schedule one Advanced Interleaved configuration and render its results
        (cycles, yield/energy, utilisation, idle breakdown, Gantt chart)."""
        schedule_df = run_advanced_interleaved(
            pairs, durations_by_pair, TOTAL_MINUTES_ADV, adv_enforce_evac_cool,
            free_adsorption_overlap=free_adsorption_overlap,
        )
        st.markdown(f"### Results: {title}")

        st.markdown("#### Complete Cycles")
        yield_energy_df = _complete_cycles_and_yield_energy(schedule_df, pairs, yield_per_cycle, energy_per_cycle)
        st.dataframe(yield_energy_df[["Pair", "Complete Cycles"]], use_container_width=True, hide_index=True)

        st.markdown("#### Yield & Energy Analysis")
        st.caption(rate_caption)
        st.dataframe(yield_energy_df, use_container_width=True, hide_index=True)
        ye_metric_col1, ye_metric_col2 = st.columns(2)
        with ye_metric_col1:
            st.metric("Plant Total Yield", f"{yield_energy_df['Total Yield (kg CO2)'].sum():.1f} kg CO2")
        with ye_metric_col2:
            st.metric("Plant Total Energy", f"{yield_energy_df['Total Energy (kWh)'].sum():.1f} kWh")

        st.markdown("#### Pair Utilisation & Idle Time")
        st.caption("Active = time in any phase. Idle = waiting/unused time. Utilisation % = Active ÷ Operating Period.")
        util_df = _pair_utilization(schedule_df, pairs, TOTAL_MINUTES_ADV)
        st.dataframe(util_df, use_container_width=True, hide_index=True)
        _draw_idle_chart(util_df, f"Idle Time per Pair — {title}")
        _show_idle_breakdown(schedule_df, pairs)

        # Publish totals for the Yield vs Cycles comparison tab. Only refreshes when
        # this configuration's "Generate" button is (re)clicked — see that tab's caption.
        st.session_state.setdefault("process_comparison", {})
        st.session_state["process_comparison"][comparison_key] = {
            "Run": st.session_state["app_run_id"],
            "Total Cycles": int(yield_energy_df["Complete Cycles"].sum()),
            "Total Yield (kg CO2)": float(yield_energy_df["Total Yield (kg CO2)"].sum()),
            "Total Energy (kWh)": float(yield_energy_df["Total Energy (kWh)"].sum()),
            "Avg Idle (min)": round(float(util_df["Idle (min)"].mean()), 1),
            "Avg Utilisation %": round(float(util_df["Utilisation %"].mean()), 1),
        }

        st.markdown("#### Gantt Chart")
        _draw_advanced_gantt(schedule_df, TOTAL_MINUTES_ADV, f"Advanced Interleaved Schedule - {title}")
        with st.expander(f"Schedule Data ({title})"):
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    # Each configuration's Generate button sits on its own tab, right under its
    # inputs; the button at the top runs all five.
    PER_PAIR_RATE = "Total Yield/Energy = Complete Cycles × the per-cycle rate for that pair."
    ADV_RUNS = [
        ("6-6-4", adv_results_slot_664, (
            "6-6-4 Configuration", "Advanced Interleaved (6-6-4)", PAIRS, PHASE_DURATIONS_BY_PAIR,
            PAIR_YIELD_PER_CYCLE, PAIR_ENERGY_PER_CYCLE, False,
            "Total Yield/Energy = Complete Cycles × the combined per-cycle rate of every module in that pair.",
        )),
        ("8-4-4", adv_results_slot_844, (
            "8-4-4 Configuration", "Advanced Interleaved (8-4-4)", PAIRS_8_4_4, PHASE_DURATIONS_BY_PAIR_844,
            PAIR_YIELD_PER_CYCLE_844, PAIR_ENERGY_PER_CYCLE_844, False, PER_PAIR_RATE,
        )),
        ("6-5-5", adv_results_slot_655, (
            "6-5-5 Configuration", "Advanced Interleaved (6-5-5)", PAIRS_6_5_5, PHASE_DURATIONS_BY_PAIR_655,
            PAIR_YIELD_PER_CYCLE_655, PAIR_ENERGY_PER_CYCLE_655, False, PER_PAIR_RATE,
        )),
        ("4-4-4-4", adv_results_slot_4x4, (
            "4-4-4-4 Configuration", "Advanced Interleaved (4-4-4-4)", PAIRS_4X4, PHASE_DURATIONS_BY_PAIR_4X4,
            PAIR_YIELD_PER_CYCLE_4X4, PAIR_ENERGY_PER_CYCLE_4X4, True, PER_PAIR_RATE,
        )),
        ("2-2-2-2 N3/N4", adv_results_slot_n3n4, (
            "2-2-2-2 (N3/N4) Configuration", "Advanced Interleaved (2-2-2-2 N3/N4)", PAIRS_N3N4, PHASE_DURATIONS_BY_PAIR_N3N4,
            PAIR_YIELD_PER_CYCLE_N3N4, PAIR_ENERGY_PER_CYCLE_N3N4, True, PER_PAIR_RATE,
        )),
    ]
    for adv_label, adv_slot, adv_args in ADV_RUNS:
        with adv_slot:
            adv_clicked = st.button(
                f"Generate {adv_label} Schedule", key=f"adv_generate_{adv_label}", type="primary",
            )
            if adv_clicked or adv_generate_all:
                _run_and_display_adv_config(*adv_args)

with tab5:
    # Cross-process comparison. Reads whatever tab3 (Concurrent/Interleaved) and
    # tab4 (Advanced Interleaved) last published to st.session_state["process_comparison"]
    # when their own "Generate" buttons were clicked — see the sidebar guide for why
    # this doesn't recompute live on every input change.
    st.subheader("Yield, Energy & Cycles Comparison")
    st.caption(
        "Shows the latest generated result only. Generating another configuration replaces it here; "
        "use \"Generate all\" in a tab to compare several configurations at once."
    )

    PROCESS_ORDER = [
        "Concurrent (8-8)", "Interleaved (8-8)", "Concurrent (4-4)", "Interleaved (4-4)",
        "Concurrent (2-2)", "Interleaved (2-2)", "Concurrent (6-6)", "Interleaved (6-6)",
        "Advanced Interleaved (6-6-4)", "Advanced Interleaved (8-4-4)",
        "Advanced Interleaved (6-5-5)",
        "Advanced Interleaved (4-4-4-4)",
        "Advanced Interleaved (2-2-2-2 N3/N4)",
    ]
    PROCESS_COLORS = {
        "Concurrent (8-8)": "#6C5CE7",
        "Interleaved (8-8)": "#E07C5E",
        "Concurrent (4-4)": "#8E7CF7",
        "Interleaved (4-4)": "#F0A382",
        "Concurrent (2-2)": "#5B4E9E",
        "Interleaved (2-2)": "#B0664C",
        "Concurrent (6-6)": "#4C3B99",
        "Interleaved (6-6)": "#C97A4A",
        "Advanced Interleaved (6-6-4)": "#2ECC71",
        "Advanced Interleaved (8-4-4)": "#F39C12",
        "Advanced Interleaved (6-5-5)": "#D35400",
        "Advanced Interleaved (4-4-4-4)": "#1ABC9C",
        "Advanced Interleaved (2-2-2-2 N3/N4)": "#2C7DA0",
    }
    comparison = st.session_state.get("process_comparison", {})
    # Only the latest run's results are shown: everything generated in the most
    # recent click (Concurrent + Interleaved for one Full Schedule Analysis
    # configuration, one Advanced Interleaved configuration, or a whole
    # "Generate all" batch). Earlier results drop out once something new is generated.
    latest_run = max((r.get("Run", 0) for r in comparison.values()), default=None)
    available_processes = [p for p in PROCESS_ORDER if p in comparison and comparison[p].get("Run", 0) == latest_run]

    if not available_processes:
        st.warning("No results yet — generate a schedule in Full Schedule Analysis and/or Advanced Interleaved first.")
    else:
        # Bottom line first: filled in at the end of this tab, once every cost is known.
        bottom_line_box = st.container()

        with st.expander("Assumptions: electricity and water tariffs, water use, capacity factor"):
            opex_col1, opex_col2, opex_col3 = st.columns(3)
            with opex_col1:
                cost_per_kwh = st.number_input(
                    "Electricity cost ($ per kWh)", min_value=0.0, value=0.18, step=0.01,
                    help="Used below to convert each process's Total Energy into an OPEX line. "
                         "Adjust to your actual electricity tariff.",
                    key="cost_per_kwh_tab5",
                )
            with opex_col2:
                water_per_cycle_l = st.number_input(
                    "Water use per cycle (litres)", min_value=0.0, value=275.0, step=1.0,
                    help="Litres of water consumed per completed cycle — applied to every "
                         "configuration's Total Cycles as a second OPEX line alongside energy.",
                    key="water_per_cycle_tab5",
                )
            with opex_col3:
                cost_per_m3 = st.number_input(
                    "Water cost ($ per m³)", min_value=0.0, value=4.62, step=0.01,
                    help="Cost per cubic metre of water (1 m³ = 1000 L). Adjust to your actual "
                         "water tariff.",
                    key="cost_per_m3_tab5",
                )
                cost_per_litre = cost_per_m3 / 1000.0
            capacity_factor = st.number_input(
                "Capacity factor", min_value=0.0, max_value=1.0, value=0.88, step=0.01,
                format="%.2f", key="capacity_factor_tab5",
                help="Fraction of the year the plant effectively runs at the daily-schedule rate (0.88 = 88%).",
            )

        # Levelized Cost = (OPEX from energy + water + amortized CAPEX) ÷ Total Yield in tonnes.
        # CAPEX: each process is mapped to the CAPEX line item(s) for the physical Nelion(s)
        # it actually runs (see the CAPEX section below for the line items and
        # PAIR_PHYSICAL_MODULES_* / REPRESENTS_* constants throughout this file for each
        # config's Nelion composition): 4-4, 2-2 (sourced entirely from N3-M2n4) and
        # 2-2-2-2 (N3/N4) run only Nelion 3 & 4; 6-6 runs Nelion 2, 3 & 4; 8-8 and the four
        # full-16-module Advanced Interleaved configs (6-6-4, 8-4-4, 6-5-5, 4-4-4-4) run the
        # whole plant, Utility skid included. The CAPEX table/amortization input are defined
        # in the CAPEX section further down this tab, so their current values are read here
        # from st.session_state by key. A plain widget (like the amortization number_input)
        # stores its resulting value directly under its key, but st.data_editor's key instead
        # stores an edit-delta dict ({"edited_rows", "added_rows", "deleted_rows"}), not the
        # resulting DataFrame — so that delta is replayed onto the same default rows the CAPEX
        # table itself starts from, to reconstruct the current values before the first render.
        default_capex_rows = [
            {"Item": "Utility skid + Plant", "CAPEX ($)": 374000.00},
            {"Item": "Nelion 1 + Monolith", "CAPEX ($)": 249451.05},
            {"Item": "Nelion 2 + Monolith", "CAPEX ($)": 186800.00},
            {"Item": "Nelion 3 & 4 + Monolith", "CAPEX ($)": 390900.00},
        ]
        capex_edit_state = st.session_state.get("capex_table_tab5")
        capex_rows = [dict(row) for row in default_capex_rows]
        if isinstance(capex_edit_state, dict):
            for row_idx, changes in capex_edit_state.get("edited_rows", {}).items():
                capex_rows[int(row_idx)].update(changes)
        capex_lookup = {row["Item"]: row["CAPEX ($)"] for row in capex_rows}
        amortization_years = st.session_state.get("capex_amortization_years", 10)
        FULL_PLANT_CAPEX_ITEMS = [
            "Utility skid + Plant", "Nelion 1 + Monolith", "Nelion 2 + Monolith", "Nelion 3 & 4 + Monolith",
        ]
        CAPEX_ITEMS_BY_PROCESS = {
            "Concurrent (8-8)": FULL_PLANT_CAPEX_ITEMS,
            "Interleaved (8-8)": FULL_PLANT_CAPEX_ITEMS,
            "Concurrent (4-4)": ["Nelion 3 & 4 + Monolith"],
            "Interleaved (4-4)": ["Nelion 3 & 4 + Monolith"],
            "Concurrent (2-2)": ["Nelion 3 & 4 + Monolith"],
            "Interleaved (2-2)": ["Nelion 3 & 4 + Monolith"],
            "Concurrent (6-6)": ["Nelion 2 + Monolith", "Nelion 3 & 4 + Monolith"],
            "Interleaved (6-6)": ["Nelion 2 + Monolith", "Nelion 3 & 4 + Monolith"],
            "Advanced Interleaved (6-6-4)": FULL_PLANT_CAPEX_ITEMS,
            "Advanced Interleaved (8-4-4)": FULL_PLANT_CAPEX_ITEMS,
            "Advanced Interleaved (6-5-5)": FULL_PLANT_CAPEX_ITEMS,
            "Advanced Interleaved (4-4-4-4)": FULL_PLANT_CAPEX_ITEMS,
            "Advanced Interleaved (2-2-2-2 N3/N4)": ["Nelion 3 & 4 + Monolith"],
        }
        annual_capex_map = {
            p: sum(capex_lookup.get(item, 0.0) for item in CAPEX_ITEMS_BY_PROCESS.get(p, [])) / amortization_years
            for p in available_processes
        }
        water_opex_map = {
            p: comparison[p]["Total Cycles"] * water_per_cycle_l * cost_per_litre
            for p in available_processes
        }
        # Other Operational Costs (Maintenance, Casual Labour, Security, WW Disposable,
        # Solar Lease): a flat quarterly total that cuts across every configuration, the
        # same way CAPEX does — it doesn't scale with cycles, yield, or energy use. The
        # table is defined in the section further down this tab; its current values are
        # reconstructed here the same way as the CAPEX table above.
        default_misc_opex_rows = [
            {"Item": "Maintenance", "Cost per Quarter ($)": 7095.00},
            {"Item": "Casual Labour", "Cost per Quarter ($)": 308.00},
            {"Item": "Security", "Cost per Quarter ($)": 2900.00},
            {"Item": "WW Disposable", "Cost per Quarter ($)": 2006.00},
            {"Item": "Solar Lease", "Cost per Quarter ($)": 2854.00},
        ]
        misc_opex_edit_state = st.session_state.get("misc_opex_table_tab5")
        misc_opex_rows = [dict(row) for row in default_misc_opex_rows]
        if isinstance(misc_opex_edit_state, dict):
            for row_idx, changes in misc_opex_edit_state.get("edited_rows", {}).items():
                misc_opex_rows[int(row_idx)].update(changes)
        misc_opex_quarterly_total = sum(row["Cost per Quarter ($)"] for row in misc_opex_rows)
        misc_opex_daily = misc_opex_quarterly_total / 90
        misc_opex_annual = misc_opex_daily * 365
        levelized_cost_map = {
            p: (
                (
                    comparison[p]["Total Energy (kWh)"] * cost_per_kwh
                    + water_opex_map[p]
                    + annual_capex_map[p] / 365
                    + misc_opex_daily
                )
                / (comparison[p]["Total Yield (kg CO2)"] / 1000)
            ) if comparison[p]["Total Yield (kg CO2)"] > 0 else None
            for p in available_processes
        }

        st.markdown("### Levelized Cost of Capture (annual projection)")
        annual_energy_opex_map = {
            p: comparison[p]["Total Energy (kWh)"] * 365 * capacity_factor * cost_per_kwh
            for p in available_processes
        }
        annual_water_opex_map = {
            p: comparison[p]["Total Cycles"] * 365 * capacity_factor * water_per_cycle_l * cost_per_litre
            for p in available_processes
        }
        annual_yield_tonnes_map = {
            p: comparison[p]["Total Yield (kg CO2)"] * 365 * capacity_factor / 1000
            for p in available_processes
        }
        levelized_cost_annual_map = {
            p: (
                (
                    annual_energy_opex_map[p] + annual_water_opex_map[p] + annual_capex_map[p]
                    + misc_opex_annual
                )
                / annual_yield_tonnes_map[p]
            ) if annual_yield_tonnes_map[p] > 0 else None
            for p in available_processes
        }
        annual_cost_df = pd.DataFrame([
            {
                "Process": p,
                "Annual Energy Cost ($)": round(annual_energy_opex_map[p], 2),
                "Annual Water Cost ($)": round(annual_water_opex_map[p], 2),
                "Annual CAPEX Amortization ($)": round(annual_capex_map[p], 2),
                "Annual Other OPEX ($)": round(misc_opex_annual, 2),
                "Levelized Cost ($/tonne CO2)": (
                    round(levelized_cost_annual_map[p], 2) if levelized_cost_annual_map[p] is not None else "—"
                ),
            }
            for p in available_processes
        ])
        st.dataframe(annual_cost_df, use_container_width=True, hide_index=True)
        st.caption(
            "Levelized Cost = (Annual Energy Cost + Annual Water Cost + Annual CAPEX Amortization + "
            "Annual Other OPEX) ÷ Annual Yield in tonnes, with CAPEX amortized over "
            f"{amortization_years:g} years and Other OPEX (Maintenance, Casual Labour, Security, WW "
            "Disposable, Solar Lease) annualized as a flat quarterly total ÷ 90 days × 365 — see the "
            "CAPEX and Other Operational Costs sections below for the line items. This differs "
            "slightly from the daily table's figure since CAPEX and Other OPEX amortization don't "
            "scale with the capacity factor the way Energy, Water and Yield do."
        )
        fig_ann_lc, ax_ann_lc = plt.subplots(figsize=(8, 5))
        ann_lc_values = [levelized_cost_annual_map[p] or 0.0 for p in available_processes]
        bars_ann_lc = ax_ann_lc.bar(
            available_processes, ann_lc_values,
            color=[PROCESS_COLORS[p] for p in available_processes], edgecolor="black", alpha=0.9,
        )
        for bar in bars_ann_lc:
            ax_ann_lc.annotate(
                f"{bar.get_height():,.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10,
            )
        ax_ann_lc.set_title("Levelized Cost ($/tonne CO2) by Process — Annual Projection")
        ax_ann_lc.set_ylabel("Levelized Cost ($/tonne CO2)")
        plt.setp(ax_ann_lc.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_ann_lc)
        plt.close(fig_ann_lc)

        st.markdown("#### Cost Driver Breakdown")
        st.caption(
            "Share of Energy, Water, CAPEX and Other OPEX in the Annual Levelized Cost for a "
            "selected process, expressed in $/tonne CO2 (each component's annual cost ÷ that "
            "process's Annual Yield in tonnes)."
        )
        default_breakdown_idx = (
            available_processes.index("Concurrent (4-4)") if "Concurrent (4-4)" in available_processes else 0
        )
        cost_breakdown_process = st.selectbox(
            "Process for cost breakdown", available_processes, index=default_breakdown_idx,
            key="cost_breakdown_process_tab5",
        )
        breakdown_yield_tonnes = annual_yield_tonnes_map[cost_breakdown_process]
        breakdown_labels = ["Energy", "Water", "CAPEX", "Other OPEX"]
        breakdown_values = [
            (annual_energy_opex_map[cost_breakdown_process] / breakdown_yield_tonnes) if breakdown_yield_tonnes > 0 else 0.0,
            (annual_water_opex_map[cost_breakdown_process] / breakdown_yield_tonnes) if breakdown_yield_tonnes > 0 else 0.0,
            (annual_capex_map[cost_breakdown_process] / breakdown_yield_tonnes) if breakdown_yield_tonnes > 0 else 0.0,
            (misc_opex_annual / breakdown_yield_tonnes) if breakdown_yield_tonnes > 0 else 0.0,
        ]
        # Fixed categorical order/hues (blue, orange, aqua, yellow) — kept stable across
        # processes so a given cost driver always maps to the same color.
        breakdown_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        breakdown_text_colors = ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"]
        if sum(breakdown_values) > 0:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            wedges, _, autotexts = ax_pie.pie(
                breakdown_values,
                colors=breakdown_colors,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.75,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            for autotext, text_color in zip(autotexts, breakdown_text_colors):
                autotext.set_color(text_color)
                autotext.set_fontsize(11)
                autotext.set_fontweight("bold")
            ax_pie.legend(
                wedges,
                [f"{label} (${value:,.2f}/tonne)" for label, value in zip(breakdown_labels, breakdown_values)],
                title="Cost driver", loc="center left", bbox_to_anchor=(1, 0.5),
            )
            ax_pie.set_title(f"Annual Levelized Cost Drivers ($/tonne CO2) — {cost_breakdown_process}")
            ax_pie.axis("equal")
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)
        else:
            st.info(f"No cost data to break down yet for {cost_breakdown_process}.")

        st.markdown("### Annual Yield & Energy")
        st.caption(
            "Scales each process's daily totals (from the schedule above) up to a full year: "
            "Daily total × 365 days × capacity factor. The capacity factor accounts for planned "
            "and unplanned downtime, since the plant won't run at the daily-schedule rate every "
            "day of the year."
        )

        st.markdown("#### Annual Yield")
        annual_yield_df = pd.DataFrame([
            {
                "Process": p,
                "Annual Yield (tonnes CO2)": round(
                    comparison[p]["Total Yield (kg CO2)"] * 365 * capacity_factor / 1000, 1
                ),
            }
            for p in available_processes
        ])
        st.dataframe(annual_yield_df, use_container_width=True, hide_index=True)
        fig_ann_yield, ax_ann_yield = plt.subplots(figsize=(8, 5))
        bars_ann_yield = ax_ann_yield.bar(
            available_processes, annual_yield_df["Annual Yield (tonnes CO2)"],
            color=[PROCESS_COLORS[p] for p in available_processes], edgecolor="black", alpha=0.9,
        )
        for bar in bars_ann_yield:
            ax_ann_yield.annotate(
                f"{bar.get_height():,.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10,
            )
        ax_ann_yield.set_title("Annual Yield (tonnes CO2) by Process")
        ax_ann_yield.set_ylabel("Annual Yield (tonnes CO2)")
        plt.setp(ax_ann_yield.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_ann_yield)
        plt.close(fig_ann_yield)

        st.markdown("#### Annual Energy")
        annual_energy_df = pd.DataFrame([
            {
                "Process": p,
                "Annual Energy (MWh)": round(
                    comparison[p]["Total Energy (kWh)"] * 365 * capacity_factor / 1000, 1
                ),
            }
            for p in available_processes
        ])
        st.dataframe(annual_energy_df, use_container_width=True, hide_index=True)
        fig_ann_energy, ax_ann_energy = plt.subplots(figsize=(8, 5))
        bars_ann_energy = ax_ann_energy.bar(
            available_processes, annual_energy_df["Annual Energy (MWh)"],
            color=[PROCESS_COLORS[p] for p in available_processes], edgecolor="black", alpha=0.9,
        )
        for bar in bars_ann_energy:
            ax_ann_energy.annotate(
                f"{bar.get_height():,.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10,
            )
        ax_ann_energy.set_title("Annual Energy (MWh) by Process")
        ax_ann_energy.set_ylabel("Annual Energy (MWh)")
        plt.setp(ax_ann_energy.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_ann_energy)
        plt.close(fig_ann_energy)

        st.markdown("### Daily Schedule Comparison")
        st.caption("Totals for one operating day, straight from each schedule.")
        comparison_df = pd.DataFrame([
            {
                "Process": p,
                "Total Cycles": comparison[p]["Total Cycles"],
                "Total Yield (kg CO2)": round(comparison[p]["Total Yield (kg CO2)"], 1),
                "Total Energy (kWh)": round(comparison[p]["Total Energy (kWh)"], 1),
                "kWh per kg CO2": (
                    round(comparison[p]["Total Energy (kWh)"] / comparison[p]["Total Yield (kg CO2)"], 3)
                    if comparison[p]["Total Yield (kg CO2)"] > 0 else "—"
                ),
                "Water Cost ($)": round(water_opex_map[p], 2),
                "Other OPEX ($)": round(misc_opex_daily, 2),
                "Levelized Cost ($/tonne CO2)": (
                    round(levelized_cost_map[p], 2) if levelized_cost_map[p] is not None else "—"
                ),
            }
            for p in available_processes
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.caption(
            "Levelized Cost = (Energy cost + Water cost + amortized CAPEX apportioned to one day + "
            "Other OPEX apportioned to one day) ÷ Total Yield in tonnes. Water cost = Total Cycles × "
            f"{water_per_cycle_l:g} L/cycle × water cost per litre. CAPEX is amortized over "
            f"{amortization_years:g} years (Annual CAPEX ÷ 365 for the daily share). Other OPEX "
            "(Maintenance, Casual Labour, Security, WW Disposable, Solar Lease) is a flat quarterly "
            "total ÷ 90 days, the same for every process — see the CAPEX and Other Operational Costs "
            "sections below to edit the line items, the amortization period, and which processes draw "
            "on which CAPEX lines."
        )

        st.markdown("#### Compare a parameter across processes")
        selected_metric = st.selectbox(
            "Parameter",
            ["Total Cycles", "Total Yield (kg CO2)", "Total Energy (kWh)", "Levelized Cost ($/tonne CO2)"],
            key="comparison_metric",
        )
        if selected_metric == "Levelized Cost ($/tonne CO2)":
            values = [levelized_cost_map[p] or 0.0 for p in available_processes]
        else:
            values = [comparison[p][selected_metric] for p in available_processes]
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        bars = ax_bar.bar(
            available_processes, values,
            color=[PROCESS_COLORS[p] for p in available_processes],
            edgecolor="black", alpha=0.9,
        )
        for bar in bars:
            ax_bar.annotate(
                f"{bar.get_height():,.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=10,
            )
        ax_bar.set_title(f"{selected_metric} by Process")
        ax_bar.set_ylabel(selected_metric)
        ax_bar.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.markdown("### Idle Time Summary")
        st.caption(
            "Average Idle (min) and Utilisation % across the day for each process — per module for "
            "Full Schedule Analysis, per pair for Advanced Interleaved (that's the finest-grained unit "
            "each scheduler tracks). Idle = Operating Period − Active (any phase); Cooling and "
            "Repressurization count as busy, not idle."
        )
        idle_available = [p for p in available_processes if "Avg Idle (min)" in comparison[p]]
        idle_stale = [p for p in available_processes if "Avg Idle (min)" not in comparison[p]]
        if idle_stale:
            st.info(
                "Re-click Generate for: " + ", ".join(idle_stale) +
                " to include them here (idle time wasn't recorded the last time they ran)."
            )
        if idle_available:
            idle_df = pd.DataFrame([
                {
                    "Process": p,
                    "Avg Idle (min)": comparison[p]["Avg Idle (min)"],
                    "Avg Utilisation %": comparison[p]["Avg Utilisation %"],
                }
                for p in idle_available
            ])
            st.dataframe(idle_df, use_container_width=True, hide_index=True)

            fig_idle, ax_idle = plt.subplots(figsize=(8, 5))
            bars_idle = ax_idle.bar(
                idle_available, [comparison[p]["Avg Idle (min)"] for p in idle_available],
                color=[PROCESS_COLORS[p] for p in idle_available], edgecolor="black", alpha=0.9,
            )
            for bar in bars_idle:
                ax_idle.annotate(
                    f"{bar.get_height():,.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10,
                )
            ax_idle.set_title("Average Idle Time (min) by Process")
            ax_idle.set_ylabel("Avg Idle (min)")
            plt.setp(ax_idle.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_idle)
            plt.close(fig_idle)

        st.markdown("### CAPEX")
        st.caption(
            "Capital expenditure by asset, already folded into the Levelized Cost figures above — "
            "edit the values below or the amortization period and both the daily and annual tables "
            "update."
        )
        default_capex_df = pd.DataFrame([
            {"Item": "Utility skid + Plant", "CAPEX ($)": 374000.00},
            {"Item": "Nelion 1 + Monolith", "CAPEX ($)": 249451.05},
            {"Item": "Nelion 2 + Monolith", "CAPEX ($)": 186800.00},
            {"Item": "Nelion 3 & 4 + Monolith", "CAPEX ($)": 390900.00},
        ])
        capex_df = st.data_editor(
            default_capex_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="capex_table_tab5",
            column_config={
                "CAPEX ($)": st.column_config.NumberColumn("CAPEX ($)", format="$%.2f"),
            },
        )
        st.caption(f"Total CAPEX: ${capex_df['CAPEX ($)'].sum():,.2f}")
        st.session_state["capex_table"] = capex_df

        st.number_input(
            "CAPEX amortization period (years)", min_value=1, value=10, step=1,
            key="capex_amortization_years",
            help="Each process's CAPEX line item(s) are divided by this many years to get an "
                 "Annual CAPEX Amortization, which feeds into Levelized Cost w/ CAPEX above.",
        )
        st.caption(
            "CAPEX line(s) used per process — picked by which physical Nelion(s) that process's "
            "modules actually run (see each config's \"Physical Module Pairing\" table): "
            "Concurrent/Interleaved (4-4), (2-2) and Advanced Interleaved (2-2-2-2 N3/N4) use "
            "Nelion 3 & 4 + Monolith only; Concurrent/Interleaved (6-6) use Nelion 2 + Monolith and "
            "Nelion 3 & 4 + Monolith; Concurrent/Interleaved (8-8) and Advanced Interleaved "
            "(6-6-4, 8-4-4, 6-5-5, 4-4-4-4) run the full 16-module plant, so they use all four lines "
            "including Utility skid + Plant."
        )

        st.markdown("### Other Operational Costs")
        st.caption(
            "Non-energy, non-water operational costs billed quarterly, already folded into the "
            "Levelized Cost figures above as a flat daily/annual share applied equally to every "
            "configuration — edit the values below and both the daily and annual tables update."
        )
        default_misc_opex_df = pd.DataFrame(default_misc_opex_rows)
        misc_opex_df = st.data_editor(
            default_misc_opex_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="misc_opex_table_tab5",
            column_config={
                "Cost per Quarter ($)": st.column_config.NumberColumn("Cost per Quarter ($)", format="$%.2f"),
            },
        )
        st.caption(
            f"Total per quarter: ${misc_opex_df['Cost per Quarter ($)'].sum():,.2f} → "
            f"${misc_opex_daily:,.2f}/day (÷ 90) → ${misc_opex_annual:,.2f}/year (× 365), "
            "the same for every configuration."
        )

        # === Bottom line (rendered at the top of the tab via bottom_line_box) ===
        with bottom_line_box:
            ranked = sorted(
                [p for p in available_processes if levelized_cost_annual_map[p] is not None],
                key=lambda p: levelized_cost_annual_map[p],
            )
            # available_processes is already limited to the latest run (see the top of this tab).
            if not ranked:
                st.markdown("### Bottom Line")
                st.info(
                    "Levelized cost of capture can't be calculated for the latest result ("
                    + ", ".join(available_processes) + "): its yield is 0. Check the Yield per Cycle "
                    "values in that configuration's tables."
                )
            else:
                best = ranked[0]
                best_lc = levelized_cost_annual_map[best]
                # "$" is written as &#36; throughout: Streamlit's markdown treats text
                # between two "$" signs as a LaTeX formula.
                best_name = html_lib.escape(best)
                if len(ranked) > 1:
                    runner_up = ranked[1]
                    gap = levelized_cost_annual_map[runner_up] - best_lc
                    gap_pct = 100 * gap / levelized_cost_annual_map[runner_up]
                    verdict = (
                        f"<b>{best_name}</b> is the cheaper option, "
                        f"<b>&#36;{gap:,.0f}/t ({gap_pct:.0f}%) below</b> {html_lib.escape(runner_up)}."
                    )
                else:
                    verdict = f"<b>{best_name}</b>: latest generated result."
                idle = comparison[best].get("Avg Idle (min)")
                tiles = [
                    ("Annual yield", f"{annual_yield_tonnes_map[best]:,.1f}", "t CO2 / year"),
                    ("Cycles per day", f"{comparison[best]['Total Cycles']}", "complete cycles"),
                    ("Avg idle per day", f"{idle:,.0f}" if idle is not None else "—", "minutes per unit"),
                ]
                tiles_html = "".join(
                    f'<div class="bl-tile"><div class="bl-label">{label}</div>'
                    f'<div class="bl-value">{value}</div><div class="bl-unit">{unit}</div></div>'
                    for label, value, unit in tiles
                )
                st.markdown(
                    f"""
<style>
.bl-card {{ background: linear-gradient(135deg, #0b3d2e 0%, #11694f 100%); color: #ffffff;
           border-radius: 16px; padding: 24px 28px; margin: 4px 0 18px 0;
           box-shadow: 0 6px 18px rgba(11, 61, 46, 0.25); }}
.bl-eyebrow {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
              color: #9be7c4; margin-bottom: 14px; }}
.bl-hero {{ display: flex; flex-wrap: wrap; align-items: flex-end; gap: 6px 28px; margin-bottom: 10px; }}
.bl-hero-value {{ font-size: 3.4rem; font-weight: 800; line-height: 1; color: #ffffff; }}
.bl-hero-unit {{ font-size: 1.1rem; font-weight: 600; color: #c8f2df; }}
.bl-hero-label {{ font-size: 1rem; font-weight: 600; color: #9be7c4; }}
.bl-verdict {{ font-size: 1.05rem; color: #e6fff4; margin: 8px 0 20px 0; }}
.bl-verdict b {{ color: #ffffff; }}
.bl-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.bl-tile {{ background: rgba(255, 255, 255, 0.10); border: 1px solid rgba(255, 255, 255, 0.18);
           border-radius: 12px; padding: 14px 16px; }}
.bl-label {{ font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9be7c4; }}
.bl-value {{ font-size: 2rem; font-weight: 800; color: #ffffff; line-height: 1.2; margin-top: 4px; }}
.bl-unit {{ font-size: 0.85rem; color: #c8f2df; }}
</style>
<div class="bl-card">
  <div class="bl-eyebrow">Latest result &middot; Levelized cost of capture</div>
  <div class="bl-hero">
    <div>
      <div class="bl-hero-label">{best_name}</div>
      <span class="bl-hero-value">&#36;{best_lc:,.0f}</span>
      <span class="bl-hero-unit">per tonne CO2</span>
    </div>
  </div>
  <div class="bl-verdict">{verdict}</div>
  <div class="bl-tiles">{tiles_html}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                ranking_df = pd.DataFrame([
                    {
                        "Rank": rank,
                        "Process": p,
                        "Levelized Cost ($/tonne CO2)": round(levelized_cost_annual_map[p], 0),
                        "Annual Yield (t CO2)": round(annual_yield_tonnes_map[p], 1),
                        "Cycles per Day": comparison[p]["Total Cycles"],
                        "Avg Idle (min)": comparison[p].get("Avg Idle (min)", "—"),
                    }
                    for rank, p in enumerate(ranked, start=1)
                ])
                st.dataframe(
                    ranking_df.style.apply(
                        lambda row: ["background-color: #d8f3e6; color: #0b3d2e; font-weight: 700"
                                     if row["Process"] == best else "" for _ in row],
                        axis=1,
                    ),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Levelized Cost ($/tonne CO2)": st.column_config.NumberColumn(format="dollar"),
                    },
                )
                st.caption(
                    f"Latest generated result, ranked by levelized cost of capture (annual projection at a "
                    f"{capacity_factor:.0%} capacity factor). Assumptions and the full cost breakdown are below."
                )