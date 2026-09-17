# === Streamlit App: Interleaved Desorption Scheduling ===
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Real plant cycle log (Cycle #, Module, Start Time, DES CO2 (kg), DES Hours,
# BAG CO2 (kg), DES Vol Cap, eTotal kWh) used to seed realistic default
# Energy/Yield-per-cycle values in the Advanced Interleaved tab, instead of 0.0.
PLANT_CYCLES_CSV = os.path.join(os.path.dirname(__file__), "carbonnest_plant_cycles.csv")

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

# Alternative 8-4-4 pairing (vs. the 6-6-4 Group A/B/C above), for side-by-side
# comparison in Advanced Interleaved. Phase durations AND Energy/Yield both use
# one Module average per pair (no weighted formula, unlike PAIR_YIELD_FORMULA):
# the 8-module pair from N1N2N3-M1n3, both 4-module pairs from N1N2-M1n3.
PAIRS_8_4_4 = ["Pair (8)", "Pair (4a)", "Pair (4b)"]
PAIR_PLANT_MODULE_8_4_4 = {
    "Pair (8)": "N1N2N3-M1n3",
    "Pair (4a)": "N1N2-M1n3",
    "Pair (4b)": "N1N2-M1n3",
}

def _plant_cycles_df():
    """Load the plant cycle log, or None if it's missing/malformed."""
    try:
        return pd.read_csv(PLANT_CYCLES_CSV)
    except (FileNotFoundError, KeyError, ValueError):
        return None

def _module_average(plant_cycles_df, module):
    """(avg eTotal kWh, avg DES CO2 kg) for one Module value's rows, or (0.0, 0.0)
    if there are none (or the log failed to load)."""
    if plant_cycles_df is None:
        return 0.0, 0.0
    subset = plant_cycles_df[plant_cycles_df["Module"] == module]
    if len(subset) == 0:
        return 0.0, 0.0
    return (
        round(float(subset["eTotal kWh"].mean()), 2),
        round(float(subset["DES CO2 (kg)"].mean()), 2),
    )

def load_plant_cycle_defaults_by_pair(pairs):
    """Return {pair: (avg eTotal kWh, avg DES CO2 kg)}. Energy per Cycle comes from
    each pair's single matching Module (PAIR_PLANT_MODULE — same source as phase
    durations). Yield per Cycle instead comes from a weighted sum of one or more
    Module averages per PAIR_YIELD_FORMULA. A pair missing from either mapping
    gets 0.0 for that value."""
    plant_cycles_df = _plant_cycles_df()
    defaults = {}
    for pair in pairs:
        energy, _ = _module_average(plant_cycles_df, PAIR_PLANT_MODULE.get(pair))
        yield_total = 0.0
        for module, weight in PAIR_YIELD_FORMULA.get(pair, []):
            _, co2 = _module_average(plant_cycles_df, module)
            yield_total += weight * co2
        defaults[pair] = (round(energy, 2), round(yield_total, 2))
    return defaults

def load_plant_cycle_defaults_simple(pairs, pair_module_map):
    """Return {pair: (avg eTotal kWh, avg DES CO2 kg)} using one Module average per
    pair for BOTH Energy and Yield equally — no weighted formula, unlike
    load_plant_cycle_defaults_by_pair. Used for the 8-4-4 comparison config."""
    plant_cycles_df = _plant_cycles_df()
    return {pair: _module_average(plant_cycles_df, pair_module_map.get(pair)) for pair in pairs}

# Full Schedule Analysis's two pairs (Group A, Group B; 8 modules each) don't map to
# a single Module value in the plant log the way tab4's pairs do, so both pairs
# instead share the SUM of two module-type averages: the 3-Nelion desorption
# combination (N1N2N3-M1n3) plus the Nelion-3 M2/M4 combination (N3-M2n4).
TAB3_PLANT_MODULES = ["N1N2N3-M1n3", "N3-M2n4"]

def load_plant_cycle_defaults_tab3():
    """Return (avg eTotal kWh, avg DES CO2 kg) summed across TAB3_PLANT_MODULES,
    used as the shared default for both Group A and Group B."""
    plant_cycles_df = _plant_cycles_df()
    energy_total = 0.0
    yield_total = 0.0
    for module in TAB3_PLANT_MODULES:
        energy, co2 = _module_average(plant_cycles_df, module)
        energy_total += energy
        yield_total += co2
    return round(energy_total, 2), round(yield_total, 2)

# Real per-stage duration log (Cycle #, Module, Adsorption/Evacuation/Preheating/
# NCGs Purging/NCGs Purging-Heating/Heating-CO2 Purging/CO2 Purging/Cooling, all in
# minutes) used to seed realistic default phase durations in Advanced Interleaved.
STAGE_TIMESTAMPS_CSV = os.path.join(os.path.dirname(__file__), "carbonnest_stage_timestamps.csv")

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

# The log has no Repressurization column at all, so that phase always keeps
# whatever value is already in the table (originally defaulted to 5 minutes).
def load_stage_duration_defaults_by_pair(pairs, pair_module_map=None):
    """Return {pair: {phase: avg minutes}} for the 6 phases in STAGE_PHASE_COLUMNS,
    sourced from each pair's matching Module rows (pair_module_map, default
    PAIR_PLANT_MODULE) in carbonnest_stage_timestamps.csv. A pair with a missing
    file/module/phase simply has no entry for it — callers should fall back to
    their own default."""
    pair_module_map = pair_module_map if pair_module_map is not None else PAIR_PLANT_MODULE
    try:
        stage_df = pd.read_csv(STAGE_TIMESTAMPS_CSV)
    except (FileNotFoundError, KeyError, ValueError):
        return {pair: {} for pair in pairs}

    defaults = {}
    for pair in pairs:
        module = pair_module_map.get(pair)
        subset = stage_df[stage_df["Module"] == module] if module else stage_df.iloc[0:0]
        if len(subset) == 0:
            defaults[pair] = {}
            continue
        defaults[pair] = {
            phase: round(float(subset[cols].sum(axis=1).mean()), 1)
            for phase, cols in STAGE_PHASE_COLUMNS.items()
        }
    return defaults

# Phase durations for Full Schedule Analysis's two pairs are sourced directly from
# this one module combination's own per-phase averages — no summing with N3-M2n4
# (that summing is still used for the Energy/Yield table above, which is a
# separate, deliberately different sourcing choice).
TAB3_PHASE_DURATION_MODULE = "N1N2N3-M1n3"

def load_stage_duration_defaults_tab3():
    """Return {phase: avg minutes}, the shared phase-duration default for both
    Group A and Group B in Full Schedule Analysis, sourced directly from
    TAB3_PHASE_DURATION_MODULE's own per-phase averages (no summing). Falls back
    to CORE_PHASE_FALLBACK_DURATIONS if the log is missing or has no matching rows."""
    try:
        stage_df = pd.read_csv(STAGE_TIMESTAMPS_CSV)
    except (FileNotFoundError, KeyError, ValueError):
        return dict(CORE_PHASE_FALLBACK_DURATIONS)

    subset = stage_df[stage_df["Module"] == TAB3_PHASE_DURATION_MODULE]
    if len(subset) == 0:
        return dict(CORE_PHASE_FALLBACK_DURATIONS)

    return {
        phase: round(float(subset[cols].sum(axis=1).mean()), 1)
        for phase, cols in STAGE_PHASE_COLUMNS.items()
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
**Tab 1: General Test**  
Customize:
- **Adsorption and Desorption durations**
- **Resource limits** – define how many modules can run each phase concurrently

**Tab 2: M2&M4 + LRVP**  
Stage-based visualization for paired modules.

**Tab 3: Full Schedule Analysis**
- Schedule 2–32 modules (Group A + B) with shared resource limits
- Configure phase durations per group, adsorption capacity, and shared resource caps (Evacuation+Cooling, NCG+Heating+CO2)
- Baseline analysis: Group A only (no sharing) vs Group A+B Concurrent vs Group A+B Interleaved
- Gantt charts, cycle counts, and phase breakdown (total minutes per phase)

**Tab 4: Advanced Interleaved**

Models a Carbon Nest schedule for a 16-module plant, grouped into three pairs — Group A, Group B, and Group C.
- Each pair cycles through three phase groups: Adsorption, the Desorption chain (Evacuation → NCG Purging → Heating → CO2 Purging), and Cooling
- Adsorption is the only phase group that can run for two pairs at once; the Desorption chain and Cooling are each limited to one pair at a time across the whole plant
- Evacuation takes priority over Cooling: if another pair is ready to begin its Desorption chain while a pair is still cooling, that pair's Cooling pauses and resumes with its remaining duration as soon as the conflicting Evacuation ends
- Gantt charts, complete-cycle counts, and phase breakdowns (total minutes per phase) are generated once phase durations are filled in for every pair and the schedule is generated

*Note: schedule quality depends heavily on the phase durations entered — configure realistic per-phase timings for each pair before drawing conclusions from the results.*

**Tab 5: Yield vs Cycles**
Compares Total Cycles, Total Yield, and Total Energy across Concurrent, Interleaved, and Advanced Interleaved. It shows each process's numbers from the last time its own "Generate" button was clicked — it does not recompute live as you edit inputs elsewhere, since Full Schedule Analysis's optimization is too heavy to re-run on every keystroke. Re-click Generate in a tab to refresh its entry here.
""")

st.markdown("<h1 style='text-align: center;'>Nelion Cycle Schedule</h1>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Test", "M2&M4 + LRVP", "Full Schedule Analysis", "Advanced Interleaved", "Yield vs Cycles"])

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
     
with tab1:
# === INDIVIDUAL MODULE OPERATION ===
# "General Test": the simplest scheduler in the app. Every module runs the same
# two phases (Adsorption, Desorption) back to back, sharing two capacity-limited
# resource pools, for as many complete cycles as fit in the operating period.
    st.markdown("<h2 style='text-align: center;'>Input Configuration</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Phase Duration")
        adsorption_duration = st.number_input("*Adsorption (min)*", 10, 240, 30)
        desorption_duration = st.number_input("*Desorption (min)*", 10, 240, 100)
        total_mins = st.number_input("*Operating Period (min)*", 0, 1440, 1440)
  
    with col2:
        st.markdown("Resource Limits")
        tots = st.number_input("*Total Modules*", 1, 16, 2)
        ads_mod = st.number_input("*Adsorbing Modules*", 1, 16, 2)
        des_mod = st.number_input("*Desorbing Modules*", 1, 16, 2)

    RESOURCE_LIMITS = {'Adsorption': ads_mod, 
                       'Desorption': des_mod}
    PHASE_DURATIONS = {'Adsorption': adsorption_duration,
                        'Desorption': desorption_duration}
    PHASES = list(PHASE_DURATIONS.keys())
    TOTAL_MODULES = tots
    MODULES = [f'M{i}' for i in range(1, TOTAL_MODULES + 1)]
    TOTAL_MINUTES = total_mins

# === RESOURCE TRACKING ===
    resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
    module_timers = {mod: 0 for mod in MODULES}
    schedule = []

# === SCHEDULING FUNCTIONS ===
    def can_allocate(phase, start, duration):
        """True if every minute of [start, start+duration) still has spare capacity for this phase."""
        return all(resource_usage[phase][t] < RESOURCE_LIMITS[phase] for t in range(start, start + duration))

    def reserve(phase, start, duration):
        """Occupy one capacity slot of this phase for every minute in [start, start+duration)."""
        for t in range(start, start + duration):
            resource_usage[phase][t] += 1

# === SCHEDULING LOOP ===
# Repeatedly sweep all modules, giving each the next phase in its cycle as soon as
# capacity allows, until a full pass places nothing new (either the operating period
# is used up or every module is waiting on a full resource).
    while True:
        progress = False
        for mod in MODULES:
            t = module_timers[mod]
            cycle_phases = []
            for phase in PHASES:
                duration = PHASE_DURATIONS[phase]
                # Look for the earliest minute, from this module's current time onward,
                # where the phase's resource pool has a free slot.
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
                t += duration
            # Only commit this module's phases if it completed the FULL phase list
            # (Adsorption + Desorption) — a partial cycle at the end of the period is dropped.
            if len(cycle_phases) == len(PHASES):
                for phase, start, end in cycle_phases:
                    schedule.append({"Module": mod, "Phase": phase, "Start": start, "End": end})
                module_timers[mod] = t
                progress = True
        if not progress:
            break

    df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start'])

# === FLEXIBLE CYCLE COUNT ===
# Count each module's complete cycles by sliding a window over its scheduled phases
# and matching it against the expected phase order (handles any leftover/partial rows).
    cycle_counts = []
    for mod in MODULES:
        mod_df = df_schedule[df_schedule['Module'] == mod].sort_values(by='Start').reset_index(drop=True)
        count = 0
        i = 0
        while i <= len(mod_df) - len(PHASES):
            window = mod_df.iloc[i:i+len(PHASES)]
            if list(window['Phase']) == PHASES:
                count += 1
                i += len(PHASES)
            else:
                i += 1
        cycle_counts.append({'Module': mod, 'Complete Cycles': count})
    cycle_counts_df = pd.DataFrame(cycle_counts)
    
    st.markdown("<h2 style='text-align: center;'>Complete Cycles</h2>", unsafe_allow_html=True)
    st.dataframe(cycle_counts_df)

    # === GANTT CHART ===
    colors = {'Adsorption': '#4B9CD3', 'Desorption': '#90EE90'}

    fig, ax = plt.subplots(figsize=(12, 5))
    df_schedule['Module'] = pd.Categorical(df_schedule['Module'], categories=MODULES, ordered=True)
# Then sort the DataFrame by the new categorical 'Module' column to ensure plotting order.
    df_schedule = df_schedule.sort_values('Module')
    for _, row in df_schedule.iterrows():
        ax.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                color=colors[row['Phase']], edgecolor='black')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Process Sequence')
    ax.set_xlim(0, TOTAL_MINUTES)
    ax.set_yticks(MODULES)
    ax.set_yticklabels(MODULES) # Ensure labels match the ticks
    ax.invert_yaxis()
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=colors.keys(), loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
 
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
    st.subheader("Inputs")
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        total_modules = st.number_input("Total Modules", 2, 32, 16, step=2)
        target_time = st.number_input("Operating Period (Mins)", 0, 1440, 1440)
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
            help="One shared resource for all three phases. Max modules in NCG Purging, Heating, or CO2 Purging combined at once."
        )
    st.caption("Phase durations (minutes) — edit directly in the table")
    default_phase_durations_tab3 = load_stage_duration_defaults_tab3()
    st.caption(
        "Defaults for both pairs are seeded from carbonnest_stage_timestamps.csv using the "
        f"N1N2N3-M1n3 cycle average: {default_phase_durations_tab3}. "
        "Edit per pair if a pair's real durations differ."
    )
    PHASE_DURATIONS_TAB3_VERSION = 4  # bump whenever load_stage_duration_defaults_tab3()'s sourcing changes
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

    enforce_evac_cool = st.checkbox("Evacuation Overrides Cooling", value=True)
    allow_cooling_pause = st.checkbox(
    "Allow Cooling Pause During Evacuation",
    value=True
)

    # --- Fixed Constants ---
    TOTAL_MINUTES = target_time # 24 hours * 60 minutes
    TOTAL_MODULES = int(total_modules)
    MODULES = list(range(1, TOTAL_MODULES + 1))

    GROUP_IDS = ["A", "B"]
    GROUP_OF = {m: ("A" if m % 2 == 1 else "B") for m in MODULES}

    default_energy_tab3, default_yield_tab3 = load_plant_cycle_defaults_tab3()
    st.caption("Energy used & plant yield per pair, per completed cycle — edit directly in the table")
    st.caption(
        f"Defaults for both pairs are seeded from carbonnest_plant_cycles.csv as the sum of the "
        f"N1N2N3-M1n3 and N3-M2n4 cycle averages: {default_energy_tab3} kWh/cycle, "
        f"{default_yield_tab3} kg CO2/cycle. Edit per pair if a pair's real output differs."
    )
    pair_labels_tab3 = [f"Group {gid}" for gid in GROUP_IDS]
    ENERGY_YIELD_TAB3_DEFAULTS_VERSION = 1  # bump whenever load_plant_cycle_defaults_tab3()'s sourcing changes

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

    selected_delay = 0 # Default value, will be updated based on user choice

    def build_module_delays(offset, group_ids, group_of):
        """Stagger each group's start time by `offset` minutes (Group A starts at 0,
        Group B at 1x offset, etc.) so their phases interleave instead of colliding."""
        group_index = {g: i for i, g in enumerate(group_ids)}
        return {m: group_index[group_of[m]] * offset for m in MODULES}



    # --- Core Simulation Function ---
    def run_simulation(module_delays_config, phase_durations_config, resource_limits_config, current_desorption_mode, group_ids, group_of, phase_durations_by_group=None, concurrent_desorption_cap=1, shared_evac_cooling_cap=1, shared_purge_heat_co2_cap=1, modules_to_run=None, baseline_mode=False):
        """Runs scheduling simulation; returns schedule, resource_usage, evac/cool stats.
        modules_to_run: optional list of module IDs (default: all MODULES).
        baseline_mode: if True, no shared resource constraints—each phase has its own capacity."""
        modules = modules_to_run if modules_to_run is not None else list(MODULES)
        resource_usage = {phase: np.zeros(TOTAL_MINUTES, dtype=int) for phase in PHASES}
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
            if end_time > TOTAL_MINUTES or start < 0:
                return False
            end_safe = min(end_time, TOTAL_MINUTES)  # clamp for array bounds (indices 0..TOTAL_MINUTES-1)
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
                    for t in range(start, end_safe):
                        combined = sum(current_resource_state[p][t] for p in purge_heat_co2_phases)
                        if combined >= shared_purge_heat_co2_cap:
                            return False
                    return True
            # Concurrent: per-phase caps ignored for desorption phases; concurrent_desorption_cap gates Evacuation start
            if current_desorption_mode == "Concurrent" and phase == "Evacuation":
                desorption_phases_plus_cooling = list(DESORPTION_PHASES) + ["Cooling"]
                for t in range(start, end_safe):
                    total_in_desorption = sum(current_resource_state[p][t] for p in desorption_phases_plus_cooling)
                    if total_in_desorption >= concurrent_desorption_cap:
                        return False
            return True

        def reserve_internal(phase, start, duration, current_resource_state):
            """Occupy one capacity slot of this phase for every minute in [start, start+duration)."""
            end_time = start + duration
            end_safe = min(end_time, TOTAL_MINUTES)
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
            # end_safe = min(end_time, TOTAL_MINUTES)
            # enforce_evac_cool: Evacuation overrides Cooling (Cooling pauses). Only Cooling waits.
            # When off: No interruption—both block each other; whoever started first runs to completion.
            if phase != "Cooling":
                return False
             
            end_time = start + duration
            end_safe = min(end_time, TOTAL_MINUTES)

            this_group = group_of[mod]
            other_group = "B" if this_group == "A" else "A"
            
            return np.any(current_resource_state["Evacuation"][start:end_safe] > 0)
            # if phase == "Evacuation":
            #     return False if enforce_evac_cool else np.any(current_resource_state["Cooling"][start:end_safe] > 0)
            # return False
        
        # Respect group pairings (odd=Group A, even=Group B): process Group A first, then Group B
        mod_order = sorted(modules, key=lambda m: (group_of.get(m, "A") != "A", m))
        progress_made = True
        # Main scheduling loop: repeatedly sweep every module, advancing each by one
        # more phase per pass, until a full sweep places nothing new (the operating
        # period is exhausted for every module). This is the same greedy pattern as
        # tab1/tab2, generalized to N modules across two groups with shared resources.
        while progress_made:
            progress_made = False
            for mod in mod_order:
                t = module_timers[mod]
                cycle_phases = []
                group_id = group_of[mod]

                # Work on a scratch copy of resource usage so a module's whole next
                # phase can be tried and rolled back (via cycle_success) without
                # corrupting the shared state if it turns out not to fit.
                temp_resource_usage_for_cycle = {phase: np.copy(resource_usage[phase]) for phase in PHASES}
                cycle_success = True

                for phase in PHASES:
                    if phase == "Adsorption" and adsorption_dur_uniform is not None:
                        duration = adsorption_dur_uniform
                    else:
                        base_dur = phase_durations_by_group[group_id][phase] if phase_durations_by_group else phase_durations_config[phase]
                        duration = adjusted_duration(phase, base_dur)

                    # Find the earliest minute, from t onward, where this phase can start.
                    attempt_start = t
                    conflict_hit = False
                    while attempt_start + duration <= TOTAL_MINUTES:
                        if evac_cool_conflict_internal(phase, attempt_start, duration, temp_resource_usage_for_cycle, mod):
                            # This is a Cooling attempt colliding with the other group's
                            # Evacuation. Rather than scan minute-by-minute, jump the
                            # search forward by that Evacuation's full duration (from the
                            # phase-duration table) and retry from there.
                            this_group = group_of[mod]
                            other_group = "B" if this_group == "A" else "A"
                            pause_minutes = int(PHASE_DURATIONS_BY_GROUP[other_group]["Evacuation"])

                            if not conflict_hit:
                                evac_cool_stats["cooling_delay_events"] += 1
                                conflict_hit = True

                            evac_cool_stats["cooling_delay_minutes"] += pause_minutes
                            attempt_start += pause_minutes
                            continue

                        if can_allocate_internal(
                            phase, attempt_start, duration,
                            temp_resource_usage_for_cycle
                        ):
                            break
                        attempt_start += 1

                    if attempt_start + duration > TOTAL_MINUTES:
                        # No valid start time left before the operating period ends —
                        # this module doesn't get another complete cycle.
                        cycle_success = False
                        break

                    # reserve_internal(
                    #     phase, attempt_start, duration,
                    #     temp_resource_usage_for_cycle
                    # )
                    # cycle_phases.append((phase, attempt_start, attempt_start + duration))
                    
                    # t = attempt_start + duration

                    if phase == "Cooling" and allow_cooling_pause:
                        # Cooling actually pauses here (separate from the search above,
                        # which only found a valid *start* time). Even after Cooling has
                        # begun, an Evacuation elsewhere can still start mid-way through
                        # it; this walks minute-by-minute through the Cooling duration
                        # and splits it into multiple (start, end) segments around any
                        # such conflicts, so Cooling stops for the conflict and resumes
                        # for its remaining time as soon as Evacuation frees up.
                        remaining = duration
                        cooling_segments = []
                        current_start = attempt_start

                        while remaining > 0:

                            evac_busy = temp_resource_usage_for_cycle["Evacuation"]

                            # Scan forward for the next minute Evacuation is active.
                            next_conflict = None

                            for t_check in range(
                                current_start,
                                min(TOTAL_MINUTES, current_start + remaining)
                            ):
                                if evac_busy[t_check] > 0:
                                    next_conflict = t_check
                                    break

                            if next_conflict is None:
                                # No conflict for the rest of the remaining duration —
                                # run straight through to the end.
                                cooling_segments.append(
                                    (current_start, current_start + remaining)
                                )
                                remaining = 0

                            else:
                                # Close out the segment that ran up to the conflict...
                                if next_conflict > current_start:

                                    run_time = next_conflict - current_start

                                    cooling_segments.append(
                                        (current_start, next_conflict)
                                    )

                                    remaining -= run_time

                                # ...then skip forward past the whole Evacuation window
                                # before resuming the search for the next segment.
                                pause_end = next_conflict

                                while (
                                    pause_end < TOTAL_MINUTES and
                                    evac_busy[pause_end] > 0
                                ):
                                    pause_end += 1

                                current_start = pause_end

                        # Reserve and record each Cooling segment as its own scheduled
                        # block (this is why a single Cooling "phase" can show up as
                        # more than one bar on the Gantt chart when it was paused).
                        for seg_start, seg_end in cooling_segments:

                            reserve_internal(
                                "Cooling",
                                seg_start,
                                seg_end - seg_start,
                                temp_resource_usage_for_cycle
                            )

                            cycle_phases.append(
                                ("Cooling", seg_start, seg_end)
                            )

                        t = cooling_segments[-1][1]

                    else:

                        reserve_internal(
                            phase,
                            attempt_start,
                            duration,
                            temp_resource_usage_for_cycle
                        )

                        cycle_phases.append(
                            (phase, attempt_start, attempt_start + duration)
                        )

                        t = attempt_start + duration

                    next_index = PHASES.index(phase) + 1
                    if next_index < len(PHASES):
                        next_phase = PHASES[next_index]
                        buffer_time = PHASE_GAPS.get((phase, next_phase), 0)
                        t += buffer_time
                
                if cycle_success:
                    for phase_name in PHASES:
                        resource_usage[phase_name] = np.copy(temp_resource_usage_for_cycle[phase_name])
                    for phase, start, end in cycle_phases:
                        schedule.append({"Module": mod, "Phase": phase, "Start": start, "End": end})
                    module_timers[mod] = t
                    progress_made = True
                
        df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start']).reset_index(drop=True)
        return df_schedule, resource_usage, evac_cool_stats


    # --- Main Run Button ---
    st.subheader("Results")
    st.markdown("---")
    if st.button("Generate Schedule and Analyze"):
        final_delay_to_use = 0
        optimization_summary_df = None

        def count_complete_cycles_df(df_schedule):
            """Count each module's completed cycles by sliding a window over its scheduled
            phases and matching it against the full PHASES order; returns one row per
            module with a "M<n>"-formatted label."""
            cycle_counts = []
            for mod in MODULES:
                mod_df = df_schedule[df_schedule["Module"] == mod].sort_values(by="Start").reset_index(drop=True)
                count = 0
                i = 0
                while i <= len(mod_df) - len(PHASES):
                    window = mod_df.iloc[i:i+len(PHASES)]
                    if list(window["Phase"]) == PHASES:
                        count += 1
                        i += len(PHASES)
                    else:
                        i += 1
                cycle_counts.append({"Module": mod, "Complete Cycles": count})
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

                for i, delay in enumerate(delay_search_range):
                    current_module_delays_for_opt = build_module_delays(delay, group_ids, group_of)
                    schedule_int, _, _ = run_simulation(
                        current_module_delays_for_opt, PHASE_DURATIONS, RESOURCE_LIMITS, "Interleaved", group_ids, group_of, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap
                    )
                    schedule_conc, _, _ = run_simulation(
                        current_module_delays_for_opt, PHASE_DURATIONS, RESOURCE_LIMITS, "Concurrent", group_ids, group_of, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap
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
        actual_module_delays = build_module_delays(final_delay_to_use, GROUP_IDS, GROUP_OF)
        
        schedule_int, resource_usage_int, conflicts_int = run_simulation(
            actual_module_delays, PHASE_DURATIONS, RESOURCE_LIMITS, "Interleaved", GROUP_IDS, GROUP_OF, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap
        )
        schedule_conc, resource_usage_conc, _ = run_simulation(
            actual_module_delays, PHASE_DURATIONS, RESOURCE_LIMITS, "Concurrent", GROUP_IDS, GROUP_OF, phase_durations_by_group=PHASE_DURATIONS_BY_GROUP, concurrent_desorption_cap=concurrent_desorption_cap, shared_evac_cooling_cap=shared_evac_cooling_cap, shared_purge_heat_co2_cap=shared_purge_heat_co2_cap
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
            entered for that pair (PAIR_*_PER_CYCLE_TAB3) to get plant output totals,
            plus a derived kg CO2 per kWh efficiency figure."""
            df = pair_totals_df.copy()
            gids = df["Pair"].str.replace("Group ", "", regex=False)
            df["Total Yield (kg CO2)"] = [
                round(cycles * PAIR_YIELD_PER_CYCLE_TAB3[gid], 1) for cycles, gid in zip(df["Total Cycles"], gids)
            ]
            df["Total Energy (kWh)"] = [
                round(cycles * PAIR_ENERGY_PER_CYCLE_TAB3[gid], 1) for cycles, gid in zip(df["Total Cycles"], gids)
            ]
            df["kg CO2 per kWh"] = [
                round(y / e, 3) if e > 0 else "—"
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
        # table above), NOT cycles_conc/cycles_int — those sum each of the 8
        # modules' own cycle counts per pair and overcount for the same reason
        # fixed earlier in "Cycle count per pair".
        st.session_state.setdefault("process_comparison", {})
        st.session_state["process_comparison"]["Concurrent"] = {
            "Total Cycles": int(yield_energy_conc["Total Cycles"].sum()),
            "Total Yield (kg CO2)": float(yield_energy_conc["Total Yield (kg CO2)"].sum()),
            "Total Energy (kWh)": float(yield_energy_conc["Total Energy (kWh)"].sum()),
        }
        st.session_state["process_comparison"]["Interleaved"] = {
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

        # === Baseline Analysis ===
        st.markdown("### Baseline Analysis")
        st.caption("Group A only (no resource sharing) compared against Group A + B Concurrent and Interleaved")
        inc_int = cycles_int - cycles_baseline
        inc_conc = cycles_conc - cycles_baseline
        bl_col1, bl_col2, bl_col3 = st.columns(3)
        with bl_col1:
            st.metric("Group A only (baseline)", cycles_baseline, help="Group A modules, Group A phase durations, each phase has its own capacity—no shared resource constraints")
        with bl_col2:
            st.metric("Group A + B Concurrent", cycles_conc, delta=inc_conc, help="All modules with Concurrent desorption cap")
        with bl_col3:
            st.metric("Group A + B Interleaved", cycles_int, delta=inc_int, help="All modules with Interleaved shared resource constraints")

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

        # === Baseline vs Concurrent vs Interleaved Comparison ===
        st.markdown("### Baseline vs Concurrent vs Interleaved Comparison")
        target_pct = 80  # % increase over baseline
        pct_conc_vs_baseline = ((cycles_conc - cycles_baseline) / cycles_baseline * 100) if cycles_baseline > 0 else 0
        pct_int_vs_baseline = ((cycles_int - cycles_baseline) / cycles_baseline * 100) if cycles_baseline > 0 else 0
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            st.metric("Baseline (Group A only)", f"{cycles_baseline} cycles", "no resource sharing")
        with comp_col2:
            delta_conc = f"+{pct_conc_vs_baseline:.1f}%" if pct_conc_vs_baseline >= 0 else f"{pct_conc_vs_baseline:.1f}%"
            st.metric("Concurrent (A+B)", f"{cycles_conc} cycles", delta_conc)
        with comp_col3:
            delta_int = f"+{pct_int_vs_baseline:.1f}%" if pct_int_vs_baseline >= 0 else f"{pct_int_vs_baseline:.1f}%"
            st.metric("Interleaved (A+B)", f"{cycles_int} cycles", delta_int)
        # Bar chart
        fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
        bars = ax_comp.bar(["Baseline (A only)", "Concurrent (A+B)", "Interleaved (A+B)"],
                           [cycles_baseline, cycles_conc, cycles_int],
                           color=["#95a5a6", "#6C5CE7", "#E07C5E"], alpha=0.85, edgecolor="black")
        ax_comp.set_ylabel("Total Complete Cycles")
        ax_comp.set_title(f"Output comparison (target: ≥{target_pct}% increase over baseline)")
        for b in bars:
            ax_comp.annotate(f"{int(b.get_height())}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                             ha="center", va="bottom", fontsize=12)
        ax_comp.set_ylim(0, max(cycles_baseline, cycles_int, cycles_conc) * 1.2)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close(fig_comp)
        met_conc = pct_conc_vs_baseline >= target_pct
        met_int = pct_int_vs_baseline >= target_pct
        if met_int and met_conc:
            st.success(f"✅ Both Concurrent and Interleaved achieve ≥{target_pct}% increase over baseline")
        elif met_int:
            st.success(f"✅ Interleaved achieves ≥{target_pct}% increase over baseline. Concurrent: {pct_conc_vs_baseline:.1f}%")
        elif met_conc:
            st.success(f"✅ Concurrent achieves ≥{target_pct}% increase over baseline. Interleaved: {pct_int_vs_baseline:.1f}%")
        else:
            st.warning(f"⚠️ Neither meets the {target_pct}% target. Concurrent: {pct_conc_vs_baseline:.1f}%, Interleaved: {pct_int_vs_baseline:.1f}%")

        st.markdown("### Evacuation/Cooling (Interleaved)")
        cool_ev = conflicts_int['cooling_delay_events']
        cool_min = conflicts_int['cooling_delay_minutes']
        evac_ev = conflicts_int['evac_delay_events']
        evac_min = conflicts_int['evac_delay_minutes']
        total_delay_min = cool_min + evac_min
        pct_lost = round(100 * total_delay_min / TOTAL_MINUTES, 2) if TOTAL_MINUTES > 0 else 0
        avg_cool = round(cool_min / cool_ev, 1) if cool_ev > 0 else 0
        avg_evac = round(evac_min / evac_ev, 1) if evac_ev > 0 else 0
        if enforce_evac_cool:
            st.caption("Evacuation overrides Cooling: Cooling can pause for Evacuation. Only Cooling delays are tracked.")
            evac_cool_col1, evac_cool_col2 = st.columns(2)
            with evac_cool_col1:
                st.metric("Cooling pauses (for Evacuation)", f"{cool_min} min total", f"{cool_ev} events")
                st.caption(f"Avg {avg_cool} min per event")
            with evac_cool_col2:
                st.metric("Evacuation delays", f"{evac_min} min total", f"{evac_ev} events (expected 0)")
            st.metric("Total delay", f"{total_delay_min} min", f"{pct_lost}% of operating time")
        else:
            st.caption("No override: neither phase interrupts the other. Both can be delayed.")
            evac_cool_col1, evac_cool_col2 = st.columns(2)
            with evac_cool_col1:
                st.metric("Cooling delays", f"{cool_min} min total", f"{cool_ev} events")
                st.caption(f"Avg {avg_cool} min per event")
            with evac_cool_col2:
                st.metric("Evacuation delays", f"{evac_min} min total", f"{evac_ev} events")
                st.caption(f"Avg {avg_evac} min per event")
            st.metric("Total delay", f"{total_delay_min} min", f"{pct_lost}% of operating time")
        if total_delay_min > 0:
            severity = "High" if pct_lost > 5 else ("Medium" if pct_lost > 1 else "Low")
            st.info(
                f"**{severity} impact.** Shared resource between Evacuation and Cooling. "
                "Consider adjusting group offset or capacity to reduce conflicts."
            )
        else:
            st.success("No Evacuation/Cooling conflicts — schedule ran without delays.")

        # Diagnostic: Adsorption durations by module
        ads_df = schedule_int[schedule_int["Phase"] == "Adsorption"].copy()
        if len(ads_df) > 0:
            ads_df["Duration"] = ads_df["End"] - ads_df["Start"]
            ads_summary = ads_df.groupby("Module")["Duration"].agg(["min", "max", "mean", "count"]).round(1)
            expected = int(PHASE_DURATIONS["Adsorption"])
            with st.expander("Schedule diagnostic: Adsorption durations by module"):
                diag_col1, diag_col2 = st.columns(2)
                with diag_col1:
                    st.caption("Interleaved")
                    st.dataframe(ads_summary, use_container_width=True, hide_index=True)
                    deviants_int = ads_summary[(ads_summary["min"] != expected) | (ads_summary["max"] != expected)]
                    if len(deviants_int) > 0 and grp_a == grp_b:
                        st.warning(f"Modules {list(deviants_int.index)} have Adsorption ≠ {expected} min")
                with diag_col2:
                    st.caption("Concurrent")
                    ads_conc = schedule_conc[schedule_conc["Phase"] == "Adsorption"].copy()
                    if len(ads_conc) > 0:
                        ads_conc["Duration"] = ads_conc["End"] - ads_conc["Start"]
                        ads_conc_summary = ads_conc.groupby("Module")["Duration"].agg(["min", "max", "mean", "count"]).round(1)
                        st.dataframe(ads_conc_summary, use_container_width=True, hide_index=True)
                        deviants_conc = ads_conc_summary[(ads_conc_summary["min"] != expected) | (ads_conc_summary["max"] != expected)]
                        if len(deviants_conc) > 0 and grp_a == grp_b:
                            st.warning(f"Modules {list(deviants_conc.index)} have Adsorption ≠ {expected} min")
                st.caption("All modules should show the same min/max when 'Different phase durations per group' is off.")
                st.download_button("Download Concurrent schedule (CSV)", schedule_conc.to_csv(index=False), file_name="schedule_concurrent.csv", mime="text/csv", key="dl_schedule_conc")
                st.download_button("Download Interleaved schedule (CSV)", schedule_int.to_csv(index=False), file_name="schedule_interleaved.csv", mime="text/csv", key="dl_schedule_int")

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

with tab2:
    # "M2&M4 + LRVP": fixed two-module (M2, M4) full 6-phase cycle, used to visualize
    # how a staggered start delay (M4 starts `delay_m4` minutes after M2) plays out,
    # with an optional "Concurrent" mode that locks the whole desorption chain +
    # Cooling to one module at a time.
    # === MODULES ===
    MODULES = ["M2", "M4"]

    # === PHASES ===
    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
    DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling'}
    
    # === STAGE DEFINITIONS ===
    STAGE_1_PHASES = {'Cooling', 'Adsorption'}  # Stage 1: Cooling & Adsorption
    STAGE_2_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging'}  # Stage 2: Desorption

    # === Input Configuration ===
    st.markdown("<h2 style='text-align: center;'>Input Configuration</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        delay_m4 = st.number_input("Start Delay for M4 (min) ", 0, 300, 49)
        TOTAL_MINUTES = st.number_input("Operating Period (min) *Low Default Value set for better visibility of the stages* ", 0, 1440, 300)

    with col2:
        st.markdown("Stage 1")
        ad_d = st.number_input("Adsorption Duration (mins)", 0, 120, 25)
        cool_d = st.number_input("Cooling Duration (mins)", 0, 60, 30)
    
    with col3:
        st.markdown("Stage 2")
        evac_d = st.number_input("Evacuation Duration (mins)", 0, 60, 7)
        ncg_d = st.number_input("NCG Purging Duration (mins)", 0, 60, 2)
        heat_d = st.number_input("Heating Duration (mins)", 0, 60, 20)
        co2_d = st.number_input("CO2 Purging Duration (mins)", 0, 60, 20)      

    PHASE_DURATIONS = {
        'Adsorption': ad_d,
        'Evacuation': evac_d,
        'NCG Purging': ncg_d,
        'Heating': heat_d,
        'CO2 Purging': co2_d,
        'Cooling': cool_d
    }

    desorption_mode = st.radio(
        "Desorption Strategy",
        ('Interleaved', 'Concurrent'),
        key='tab2_desorption_mode',
        help="Interleaved: Multiple modules can run desorption phases in parallel. Concurrent: Limits how many pairs can be in adsorption and desorption at the same time (one-at-a-time per phase in this view)."
    )

    # === RESOURCE TRACKING ===
    resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
    MODULE_DELAYS = {'M2': 0, 'M4': delay_m4}
    module_timers = {mod: MODULE_DELAYS.get(mod, 0) for mod in MODULES}
    schedule = []
    desorption_lock_time = 0

    def can_allocate(phase, start, duration):
        """True if this phase is completely free (no other module in it) for [start, start+duration)."""
        return all(resource_usage[phase][t] == 0 for t in range(start, start + duration))

    def reserve(phase, start, duration):
        """Mark [start, start+duration) as occupied for this phase."""
        for t in range(start, start + duration):
            resource_usage[phase][t] += 1

    # Same greedy "advance each module through its phases in turn" loop as tab1/tab3,
    # but with only one slot per phase (can_allocate requires it to be fully free),
    # plus an extra Concurrent-mode rule below.
    while True:
        progress = False
        for mod in MODULES:
            t = module_timers[mod]
            cycle_phases = []
            for phase in PHASES:
                duration = PHASE_DURATIONS[phase]
                # Concurrent mode: a module can't start its desorption chain until the
                # previous module has finished Cooling (desorption_lock_time), so the
                # whole Evacuation->...->Cooling block runs for one module at a time.
                if desorption_mode == "Concurrent" and phase in DESORPTION_PHASES:
                    t = max(t, desorption_lock_time)
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
                if desorption_mode == "Concurrent" and phase == 'Cooling':
                    # This module just finished Cooling; the next module's desorption
                    # chain can't begin any earlier than this.
                    desorption_lock_time = t + duration
                t += duration
            if len(cycle_phases) == len(PHASES):
                for phase, start, end in cycle_phases:
                    schedule.append({"Module": mod, "Phase": phase, "Start": start, "End": end})
                module_timers[mod] = t
                progress = True

                
        if not progress:
            break

    df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start'])

# === FLEXIBLE CYCLE COUNT ===
    cycle_counts = []
    for mod in MODULES:
        mod_df = df_schedule[df_schedule['Module'] == mod].sort_values(by='Start').reset_index(drop=True)
        count = 0
        i = 0
        while i <= len(mod_df) - len(PHASES):
            window = mod_df.iloc[i:i+len(PHASES)]
            if list(window['Phase']) == PHASES:
                count += 1
                i += len(PHASES)
            else:
                i += 1
        cycle_counts.append({'Module': mod, 'Complete Cycles': count})
    cycle_counts_df = pd.DataFrame(cycle_counts)
    
    st.markdown("<h2 style='text-align: center;'>Complete Cycles</h2>", unsafe_allow_html=True)
    st.dataframe(cycle_counts_df)
    
# === Gantt Chart with Stage Hatching Patterns ===
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Phase colors
    colors = {'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
              'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'}
    
    # Draw bars with hatching patterns based on stage
    for _, row in df_schedule.iterrows():
        if row['Phase'] in STAGE_1_PHASES:
            # Stage 1: Solid fill (no hatching)
            ax1.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                     color=colors[row['Phase']], edgecolor='black', linewidth=1)
        elif row['Phase'] in STAGE_2_PHASES:
            # Stage 2: Diagonal line pattern overlay
            ax1.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                     color=colors[row['Phase']], edgecolor='black', linewidth=1, hatch='///')
    
    ax1.set_title("Process Sequence (Solid = Stage 1, Hatched = Stage 2)")
    ax1.set_xlim(0, TOTAL_MINUTES)
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Modules")
    
    # Create legend with phases and hatching styles
    phase_legend = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    hatch_legend = [plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black'),
                    plt.Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', hatch='///')]
    
    # Combine legends
    all_handles = phase_legend 
    all_labels = list(colors.keys()) 
    ax1.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(1.25, 1))
    st.pyplot(fig)

    st.markdown("**Stage 1:** Cooling & Adsorption | **Stage 2:** Evacuation, NCG Purging, Heating, CO2 Purging")

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

    st.markdown("### 6-6-4 Configuration")
    st.caption("Phase durations per pair - edit directly in the table")

    PAIRS = ["Group A", "Group B", "Group C"]
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
        "seeded from carbonnest_stage_timestamps.csv: Group A and Group C from the N1N2N3-M1n3 "
        "cycle average, Group B from the N1N2-M1n3 cycle average. Repressurization has no "
        "equivalent in that log, so it keeps its original default. Edit any cell if a pair's real "
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
        "Defaults are seeded from carbonnest_plant_cycles.csv. Energy per Cycle uses each pair's "
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

    # === 8-4-4 comparison configuration ===
    # Same scheduling engine, phase list, operating period, and Evacuation-overrides-
    # Cooling setting as the 6-6-4 config above — only the pairing/module sourcing
    # differs. See PAIRS_8_4_4 / PAIR_PLANT_MODULE_8_4_4 near the top of the file.
    st.markdown("### 8-4-4 Configuration (Comparison)")
    st.caption("Same scheduling rules as above, applied to an 8-4-4 pairing instead of 6-6-4, for side-by-side comparison.")

    adv_phase_columns_844 = ["Phase"] + [f"{p} (min)" for p in PAIRS_8_4_4]
    stage_defaults_by_pair_844 = load_stage_duration_defaults_by_pair(PAIRS_8_4_4, PAIR_PLANT_MODULE_8_4_4)
    ADV_PHASE_DURATIONS_844_VERSION = 1

    def _pair_phase_durations_844(pair):
        pair_defaults = stage_defaults_by_pair_844.get(pair, {})
        return [pair_defaults.get(phase, FALLBACK_PHASE_DURATIONS[phase]) for phase in PHASES]

    get_versioned_default_table(
        "adv_phase_durations_844", adv_phase_columns_844, ADV_PHASE_DURATIONS_844_VERSION,
        lambda: pd.DataFrame({
            "Phase": PHASES,
            "Pair (8) (min)": _pair_phase_durations_844("Pair (8)"),
            "Pair (4a) (min)": _pair_phase_durations_844("Pair (4a)"),
            "Pair (4b) (min)": _pair_phase_durations_844("Pair (4b)"),
        }),
    )
    st.caption(
        "Phase durations seeded from carbonnest_stage_timestamps.csv: Pair (8) from the N1N2N3-M1n3 "
        "cycle average, both 4-module pairs from the N1N2-M1n3 cycle average."
    )
    adv_phase_table_844 = st.data_editor(
        st.session_state.adv_phase_durations_844,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Phase": st.column_config.TextColumn("Phase", disabled=True),
            "Pair (8) (min)": st.column_config.NumberColumn("Pair (8) (min)", min_value=0, max_value=240, required=True),
            "Pair (4a) (min)": st.column_config.NumberColumn("Pair (4a) (min)", min_value=0, max_value=240, required=True),
            "Pair (4b) (min)": st.column_config.NumberColumn("Pair (4b) (min)", min_value=0, max_value=240, required=True),
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

    pair_plant_defaults_844 = load_plant_cycle_defaults_simple(PAIRS_8_4_4, PAIR_PLANT_MODULE_8_4_4)
    ENERGY_YIELD_844_VERSION = 1
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
        "Energy & Yield per cycle, same sourcing as phase durations (no summing/weighting): "
        "Pair (8) from N1N2N3-M1n3, both 4-module pairs from N1N2-M1n3."
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

    def run_advanced_interleaved(pairs, phase_durations_by_pair, total_minutes, enforce_evac_cool):
        """
        Advanced 3-pair interleaved scheduler. Builds a schedule by repeatedly giving
        each pair its next phase group as soon as that phase group's shared resource
        is free, until nothing more can be scheduled before total_minutes runs out.
        Works for any 3 pair names — pairs[0] is the "privileged" pair whose
        Adsorption can overlap with the other two; pairs[1] and pairs[2] cannot
        overlap each other's Adsorption.

        Sequence per pair:
        Adsorption -> Evacuation -> NCG Purging -> Heating -> CO2 Purging -> Cooling -> Repressurization

        Rules actually enforced below:
        - pairs[0] can overlap adsorption with pairs[1] and pairs[2].
        - pairs[1] and pairs[2] cannot overlap each other's adsorption.
        - Only one pair can be in the Desorption chain (Evacuation..CO2 Purging) at a time
          — tracked via resource_ready_time["Desorption"].
        - Only one pair can be in Cooling at a time — tracked via
          resource_ready_time["Cooling"] — but a pair's Cooling will pause (see below)
          rather than wait outright if another pair's Evacuation overlaps it.
        - Only one pair can be in Repressurization at a time — tracked via
          resource_ready_time["Repressurization"].

        Different pairs start staggered into different phase groups (see
        pair_next_phase below) precisely so they don't all compete for the same
        shared resource at once.
        """

        schedule = []
        pair_a, pair_b, pair_c = pairs

        DESORPTION_CHAIN = [
            "Evacuation",
            "NCG Purging",
            "Heating",
            "CO2 Purging"
        ]

        # Stagger the three pairs across different phase groups from the start, so
        # they aren't all trying to claim the same shared resource in the first pass.
        pair_next_phase = {
            pair_a: "Adsorption",
            pair_b: "Desorption",
            pair_c: "Cooling",
        }

        # Each pair's own local clock: the earliest minute IT is free to start its
        # next phase (independent of whether the shared resource for that phase is free).
        pair_ready_time = {pair: 0 for pair in pairs}

        # Shared, plant-wide resources: the earliest minute each one is free for the
        # NEXT pair to use, regardless of which pair is asking.
        resource_ready_time = {
            "Desorption": 0,
            "Cooling": 0,
            "Repressurization": 0,
        }

        # Main loop: sweep the three pairs, advancing each by one phase group per pass,
        # until a full pass makes no progress (every pair is either done or blocked by
        # the operating-period limit).
        while True:
            progress = False

            for pair in pairs:
                phase_group = pair_next_phase[pair]

                if phase_group == "Adsorption":
                    duration = phase_durations_by_pair[pair]["Adsorption"]

                    # pair_a can overlap with pair_b and pair_c.
                    # pair_b and pair_c cannot overlap each other.
                    start = pair_ready_time[pair]

                    if pair in (pair_b, pair_c):
                        blocking_pair = pair_c if pair == pair_b else pair_b
                        for row in schedule:
                            if row["Phase"] == "Adsorption" and row["Pair"] == blocking_pair:
                                if start < row["End"] and start + duration > row["Start"]:
                                    start = row["End"]

                    end = start + duration

                    if end > total_minutes:
                        continue

                    schedule.append({
                        "Pair": pair,
                        "Phase": "Adsorption",
                        "Start": start,
                        "End": end
                    })

                    pair_ready_time[pair] = end
                    pair_next_phase[pair] = "Desorption"
                    progress = True

                elif phase_group == "Desorption":
                    # Earliest this pair can start is whichever is later: its own
                    # readiness, or the shared Desorption resource being free.
                    earliest_start = max(pair_ready_time[pair], resource_ready_time["Desorption"])
                    start = earliest_start

                    # Schedule the whole Evacuation -> NCG Purging -> Heating -> CO2
                    # Purging chain back-to-back as one block; the shared "Desorption"
                    # resource is held for its full total_desorption_duration.
                    phase_start = start
                    total_desorption_duration = sum(
                        phase_durations_by_pair[pair][phase]
                        for phase in DESORPTION_CHAIN
                    )

                    end = start + total_desorption_duration

                    if end > total_minutes:
                        continue

                    for phase in DESORPTION_CHAIN:
                        duration = phase_durations_by_pair[pair][phase]

                        if duration > 0:
                            schedule.append({
                                "Pair": pair,
                                "Phase": phase,
                                "Start": phase_start,
                                "End": phase_start + duration
                            })

                        phase_start += duration

                    pair_ready_time[pair] = end
                    resource_ready_time["Desorption"] = end
                    pair_next_phase[pair] = "Cooling"
                    progress = True

                elif phase_group == "Cooling":
                    start = max(pair_ready_time[pair], resource_ready_time["Cooling"])
                    duration = phase_durations_by_pair[pair]["Cooling"]

                    # Evacuation gets priority over Cooling: if another pair's Evacuation
                    # overlaps the Cooling window, Cooling pauses and resumes with its
                    # remaining duration as soon as that Evacuation ends.
                    other_evac_events = sorted(
                        (row for row in schedule
                         if row["Phase"] == "Evacuation" and row["Pair"] != pair),
                        key=lambda r: r["Start"]
                    ) if enforce_evac_cool else []

                    cooling_segments = []
                    remaining = duration
                    current_start = start

                    while remaining > 0:
                        # Is current_start inside an Evacuation window? If so, jump past it.
                        blocking = next(
                            (row for row in other_evac_events
                             if row["Start"] <= current_start < row["End"]),
                            None
                        )
                        if blocking is not None:
                            current_start = blocking["End"]
                            continue

                        # Run until the next upcoming Evacuation start, or until done.
                        next_evac_start = next(
                            (row["Start"] for row in other_evac_events
                             if row["Start"] > current_start),
                            None
                        )
                        run_end = current_start + remaining
                        if next_evac_start is not None and next_evac_start < run_end:
                            run_end = next_evac_start

                        if run_end > current_start:
                            cooling_segments.append((current_start, run_end))
                            remaining -= (run_end - current_start)
                        current_start = run_end

                    end = cooling_segments[-1][1]

                    if end > total_minutes:
                        continue

                    for seg_start, seg_end in cooling_segments:
                        schedule.append({
                            "Pair": pair,
                            "Phase": "Cooling",
                            "Start": seg_start,
                            "End": seg_end
                        })

                    pair_ready_time[pair] = end
                    resource_ready_time["Cooling"] = end
                    pair_next_phase[pair] = "Repressurization"
                    progress = True

                elif phase_group == "Repressurization":
                    # Final phase of the cycle; frees the pair back to Adsorption once done.
                    start = max(pair_ready_time[pair], resource_ready_time["Repressurization"])
                    duration = phase_durations_by_pair[pair]["Repressurization"]
                    end = start + duration

                    if end > total_minutes:
                        continue

                    schedule.append({
                        "Pair": pair,
                        "Phase": "Repressurization",
                        "Start": start,
                        "End": end
                    })

                    pair_ready_time[pair] = end
                    resource_ready_time["Repressurization"] = end
                    pair_next_phase[pair] = "Adsorption"
                    progress = True

            if not progress:
                break

        return pd.DataFrame(schedule)

    def _complete_cycles_and_yield_energy(schedule_df, pairs, yield_per_cycle, energy_per_cycle):
        """Shared logic for both configurations: count each pair's complete cycles
        from its scheduled rows, then multiply by its per-cycle rates."""
        rows = []
        for pair in pairs:
            pair_df = schedule_df[schedule_df["Pair"] == pair]
            # Each full cycle contributes one scheduled row per phase in PHASES, so
            # total rows / phase count gives the completed-cycle count. Note: if a
            # Cooling phase got split into multiple segments (paused for another
            # pair's Evacuation), that cycle has one extra row, which can make this
            # a slight undercount versus the true cycle count.
            cycles = len(pair_df) // len(PHASES)
            total_yield = cycles * yield_per_cycle.get(pair, 0.0)
            total_energy = cycles * energy_per_cycle.get(pair, 0.0)
            rows.append({
                "Pair": pair,
                "Complete Cycles": cycles,
                "Total Yield (kg CO2)": round(total_yield, 1),
                "Total Energy (kWh)": round(total_energy, 1),
                "kg CO2 per kWh": round(total_yield / total_energy, 3) if total_energy > 0 else "—",
            })
        return pd.DataFrame(rows)

    if st.button("Generate Advanced Interleaved Schedules (6-6-4 & 8-4-4)", key="adv_generate"):
        adv_schedule = run_advanced_interleaved(PAIRS, PHASE_DURATIONS_BY_PAIR, TOTAL_MINUTES_ADV, adv_enforce_evac_cool)
        adv_schedule_844 = run_advanced_interleaved(PAIRS_8_4_4, PHASE_DURATIONS_BY_PAIR_844, TOTAL_MINUTES_ADV, adv_enforce_evac_cool)

        st.markdown("## 6-6-4 Configuration")
        st.markdown("### Complete Cycles")

        yield_energy_df = _complete_cycles_and_yield_energy(adv_schedule, PAIRS, PAIR_YIELD_PER_CYCLE, PAIR_ENERGY_PER_CYCLE)
        st.dataframe(yield_energy_df[["Pair", "Complete Cycles"]], use_container_width=True, hide_index=True)

        st.markdown("### Yield & Energy Analysis")
        st.caption("Total Yield/Energy = Complete Cycles × the combined per-cycle rate of every module in that pair.")
        st.dataframe(yield_energy_df, use_container_width=True, hide_index=True)

        ye_metric_col1, ye_metric_col2 = st.columns(2)
        with ye_metric_col1:
            st.metric("Plant Total Yield", f"{yield_energy_df['Total Yield (kg CO2)'].sum():.1f} kg CO2")
        with ye_metric_col2:
            st.metric("Plant Total Energy", f"{yield_energy_df['Total Energy (kWh)'].sum():.1f} kWh")

        # Publish totals for the Yield vs Cycles comparison tab. Only refreshes when
        # this "Generate" button is (re)clicked — see that tab's caption for why.
        st.session_state.setdefault("process_comparison", {})
        st.session_state["process_comparison"]["Advanced Interleaved"] = {
            "Total Cycles": int(yield_energy_df["Complete Cycles"].sum()),
            "Total Yield (kg CO2)": float(yield_energy_df["Total Yield (kg CO2)"].sum()),
            "Total Energy (kWh)": float(yield_energy_df["Total Energy (kWh)"].sum()),
        }

        st.markdown("## 8-4-4 Configuration")
        st.markdown("### Complete Cycles")

        yield_energy_df_844 = _complete_cycles_and_yield_energy(
            adv_schedule_844, PAIRS_8_4_4, PAIR_YIELD_PER_CYCLE_844, PAIR_ENERGY_PER_CYCLE_844
        )
        st.dataframe(yield_energy_df_844[["Pair", "Complete Cycles"]], use_container_width=True, hide_index=True)

        st.markdown("### Yield & Energy Analysis")
        st.caption("Total Yield/Energy = Complete Cycles × the per-cycle rate for that pair.")
        st.dataframe(yield_energy_df_844, use_container_width=True, hide_index=True)

        ye_metric_col1_844, ye_metric_col2_844 = st.columns(2)
        with ye_metric_col1_844:
            st.metric("Plant Total Yield", f"{yield_energy_df_844['Total Yield (kg CO2)'].sum():.1f} kg CO2")
        with ye_metric_col2_844:
            st.metric("Plant Total Energy", f"{yield_energy_df_844['Total Energy (kWh)'].sum():.1f} kWh")

        st.markdown("### Advanced Interleaved Gantt Chart (6-6-4 Configuration)")

        colors = {
            'Adsorption': '#4B9CD3',
            'Evacuation': '#FFB347',
            'NCG Purging': '#FFD700',
            'Heating': '#E97451',
            'CO2 Purging': '#90EE90',
            'Cooling': '#9370DB',
            'Repressurization': '#B0B0B0'
        }

        fig, ax = plt.subplots(figsize=(14, 5))

        for _, row in adv_schedule.iterrows():
            ax.barh(
                row["Pair"],
                row["End"] - row["Start"],
                left=row["Start"],
                color=colors.get(row["Phase"], "#888"),
                edgecolor="black"
            )

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Pairs")
        ax.set_title("Advanced Interleaved Schedule - 3 Pairs")
        ax.set_xlim(0, TOTAL_MINUTES_ADV)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PHASES],
            PHASES,
            loc="upper right",
            fontsize=8
        )

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Schedule Data"):
            st.dataframe(adv_schedule, use_container_width=True, hide_index=True)

with tab5:
    # Cross-process comparison. Reads whatever tab3 (Concurrent/Interleaved) and
    # tab4 (Advanced Interleaved) last published to st.session_state["process_comparison"]
    # when their own "Generate" buttons were clicked — see the sidebar guide for why
    # this doesn't recompute live on every input change.
    st.subheader("Yield, Energy & Cycles Comparison")
    st.caption(
        "Shows each process's Total Cycles, Total Yield, and Total Energy from the last time its "
        "own \"Generate\" button was clicked (Concurrent/Interleaved from Full Schedule Analysis, "
        "Advanced Interleaved from its own tab). Re-click Generate in a tab to refresh its numbers here."
    )

    PROCESS_ORDER = ["Concurrent", "Interleaved", "Advanced Interleaved"]
    PROCESS_COLORS = {
        "Concurrent": "#6C5CE7",
        "Interleaved": "#E07C5E",
        "Advanced Interleaved": "#2ECC71",
    }
    comparison = st.session_state.get("process_comparison", {})
    available_processes = [p for p in PROCESS_ORDER if p in comparison]
    missing_processes = [p for p in PROCESS_ORDER if p not in comparison]

    if missing_processes:
        st.info(
            "Missing: " + ", ".join(missing_processes) +
            ". Generate a schedule in the relevant tab(s) to add them here."
        )

    if not available_processes:
        st.warning("No results yet — generate a schedule in Full Schedule Analysis and/or Advanced Interleaved first.")
    else:
        comparison_df = pd.DataFrame([
            {
                "Process": p,
                "Total Cycles": comparison[p]["Total Cycles"],
                "Total Yield (kg CO2)": round(comparison[p]["Total Yield (kg CO2)"], 1),
                "Total Energy (kWh)": round(comparison[p]["Total Energy (kWh)"], 1),
                "kg CO2 per kWh": (
                    round(comparison[p]["Total Yield (kg CO2)"] / comparison[p]["Total Energy (kWh)"], 3)
                    if comparison[p]["Total Energy (kWh)"] > 0 else "—"
                ),
            }
            for p in available_processes
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("### Compare a parameter across processes")
        selected_metric = st.selectbox(
            "Parameter",
            ["Total Cycles", "Total Yield (kg CO2)", "Total Energy (kWh)"],
            key="comparison_metric",
        )
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
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)