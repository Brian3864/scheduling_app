# === Streamlit App: Interleaved Desorption Scheduling & Steam Demand ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.sidebar.markdown("##  Cycle Resources")
fan_power = st.sidebar.number_input("Fan Rating (kW)", 0.0, 15.0, 1.5)
boiler_power = st.sidebar.number_input("Boiler (kW)", 0.0, 500.0, 108.0)
vpump = st.sidebar.number_input("Vacuum Pump (kW)", 0.0, 15.0, 5.5)
ctower = st.sidebar.number_input("Cooling Tower (kW)", 0.0, 15.0, 3.0)

st.sidebar.markdown("Steam Demand Per Phase")
ncg_purging = st.sidebar.number_input("NCG Purging (kg/hr)", 0, 300, 150)
heating = st.sidebar.number_input("Heating (kg/hr)", 0, 300, 150)
co2_purging = st.sidebar.number_input("CO2 Purging (kg/hr)", 0, 300, 150)

st.sidebar.markdown("### ℹ️ Guide")
st.sidebar.markdown("""
**Tab 1: 2-Modules**  
Customize:
- **Adsorption and Desorption durations**
- **Resource limits** – define how many modules can run each phase concurrently

**Tab 2: 4-Modules**  
Choose between:
- **Serialized Desorption**: Entire desorption sequence is locked for one module pair at a time
- **Interleaved Desorption**: Allows overlapping of non-conflicting phases (e.g., one pair heating, another cooling) while avoiding overlap in steam-intensive steps
""")

st.markdown("<h1 style='text-align: center;'>Nelion Cycle Schedule</h1>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Test", "Module Pair Analysis", "M2&M4 + LRVP", "4 MODULES", "Automatic Optimization"])

with tab2: 
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
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def _build_fan_map(modules, fan_pairs):
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
        e = s + d
        return e <= TOTAL_MINUTES and np.all(usage[phase][s:e] < lim)

    def reserve_phase(phase, s, d, usage):
        usage[phase][s:s+d] += 1

    def can_fan(mod, phase, s, d, _fans):
        if not ads_fan_exclusive or phase != "Adsorption":
            return True
        e = s + d
        fid = fan_of[mod]
        return e <= TOTAL_MINUTES and np.all(_fans[fid][s:e] == 0)

    def reserve_fan(mod, phase, s, d, _fans):
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
    """Create a combined metrics table with cycles, CO2, utilization, and active minutes."""
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
     
with tab4:
     # === MODULES ===
    MODULES = ["M1&M3", "M2&M4"]

    # === PHASES ===
    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
    DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling'}

    # === Input Configuration ===
    st.markdown("<h2 style='text-align: center;'>Input Configuration</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        desorption_mode = st.radio("Desorption Strategy", options=["Serialized", "Interleaved"], index=1)
        delay_m2 = st.number_input("Start Delay for M2&M4 (min)", 0, 300, 62)
        TOTAL_MINUTES = st.number_input("Operating Perid (min)", 0, 1440, 1440)

    with col2:
        ad_d = st.number_input("Adsorption Duration (min)", 0, 120, 25)
        evac_d = st.number_input("Evacuation Duration (min)", 0, 60, 7)
        ncg_d = st.number_input("NCG Purging Duration (min)", 0, 60, 2)
    
    with col3:
        heat_d = st.number_input("Heating Duration (min)", 0, 60, 20)
        co2_d = st.number_input("CO2 Purging Duration (min)", 0, 60, 40)
        cool_d = st.number_input("Cooling Duration (min)", 0, 60, 30)       

    PHASE_DURATIONS = {
        'Adsorption': ad_d,
        'Evacuation': evac_d,
        'NCG Purging': ncg_d,
        'Heating': heat_d,
        'CO2 Purging': co2_d,
        'Cooling': cool_d
    }

    steam_demand_per_phase = {
        'NCG Purging': ncg_purging,
        'Heating': heating,
        'CO2 Purging': co2_purging
    }

    power_ratings = {
        'Adsorption': fan_power,
        'Evacuation': vpump,
        'NCG Purging': boiler_power + vpump + ctower,
        'Heating': boiler_power + vpump + ctower,
        'CO2 Purging': boiler_power + vpump + ctower,
        'Cooling': ctower + vpump
    }

    # === Use `delay` in your scheduling logic ===
    module_timers = {'M1&M3': 0, 'M2&M4': delay_m2}

    # === RESOURCE TRACKING ===
    resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
    MODULE_DELAYS = {'M1&M3': 0, 'M2&M4': delay_m2}
    module_timers = {mod: MODULE_DELAYS.get(mod, 0) for mod in MODULES}
    schedule = []
    desorption_lock_time = 0

    def can_allocate(phase, start, duration):
        return all(resource_usage[phase][t] == 0 for t in range(start, start + duration))

    def reserve(phase, start, duration):
        for t in range(start, start + duration):
            resource_usage[phase][t] += 1

    while True:
        progress = False
        for mod in MODULES:
            t = module_timers[mod]
            cycle_phases = []
            for phase in PHASES:
                duration = PHASE_DURATIONS[phase]
                if desorption_mode == "Serialized" and phase in DESORPTION_PHASES:
                    t = max(t, desorption_lock_time)
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
                if desorption_mode == "Serialized" and phase == 'Cooling':
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
    # === Gantt Chart ===
    fig, ax1 = plt.subplots(figsize=(12, 5))
    colors = {'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
              'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'}
    for _, row in df_schedule.iterrows():
        ax1.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                 color=colors[row['Phase']], edgecolor='black')
    ax1.set_title("Process Sequence")
    ax1.set_xlim(0, TOTAL_MINUTES)
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Modules")
    ax1.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
               colors.keys(), loc='upper right')
    st.pyplot(fig)

    # === Steam Profile ===
    steam_breakdown = pd.DataFrame(0, index=range(TOTAL_MINUTES), columns=steam_demand_per_phase.keys())
    for _, row in df_schedule.iterrows():
        if row['Phase'] in steam_demand_per_phase:
            for t in range(row['Start'], row['End']):
                steam_breakdown.loc[t, row['Phase']] += steam_demand_per_phase[row['Phase']]
    steam_profile = steam_breakdown.sum(axis=1)

    st.markdown("### Steam Demand Profile")
    fig2, ax2 = plt.subplots(figsize=(15, 3))
    ax2.plot(steam_profile.index, steam_profile.values, color='red', linewidth=2)
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Steam Demand (kg/hr)")
    ax2.set_title("Total Steam Demand")
    ax2.grid(True)
    st.pyplot(fig2)

    # === Power Profile ===
    # === Shared Power Profile (Avoid Double Counting Shared Equipment) ===
    power_profile = np.zeros(TOTAL_MINUTES)

    for t in range(TOTAL_MINUTES):
            # Get all active rows at time t
        active_rows = df_schedule[(df_schedule['Start'] <= t) & (df_schedule['End'] > t)]

        # === Module-specific: Adsorption (can run in parallel)
        adsorption_rows = active_rows[active_rows['Phase'] == 'Adsorption']
        power_profile[t] += len(adsorption_rows) * fan_power

        # === Shared Desorption Equipment (count once if active)
        active_phases = active_rows['Phase'].unique()

    # --- Shared equipment — only add once even if multiple modules are active ---
        if 'Evacuation' in active_phases:
            power_profile[t] += vpump

        if any(p in ['NCG Purging', 'Heating', 'CO2 Purging'] for p in active_phases):
            power_profile[t] += boiler_power + vpump + ctower  # shared steam equipment

        if 'Cooling' in active_phases:
            power_profile[t] += ctower + vpump  # shared again but still only once

        # === Peak Demand Info ===
    peak_power = np.max(power_profile)
    peak_time = int(np.argmax(power_profile))

    # === Plot Power Profile ===
    st.markdown("### Power Demand Profile")
    fig3, ax3 = plt.subplots(figsize=(15, 3))
    ax3.plot(power_profile, color='red', label='Power Demand')
    ax3.axvline(peak_time, color='blue', linestyle='--', label=f'Peak @ {peak_time} min')
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Power (kW)")
    ax3.set_title("Real-Time Power Demand")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Optional: Show peak value
    st.markdown(f"*Peak Power Demand: {peak_power:.1f} kW ~ {peak_power / 0.8:.1f} kVA at minute {peak_time}*")

with tab1:
# === INDIVIDUAL MODULE OPERATION ===
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
        return all(resource_usage[phase][t] < RESOURCE_LIMITS[phase] for t in range(start, start + duration))

    def reserve(phase, start, duration):
        for t in range(start, start + duration):
            resource_usage[phase][t] += 1

# === SCHEDULING LOOP ===
    while True:
        progress = False
        for mod in MODULES:
            t = module_timers[mod]
            cycle_phases = []
            for phase in PHASES:
                duration = PHASE_DURATIONS[phase]
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
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

    power_profile = np.zeros(TOTAL_MINUTES)

    for t in range(TOTAL_MINUTES):
            # Get all active rows at time t
        active_rows = df_schedule[(df_schedule['Start'] <= t) & (df_schedule['End'] > t)]

        # === Module-specific: Adsorption (can run in parallel)
        adsorption_rows = active_rows[active_rows['Phase'] == 'Adsorption']
        power_profile[t] += len(adsorption_rows) * fan_power

        # === Shared Desorption Equipment (count once if active)
        active_phases = active_rows['Phase'].unique()

    # --- Shared equipment — only add once even if multiple modules are active ---
        if 'Desorption' in active_phases:
            power_profile[t] += boiler_power + vpump + ctower

        # === Peak Demand Info ===
    peak_power = np.max(power_profile)
    peak_time = int(np.argmax(power_profile))

    # === Plot Power Profile ===
    st.markdown("### Power Demand Profile")
    fig3, ax3 = plt.subplots(figsize=(15, 3))
    ax3.plot(power_profile, color='red', label='Power Demand')
    ax3.axvline(peak_time, color='blue', linestyle='--', label=f'Peak @ {peak_time} min')
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Power (kW)")
    ax3.set_title("Real-Time Power Demand")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Optional: Show peak value
    st.markdown(f"*Peak Power Demand: {peak_power:.1f} kW ~ {peak_power / 0.8:.1f} kVA at minute {peak_time}*")
 
with tab5:
    # --- Streamlit UI Elements ---
    st.set_page_config(layout="wide") # Use wide layout for better visualization

    # === MODULES ===
    MODULES = ["M1&M3", "M2&M4"]

    # === PHASES ===
    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
    DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling'} # Used for steam demand calculation

    # === Input Configuration ===
    col1, col2, col3 = st.columns(3)

    # --- Delay Selection Option ---
    delay_method = 'Automatic'

    with col1: 
        # --- Desorption Mode Selection ---
        desorption_mode = st.radio(
            "Desorption Strategy",
            ('Interleaved', 'Serialized'),
            key='desorption_mode_radio',
        help="Interleaved: Multiple modules can desorb simultaneously if resources allow. Serialized: Only one module can be in any desorption phase at a time."
    )
        target_st = st.number_input("Maximum Steam Demand (kg/hr)", 10, 300, 150)
        target_time = st.number_input("Operating Period (Mins)", 0, 1440, 1440)
        
    with col2:
        adsorption = st.number_input("Adsorption Duration", 0, 120, 25)
        evacuation = st.number_input("Evacuation Duration ", 0, 60, 7)
        ncg = st.number_input("NCG Purging Duration", 0, 60, 2)

    with col3:
        heating = st.number_input("Heating Duration ", 0, 60, 20)
        co2 = st.number_input("CO2 Purging Duration", 0, 60, 40)
        cool = st.number_input("Cooling Duration", 0, 60, 30)

    # Input from your configuration - now driven by Streamlit numbers
    PHASE_DURATIONS = {
        'Adsorption': adsorption,
        'Evacuation': evacuation,
        'NCG Purging': ncg,
        'Heating': heating,
        'CO2 Purging': co2,
        'Cooling': cool
    }

    # --- Fixed Constants ---
    TOTAL_MINUTES = target_time # 24 hours * 60 minutes
    TOTAL_MODULES = len(MODULES) # Number of module groups

    # Simulate steam demands (kg/hr per module running that phase)
    steam_demand_per_phase = {
        'NCG Purging': 150,
        'Heating': 150,
        'CO2 Purging': 150
    }

    # Idle time required between specific phase transitions (in minutes)
    PHASE_GAPS = {
        ('NCG Purging', 'Heating'): 0,
        ('Heating', 'CO2 Purging'): 0
    }

    # Define target steam limit globally so it's always available
    target_steam_limit_kg_hr =  target_st # Your target steam limit in kg/hr
    target_steam_limit_kg_min = target_steam_limit_kg_hr / 60.0

    # --- Resource Limits will depend on desorption_mode ---
    RESOURCE_LIMITS = {}
    if desorption_mode == "Interleaved":
        for phase in PHASES:
            RESOURCE_LIMITS[phase] = TOTAL_MODULES
    else: # Serialized mode
        for phase in PHASES:
            if phase in DESORPTION_PHASES:
                RESOURCE_LIMITS[phase] = 1 # Only one module in any desorption phase at a time
            else:
                RESOURCE_LIMITS[phase] = TOTAL_MODULES # Adsorption can be interleaved

    selected_delay = 0 # Default value, will be updated based on user choice


    # --- Core Simulation Function ---
    def run_simulation(module_delays_config, phase_durations_config, resource_limits_config, current_desorption_mode):
        """
        Runs the scheduling simulation with given module delays and returns
        the generated schedule, the total steam demand profile, and resource usage.
        """
        resource_usage = {phase: np.zeros(TOTAL_MINUTES, dtype=int) for phase in PHASES}
        module_timers = {mod: 0 for mod in MODULES}

        for mod, delay in module_delays_config.items():
            module_timers[mod] = delay
        
        schedule = []
        
        desorption_lock_time = 0 

        def can_allocate_internal(phase, start, duration, current_resource_state):
            end_time = start + duration
            if end_time > TOTAL_MINUTES:
                return False
            return np.all(current_resource_state[phase][start : end_time] < resource_limits_config[phase])

        def reserve_internal(phase, start, duration, current_resource_state):
            end_time = start + duration
            current_resource_state[phase][start : end_time] += 1
        
        progress_made = True
        while progress_made:
            progress_made = False
            for mod in MODULES:
                t = module_timers[mod]
                cycle_phases = []
                
                temp_resource_usage_for_cycle = {phase: np.copy(resource_usage[phase]) for phase in PHASES}

                cycle_success = True
                
                for phase in PHASES:
                    duration = phase_durations_config[phase]
                    
                    if current_desorption_mode == "Serialized" and phase in DESORPTION_PHASES:
                        t = max(t, desorption_lock_time)

                    attempt_start = t
                    while attempt_start + duration <= TOTAL_MINUTES:
                        if can_allocate_internal(phase, attempt_start, duration, temp_resource_usage_for_cycle):
                            break
                        attempt_start += 1
                    
                    if attempt_start + duration > TOTAL_MINUTES:
                        cycle_success = False
                        break

                    reserve_internal(phase, attempt_start, duration, temp_resource_usage_for_cycle)
                    cycle_phases.append((phase, attempt_start, attempt_start + duration))
                    
                    t = attempt_start + duration

                    if current_desorption_mode == "Serialized" and phase == 'Cooling':
                        desorption_lock_time = max(desorption_lock_time, t)

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

        # --- CORRECTED STEAM CALCULATION ---
        # We build the profile in kg/hr for each minute, then scale it correctly
        steam_breakdown_kg_per_hr = pd.DataFrame(0.0, index=range(TOTAL_MINUTES), columns=steam_demand_per_phase.keys())
        for _, row in df_schedule.iterrows():
            if row['Phase'] in steam_demand_per_phase:
                steam_per_hr = steam_demand_per_phase[row['Phase']]
                for t_idx in range(row['Start'], row['End']):
                    if t_idx < TOTAL_MINUTES:
                        steam_breakdown_kg_per_hr.loc[t_idx, row['Phase']] += steam_per_hr

        # Sum across phases to get total kg/hr demand per minute
        # No division by 60 here, the values are correctly in kg/hr
        steam_profile = steam_breakdown_kg_per_hr.sum(axis=1)

        return df_schedule, steam_profile, resource_usage


    # --- Main Run Button ---
    st.markdown("---")
    if st.button("Generate Schedule and Analyze"):
        final_delay_to_use = 0
        optimization_summary_df = None

        if delay_method == 'Automatic':
            st.info("Running optimization to find the best delay...")
            
            best_delay = 0
            min_peak_steam = float('inf')
            min_peak_duration = TOTAL_MINUTES
            
            delay_search_range = range(0, 120, 2)
            
            optimization_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, delay in enumerate(delay_search_range):
                current_module_delays_for_opt = {
                    'M1&M3': 0,
                    'M2&M4': delay
                }

                _, current_steam_profile, _ = run_simulation(
                    current_module_delays_for_opt, PHASE_DURATIONS, RESOURCE_LIMITS, desorption_mode
                )
                
                # current_steam_profile is already in kg/hr from the corrected simulation, so no need to multiply by 60
                current_peak_steam_hr = current_steam_profile.max()

                # Now, `target_steam_limit_kg_hr` is the correct unit to compare against
                duration_above_target = (current_steam_profile > target_steam_limit_kg_hr).sum()

                optimization_results.append({
                    'Delay_M2&M4': delay,
                    'Peak_Steam_kg_per_hr': current_peak_steam_hr,
                    'Duration_Above_Target_min': duration_above_target
                })
                
                if current_peak_steam_hr < min_peak_steam:
                    min_peak_steam = current_peak_steam_hr
                    best_delay = delay
                    min_peak_duration = duration_above_target
                elif current_peak_steam_hr == min_peak_steam and duration_above_target < min_peak_duration:
                    min_peak_duration = duration_above_target
                    best_delay = delay
                    
                progress_bar.progress((i + 1) / len(delay_search_range))
                status_text.text(f"Testing delay: {delay} minutes. Current Peak: {current_peak_steam_hr:.2f} kg/hr")

            st.success(f"Optimization Complete!")
            st.write(f"**Optimal Delay for 'M2&M4' found: {best_delay} minutes** (Peak Steam: {min_peak_steam:.2f} kg/hr)")
            final_delay_to_use = best_delay
            optimization_summary_df = pd.DataFrame(optimization_results)
        
        # --- Run the final simulation with the determined delay ---
        st.markdown("---")
        actual_module_delays = {
            'M1&M3': 0,
            'M2&M4': final_delay_to_use
        }
        
        optimal_schedule_df, optimal_steam_profile, optimal_resource_usage = run_simulation(
            actual_module_delays, PHASE_DURATIONS, RESOURCE_LIMITS, desorption_mode
        )

        if optimization_summary_df is not None:
            st.subheader("Optimization Results by Delay")
            st.dataframe(optimization_summary_df)

        st.subheader("Complete Cycles")
        completed_cooling_phases = optimal_schedule_df[optimal_schedule_df['Phase'] == 'Cooling']
        cycle_counts = completed_cooling_phases.groupby('Module').size().reset_index(name='Complete Cycles')

        all_modules_df = pd.DataFrame(MODULES, columns=['Module'])
        cycle_counts = pd.merge(all_modules_df, cycle_counts, on='Module', how='left').fillna(0)
        cycle_counts['Complete Cycles'] = cycle_counts['Complete Cycles'].astype(int)
        
        st.dataframe(cycle_counts)

        # === Gantt Chart with Steam Demand (using optimal results) ===
        colors = {
            'Adsorption': '#4B9CD3', 'Evacuation': '#FFB347', 'NCG Purging': '#FFD700',
            'Heating': '#E97451', 'CO2 Purging': '#90EE90', 'Cooling': '#9370DB'
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1]})

        optimal_schedule_df['Module'] = pd.Categorical(optimal_schedule_df['Module'], categories=MODULES, ordered=True)
        optimal_schedule_df = optimal_schedule_df.sort_values('Module')

        for _, row in optimal_schedule_df.iterrows():
            ax1.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                    color=colors[row['Phase']], edgecolor='black')

        ax1.set_ylabel('Module')
        ax1.set_title(f'Process Schedule (M2&M4 Delay: {final_delay_to_use} min, Mode: {desorption_mode})')
        ax1.set_xlim(0, TOTAL_MINUTES)
        ax1.set_yticks(MODULES)
        ax1.set_yticklabels(MODULES)
        ax1.invert_yaxis()
        ax1.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors.values()], colors.keys(), loc='upper right', fontsize=8)

        # Plot steam demand line
        ax2.plot(optimal_steam_profile.index, optimal_steam_profile.values, color='red', linewidth=2, label='Total Steam Demand') 
        ax2.axhline(target_steam_limit_kg_hr, color='blue', linestyle='--', label=f'Target Limit ({target_steam_limit_kg_hr} kg/hr)')
        ax2.set_title('Steam Demand (kg/hr)')
        ax2.set_ylabel('Steam Demand (kg/hr)')
        ax2.set_xlabel('Time (minutes)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
        
        # === Power Demand Profile (using optimal_resource_usage from the best simulation) ===
        st.markdown("---")
        st.subheader("Power Demand Profile")

        power_profile = np.zeros(TOTAL_MINUTES)
        for t in range(TOTAL_MINUTES):
            active_rows = optimal_schedule_df[(optimal_schedule_df['Start'] <= t) & (optimal_schedule_df['End'] > t)]
            adsorption_active_modules = active_rows[active_rows['Phase'] == 'Adsorption']
            power_profile[t] += len(adsorption_active_modules) * fan_power
            desorption_active_at_t = False
            for phase in DESORPTION_PHASES:
                if phase in active_rows['Phase'].values:
                    desorption_active_at_t = True
                    break
            if desorption_active_at_t:
                power_profile[t] += boiler_power + vpump + ctower

        peak_power = np.max(power_profile)
        peak_time = int(np.argmax(power_profile))

        st.markdown(f"**Peak Power Demand:** {peak_power:.2f} kW at **{peak_time} minutes**")

        fig_power, ax_power = plt.subplots(figsize=(15, 3))
        ax_power.plot(power_profile, color='purple', label='Power Demand')
        ax_power.axvline(peak_time, color='blue', linestyle='--', label=f'Peak @ {peak_time} min ({peak_power:.2f} kW)')
        ax_power.set_xlabel("Time (minutes)")
        ax_power.set_ylabel("Power (kW)")
        ax_power.set_title("Real-Time Power Demand")
        ax_power.legend()
        ax_power.grid(True)
        plt.tight_layout()
        st.pyplot(fig_power)

with tab3: 
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

    steam_demand_per_phase = {
        'NCG Purging': ncg_purging,
        'Heating': heating,
        'CO2 Purging': co2_purging
    }

    power_ratings = {
        'Adsorption': fan_power,
        'Evacuation': vpump,
        'NCG Purging': boiler_power + vpump + ctower,
        'Heating': boiler_power + vpump + ctower,
        'CO2 Purging': boiler_power + vpump + ctower,
        'Cooling': ctower + vpump
    }

    # === Use `delay` in your scheduling logic ===
    module_timers = {'M2': 0, 'M4': delay_m2}

    # === RESOURCE TRACKING ===
    resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
    MODULE_DELAYS = {'M2': 0, 'M4': delay_m4}
    module_timers = {mod: MODULE_DELAYS.get(mod, 0) for mod in MODULES}
    schedule = []
    desorption_lock_time = 0

    def can_allocate(phase, start, duration):
        return all(resource_usage[phase][t] == 0 for t in range(start, start + duration))

    def reserve(phase, start, duration):
        for t in range(start, start + duration):
            resource_usage[phase][t] += 1

    while True:
        progress = False
        for mod in MODULES:
            t = module_timers[mod]
            cycle_phases = []
            for phase in PHASES:
                duration = PHASE_DURATIONS[phase]
                if desorption_mode == "Serialized" and phase in DESORPTION_PHASES:
                    t = max(t, desorption_lock_time)
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
                if desorption_mode == "Serialized" and phase == 'Cooling':
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
    hatch_legend = [plt.Rectangle((0, 0), 1, 1, color='gray', edgecolor='black'),
                    plt.Rectangle((0, 0), 1, 1, color='gray', edgecolor='black', hatch='///')]
    
    # Combine legends
    all_handles = phase_legend 
    all_labels = list(colors.keys()) 
    ax1.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(1.25, 1))
    st.pyplot(fig)

    st.markdown("**Stage 1:** Cooling & Adsorption | **Stage 2:** Evacuation, NCG Purging, Heating, CO2 Purging")

    # === Power Profile ===
    # === Shared Power Profile (Avoid Double Counting Shared Equipment) ===
    power_profile = np.zeros(TOTAL_MINUTES)

    for t in range(TOTAL_MINUTES):
            # Get all active rows at time t
        active_rows = df_schedule[(df_schedule['Start'] <= t) & (df_schedule['End'] > t)]

        # === Module-specific: Adsorption (can run in parallel)
        adsorption_rows = active_rows[active_rows['Phase'] == 'Adsorption']
        power_profile[t] += len(adsorption_rows) * fan_power

        # === Shared Desorption Equipment (count once if active)
        active_phases = active_rows['Phase'].unique()

    # --- Shared equipment — only add once even if multiple modules are active ---
        if 'Evacuation' in active_phases:
            power_profile[t] += vpump

        if any(p in ['NCG Purging', 'Heating', 'CO2 Purging'] for p in active_phases):
            power_profile[t] += boiler_power + vpump + ctower  # shared steam equipment

        if 'Cooling' in active_phases:
            power_profile[t] += ctower + vpump  # shared again but still only once

        # === Peak Demand Info ===
    peak_power = np.max(power_profile)
    peak_time = int(np.argmax(power_profile))

    # === Plot Power Profile ===
    st.markdown("### Power Demand Profile")
    fig3, ax3 = plt.subplots(figsize=(15, 3))
    ax3.plot(power_profile, color='red', label='Power Demand')
    ax3.axvline(peak_time, color='blue', linestyle='--', label=f'Peak @ {peak_time} min')
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Power (kW)")
    ax3.set_title("Real-Time Power Demand")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Optional: Show peak value
    st.markdown(f"*Peak Power Demand: {peak_power:.1f} kW ~ {peak_power / 0.8:.1f} kVA at minute {peak_time}*")
    
