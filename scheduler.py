# === Streamlit App: Interleaved Desorption Scheduling ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

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
Models a Carbon Nest schedule for a 16-module plant, grouped into three pairs — Pair 1, Pair 2, and Pair 3 (Groups A, B, and C).
- Each pair cycles through three phase groups: Adsorption, the Desorption chain (Evacuation → NCG Purging → Heating → CO2 Purging), and Cooling
- Adsorption is the only phase group that can run for two pairs at once; the Desorption chain and Cooling are each limited to one pair at a time across the whole plant
- Evacuation takes priority over Cooling: if another pair is ready to begin its Desorption chain while a pair is still cooling, that pair's Cooling pauses and resumes with its remaining duration as soon as the conflicting Evacuation ends
- Gantt charts, complete-cycle counts, and phase breakdowns (total minutes per phase) are generated once phase durations are filled in for every pair and the schedule is generated

*Note: schedule quality depends heavily on the phase durations entered — configure realistic per-phase timings for each pair before drawing conclusions from the results.*
""")

st.markdown("<h1 style='text-align: center;'>Nelion Cycle Schedule</h1>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["General Test", "M2&M4 + LRVP", "Full Schedule Analysis", "Advanced Interleaved"])

if False:  # Module pair analysis removed (fan pairing fixed)
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
              """Create a combined metrics table olycles, CO2, utilization, and active minutes."""
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
 
with tab3:
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
    if "phase_durations_tab3" not in st.session_state:
        st.session_state.phase_durations_tab3 = pd.DataFrame({
            "Phase": PHASES,
            "Group A (min)": [25, 7, 2, 20, 40, 30],
            "Group B (min)": [25, 7, 2, 20, 40, 30],
        })
    df = st.session_state.phase_durations_tab3.copy()
    if "Capacity" in df.columns:
        df = df.drop(columns=["Capacity"])
        st.session_state.phase_durations_tab3 = df
    phase_edited = st.data_editor(
        st.session_state.phase_durations_tab3,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Phase": st.column_config.TextColumn("Phase", disabled=True),
            "Group A (min)": st.column_config.NumberColumn("Group A (min)", min_value=0, max_value=120, default=25, required=True),
            "Group B (min)": st.column_config.NumberColumn("Group B (min)", min_value=0, max_value=120, default=25, required=True),
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
    MODULE_LABELS = [f"M{m}" for m in MODULES]

    GROUP_IDS = ["A", "B"]
    GROUP_OF = {m: ("A" if m % 2 == 1 else "B") for m in MODULES}

    def adjusted_duration(phase, base_duration):
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
            end_time = start + duration
            end_safe = min(end_time, TOTAL_MINUTES)
            current_resource_state[phase][start:end_safe] += 1

        def evac_cool_conflict_internal(phase, start, duration, current_resource_state, mod):
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
        while progress_made:
            progress_made = False
            for mod in mod_order:
                t = module_timers[mod]
                cycle_phases = []
                group_id = group_of[mod]
                
                temp_resource_usage_for_cycle = {phase: np.copy(resource_usage[phase]) for phase in PHASES}
                cycle_success = True
                
                for phase in PHASES:
                    if phase == "Adsorption" and adsorption_dur_uniform is not None:
                        duration = adsorption_dur_uniform
                    else:
                        base_dur = phase_durations_by_group[group_id][phase] if phase_durations_by_group else phase_durations_config[phase]
                        duration = adjusted_duration(phase, base_dur)

                    attempt_start = t
                    conflict_hit = False
                    while attempt_start + duration <= TOTAL_MINUTES:
                        if evac_cool_conflict_internal(phase, attempt_start, duration, temp_resource_usage_for_cycle, mod):
                            # Delay Cooling by the OTHER group's Evacuation duration from the table
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
                        cycle_success = False
                        break

                    # reserve_internal(
                    #     phase, attempt_start, duration,
                    #     temp_resource_usage_for_cycle
                    # )
                    # cycle_phases.append((phase, attempt_start, attempt_start + duration))
                    
                    # t = attempt_start + duration

                    if phase == "Cooling" and allow_cooling_pause:

                        remaining = duration
                        cooling_segments = []
                        current_start = attempt_start

                        while remaining > 0:

                            evac_busy = temp_resource_usage_for_cycle["Evacuation"]

                            next_conflict = None

                            for t_check in range(
                                current_start,
                                min(TOTAL_MINUTES, current_start + remaining)
                            ):
                                if evac_busy[t_check] > 0:
                                    next_conflict = t_check
                                    break

                            if next_conflict is None:
                                cooling_segments.append(
                                    (current_start, current_start + remaining)
                                )
                                remaining = 0

                            else:

                                if next_conflict > current_start:

                                    run_time = next_conflict - current_start

                                    cooling_segments.append(
                                        (current_start, next_conflict)
                                    )

                                    remaining -= run_time

                                pause_end = next_conflict

                                while (
                                    pause_end < TOTAL_MINUTES and
                                    evac_busy[pause_end] > 0
                                ):
                                    pause_end += 1

                                current_start = pause_end

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

        # Cycle count per group: average cycles per module (Baseline, Concurrent, Interleaved)
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.subheader("Baseline")
            group_cycle_rows = []
            for gid in GROUP_IDS:
                mods_in_group = [m for m in MODULES if GROUP_OF[m] == gid]
                if gid == "A":
                    cycles = cycle_counts_baseline[cycle_counts_baseline["Module"].isin([f"M{m}" for m in mods_in_group])]["Complete Cycles"]
                    avg_cycles = round(cycles.mean(), 1) if len(cycles) > 0 else 0
                else:
                    avg_cycles = "—"
                group_cycle_rows.append({"Group": gid, "Avg Cycles per Module": avg_cycles})
            st.dataframe(pd.DataFrame(group_cycle_rows), use_container_width=True, hide_index=True)
            st.metric("Total Cycles", cycles_baseline)
        with res_col2:
            st.subheader("Concurrent")
            group_cycle_rows = []
            for gid in GROUP_IDS:
                mods_in_group = [m for m in MODULES if GROUP_OF[m] == gid]
                cycles = cycle_counts_conc[cycle_counts_conc["Module"].isin([f"M{m}" for m in mods_in_group])]["Complete Cycles"]
                avg_cycles = round(cycles.mean(), 1) if len(cycles) > 0 else 0
                group_cycle_rows.append({"Group": gid, "Avg Cycles per Module": avg_cycles})
            st.dataframe(pd.DataFrame(group_cycle_rows), use_container_width=True, hide_index=True)
            st.metric("Total Cycles", cycles_conc)
        with res_col3:
            st.subheader("Interleaved")
            group_cycle_rows = []
            for gid in GROUP_IDS:
                mods_in_group = [m for m in MODULES if GROUP_OF[m] == gid]
                cycles = cycle_counts_int[cycle_counts_int["Module"].isin([f"M{m}" for m in mods_in_group])]["Complete Cycles"]
                avg_cycles = round(cycles.mean(), 1) if len(cycles) > 0 else 0
                group_cycle_rows.append({"Group": gid, "Avg Cycles per Module": avg_cycles})
            st.dataframe(pd.DataFrame(group_cycle_rows), use_container_width=True, hide_index=True)
            st.metric("Total Cycles", cycles_int)

        # Complete Cycles (per module): Baseline, Concurrent, Interleaved
        with st.expander("Complete Cycles (per module)"):
            cc_col1, cc_col2, cc_col3 = st.columns(3)
            with cc_col1:
                st.caption("Baseline (Group A only)")
                st.dataframe(cycle_counts_baseline)
            with cc_col2:
                st.caption("Concurrent")
                st.dataframe(cycle_counts_conc)
            with cc_col3:
                st.caption("Interleaved")
                st.dataframe(cycle_counts_int)

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
        mod_to_y = {m: i for i, m in enumerate(MODULES)}
        bar_height = 0.8
        desorption_set = DESORPTION_PHASES | {'Cooling'}

        def draw_gantt(ax, plot_df, colors, is_concurrent, title):
            plot_df = plot_df.copy()
            plot_df['Module'] = pd.Categorical(plot_df['Module'], categories=MODULES, ordered=True)
            plot_df = plot_df.sort_values(['Module', 'Start'])
            if is_concurrent:
                def merge_contiguous(segments):
                    if not segments:
                        return []
                    merged = [list(segments[0])]
                    for s, w in segments[1:]:
                        if merged[-1][0] + merged[-1][1] == s:
                            merged[-1][1] += w
                        else:
                            merged.append([s, w])
                    return [(s, w) for s, w in merged]
                for mod in MODULES:
                    mod_df = plot_df[plot_df['Module'] == mod]
                    y_pos = mod_to_y.get(mod, min(mod - 1, len(MODULES) - 1))
                    yrange = (y_pos - bar_height / 2, bar_height)
                    des_segments = []
                    ads_segments = []
                    for _, row in mod_df.iterrows():
                        start = max(0, row['Start'])
                        end = min(TOTAL_MINUTES, row['End'])
                        if start >= end:
                            continue
                        seg = (start, end - start)
                        if row['Phase'] == 'Adsorption':
                            ads_segments.append(seg)
                        elif row['Phase'] in desorption_set:
                            des_segments.append(seg)
                    des_segments = merge_contiguous(des_segments)
                    if des_segments:
                        ax.broken_barh(des_segments, yrange, facecolors=colors['Evacuation'],
                                       edgecolors=colors['Evacuation'], linewidths=0.5, zorder=1)
                    if ads_segments:
                        ax.broken_barh(ads_segments, yrange, facecolors=colors['Adsorption'],
                                       edgecolors=colors['Adsorption'], linewidths=0.5, zorder=10)
            else:
                for _, row in plot_df.iterrows():
                    start = max(0, row['Start'])
                    end = min(TOTAL_MINUTES, row['End'])
                    if start >= end:
                        continue
                    mod = int(row['Module']) if row['Module'] is not None else 1
                    y_pos = mod_to_y.get(mod, min(mod - 1, len(MODULES) - 1))
                    c = colors.get(row['Phase'], colors['Adsorption'])
                    ax.barh(y_pos, end - start, left=start, height=bar_height, color=c, edgecolor='black')
            ax.set_yticks(range(len(MODULES)))
            ax.set_yticklabels([f"M{m}" for m in MODULES])
            ax.set_ylabel('Module')
            ax.set_xlabel('Time (minutes)')
            ax.set_title(title)
            ax.set_xlim(0, TOTAL_MINUTES)
            ax.invert_yaxis()
            if is_concurrent:
                legend_handles = [plt.Rectangle((0,0),1,1,color=colors['Adsorption']), plt.Rectangle((0,0),1,1,color=colors['Evacuation'])]
                legend_labels = ['Adsorption', 'Desorption (locked)']
            else:
                legend_handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors.values()]
                legend_labels = list(colors.keys())
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8)

        gantt_col1, gantt_col2 = st.columns(2)
        fig_height = max(8, len(MODULES) * 0.5)
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
                if desorption_mode == "Concurrent" and phase in DESORPTION_PHASES:
                    t = max(t, desorption_lock_time)
                while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                    t += 1
                if t + duration > TOTAL_MINUTES:
                    break
                cycle_phases.append((phase, t, t + duration))
                reserve(phase, t, duration)
                if desorption_mode == "Concurrent" and phase == 'Cooling':
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

        st.caption("Phase durations per pair - edit directly in the table")

    if "adv_phase_durations" not in st.session_state:
        st.session_state.adv_phase_durations = pd.DataFrame({
            "Phase": PHASES,
            "Pair 1 (min)": [25, 7, 2, 20, 40, 30, 5],
            "Pair 2 (min)": [25, 7, 2, 20, 40, 30, 5],
            "Pair 3 (min)": [25, 7, 2, 20, 40, 30, 5],
        })

    adv_phase_table = st.data_editor(
        st.session_state.adv_phase_durations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Phase": st.column_config.TextColumn("Phase", disabled=True),
            "Pair 1 (min)": st.column_config.NumberColumn("Pair 1 (min)", min_value=0, max_value=240, required=True),
            "Pair 2 (min)": st.column_config.NumberColumn("Pair 2 (min)", min_value=0, max_value=240, required=True),
            "Pair 3 (min)": st.column_config.NumberColumn("Pair 3 (min)", min_value=0, max_value=240, required=True),
        },
        key="adv_phase_editor"
    )

    st.session_state.adv_phase_durations = adv_phase_table

    PAIRS = ["Pair 1", "Pair 2", "Pair 3"]

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

    def run_advanced_interleaved():
        """
        Advanced 3-pair interleaved scheduler.

        Sequence per pair:
        Adsorption -> Evacuation -> NCG Purging -> Heating -> CO2 Purging -> Cooling -> Repressurization

        Rules:
        - Pair 1 can overlap adsorption with Pair 2 and Pair 3.
        - Pair 2 and Pair 3 adsorption cannot overlap each other.
        - Only one pair can be in Desorption chain at a time.
        - Only one pair can be in Cooling at a time.
        - Evacuation can only begin when another pair is in Repressurization.
        """

        schedule = []
        PAIRS = ["Pair 1", "Pair 2", "Pair 3"]

        DESORPTION_CHAIN = [
            "Evacuation",
            "NCG Purging",
            "Heating",
            "CO2 Purging"
        ]

        pair_next_phase = {
            "Pair 1": "Adsorption",
            "Pair 2": "Desorption",
            "Pair 3": "Cooling",
        }

        pair_ready_time = {
            "Pair 1": 0,
            "Pair 2": 0,
            "Pair 3": 0,
        }

        resource_ready_time = {
            "Desorption": 0,
            "Cooling": 0,
            "Repressurization": 0,
        }

    # def find_repressurization_start(pair, earliest_start):
        
                
        # repress_rows = [
        #     row for row in schedule
        #     if row["Phase"] == "Repressurization" and row["Pair"] != pair
        # ]

        # best_start = None

        # for row in sorted(repress_rows, key=lambda r: r["Start"]):

        #     possible_start = max(earliest_start, row["Start"])

        #     # Evacuation only needs to BEGIN during repressurization
        #     if possible_start < row["End"]:

        #         if best_start is None or possible_start < best_start:
        #             best_start = possible_start

        # return best_start

        while True:
            progress = False

            for pair in PAIRS:
                phase_group = pair_next_phase[pair]

                if phase_group == "Adsorption":
                    duration = PHASE_DURATIONS_BY_PAIR[pair]["Adsorption"]

                    # Pair 1 can overlap with Pair 2 and Pair 3.
                    # Pair 2 and Pair 3 cannot overlap each other.
                    start = pair_ready_time[pair]

                    if pair in ("Pair 2", "Pair 3"):
                        blocking_pair = "Pair 3" if pair == "Pair 2" else "Pair 2"
                        for row in schedule:
                            if row["Phase"] == "Adsorption" and row["Pair"] == blocking_pair:
                                if start < row["End"] and start + duration > row["Start"]:
                                    start = row["End"]

                    end = start + duration

                    if end > TOTAL_MINUTES_ADV:
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
                    earliest_start = max(pair_ready_time[pair], resource_ready_time["Desorption"])

                    # Evacuation cannot start unless another pair is in Repressurization
                    start = earliest_start

                    phase_start = start
                    total_desorption_duration = sum(
                        PHASE_DURATIONS_BY_PAIR[pair][phase]
                        for phase in DESORPTION_CHAIN
                    )

                    end = start + total_desorption_duration

                    if end > TOTAL_MINUTES_ADV:
                        continue

                    for phase in DESORPTION_CHAIN:
                        duration = PHASE_DURATIONS_BY_PAIR[pair][phase]

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
                    duration = PHASE_DURATIONS_BY_PAIR[pair]["Cooling"]

                    # Evacuation gets priority over Cooling: if another pair's Evacuation
                    # overlaps the Cooling window, Cooling pauses and resumes with its
                    # remaining duration as soon as that Evacuation ends.
                    other_evac_events = sorted(
                        (row for row in schedule
                         if row["Phase"] == "Evacuation" and row["Pair"] != pair),
                        key=lambda r: r["Start"]
                    ) if adv_enforce_evac_cool else []

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

                    if end > TOTAL_MINUTES_ADV:
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
                    start = max(pair_ready_time[pair], resource_ready_time["Repressurization"])
                    duration = PHASE_DURATIONS_BY_PAIR[pair]["Repressurization"]
                    end = start + duration

                    if end > TOTAL_MINUTES_ADV:
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

    if st.button("Generate Advanced Interleaved Schedule", key="adv_generate"):
        adv_schedule = run_advanced_interleaved()

        if adv_schedule is None:
            st.error("run_advanced_interleaved() returned None")
            st.stop()


        st.markdown("### Complete Cycles")

        cycle_rows = []
        for pair in PAIRS:
            pair_df = adv_schedule[adv_schedule["Pair"] == pair]
            complete_cycles = len(pair_df) // len(PHASES)
            cycle_rows.append({
                "Pair": pair,
                "Complete Cycles": complete_cycles
            })

        st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)

        st.markdown("### Advanced Interleaved Gantt Chart")

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