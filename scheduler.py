# === Streamlit App: Interleaved Desorption Scheduling & Steam Demand ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
tab1, tab2 = st.tabs(["2 MODULES", "4 MODULES"])

with tab2:
    # === MODULES ===
    MODULES = ["M1&M3", "M2&M4"]

    # === PHASES ===
    PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
    DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling'}

    # === Input Configuration ===
    st.markdown("<h2 style='text-align: center;'>Input Configuration</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        TOTAL_MINUTES = 1440
        desorption_mode = st.radio("Desorption Strategy", options=["Serialized", "Interleaved"], index=1)
        delay_m2 = st.number_input("Start Delay for M2&M4 (min)", 0, 300, 83)

    with col2:
        ad_duration = st.number_input("Adsorption Duration (min)", 10, 120, 60)
        evac_duration = st.number_input("Evacuation Duration (min)", 1, 60, 6)
        ncg_duration = st.number_input("NCG Purging Duration (min)", 5, 60, 13)
    
    with col3:
        heat_duration = st.number_input("Heating Duration (min)", 10, 60, 17)
        co2_duration = st.number_input("CO2 Purging Duration (min)", 10, 60, 53)
        cool_duration = st.number_input("Cooling Duration (min)", 10, 60, 25)       

    PHASE_DURATIONS = {
        'Adsorption': ad_duration,
        'Evacuation': evac_duration,
        'NCG Purging': ncg_duration,
        'Heating': heat_duration,
        'CO2 Purging': co2_duration,
        'Cooling': cool_duration
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
    ax1.set_ylabel("Module Pair")
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
        adsorption_duration = st.number_input("*Adsorption (min)*", 10, 240, 60)
        desorption_duration = st.number_input("*Desorption (min)*", 10, 240, 120)

# === Set Steam Flowrates ===   
    with col2:
        st.markdown("Resource Limits")
        ads_mod = st.number_input("*Adsorbing Modules*", 1, 2, 2)
        des_mod = st.number_input("*Desorbing Modules*", 1, 2, 2)

    RESOURCE_LIMITS = {'Adsorption': ads_mod, 
                       'Desorption': des_mod}
    PHASE_DURATIONS = {'Adsorption': adsorption_duration,
                        'Desorption': desorption_duration}
    PHASES = list(PHASE_DURATIONS.keys())
    MODULES = ['M1', 'M2']
    TOTAL_MINUTES = 1440

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
    for _, row in df_schedule.iterrows():
        ax.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
                color=colors[row['Phase']], edgecolor='black')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Process Sequence')
    ax.set_xlim(0, TOTAL_MINUTES)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=colors.keys(), loc='upper right', fontsize=8)
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
 