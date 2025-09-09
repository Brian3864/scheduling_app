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
tab1, tab2, tab3, tab4 = st.tabs(["General Test", "M2&M4 + LRVP", "4 MODULES", "Automatic Optimization"])

with tab3:
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
 
with tab4:
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
    