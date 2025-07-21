# === Streamlit App: Interleaved Desorption Scheduling & Steam Demand ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Input Configuration ===
st.title("Process Control Scheduler")

st.subheader("Cycle Times")
ad_duration = st.number_input("Adsorption Duration (min)", 10, 120, 60)
evac_duration = st.number_input("Evacuation Duration (min)", 1, 60, 6)
ncg_duration = st.number_input("NCG Purging Duration (min)", 5, 60, 13)
heat_duration = st.number_input("Heating Duration (min)", 10, 60, 17)
co2_duration = st.number_input("CO2 Purging Duration (min)", 10, 60, 53)
cool_duration = st.number_input("Cooling Duration (min)", 10, 60, 25)
delay_m2 = st.number_input("Start Delay for M2&M4 (min)", 0, 300, 90)

# === Set Steam Flowrates ===
st.subheader("Steam Demand")
ncg_purging = st.number_input("NCG Purging (kg/hr)", 0, 300, 150)
heating = st.number_input("Heating (kg/hr)", 0, 300, 150)
co2_purging = st.number_input("CO2 Purging (kg/hr)", 0, 300, 150)

PHASE_DURATIONS = {
    'Adsorption': ad_duration,
    'Evacuation': evac_duration,
    'NCG Purging': ncg_duration,
    'Heating': heat_duration,
    'CO2 Purging': co2_duration,
    'Cooling': cool_duration
}
PHASES = list(PHASE_DURATIONS.keys())
MODULES = ['M1&M3', 'M2&M4']
TOTAL_MINUTES = 1440

steam_demand_per_phase = {
    'NCG Purging': ncg_purging,
    'Heating': heating,
    'CO2 Purging': co2_purging
}
PHASE_GAPS = {
    ('NCG Purging', 'Heating'): 2,
    ('Heating', 'CO2 Purging'): 2
}

# === Resource Usage Initialization ===
resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
module_timers = {'M1&M3': 0, 'M2&M4': delay_m2}
schedule = []

# === Scheduling Functions ===
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
            while t + duration <= TOTAL_MINUTES and not can_allocate(phase, t, duration):
                t += 1
            if t + duration > TOTAL_MINUTES:
                break
            cycle_phases.append((phase, t, t + duration))
            reserve(phase, t, duration)
            next_index = PHASES.index(phase) + 1
            buffer_time = PHASE_GAPS.get((phase, PHASES[next_index]) if next_index < len(PHASES) else (None, None), 0)
            t += duration + buffer_time
        if len(cycle_phases) == len(PHASES):
            for phase, start, end in cycle_phases:
                schedule.append({"Module": mod, "Phase": phase, "Start": start, "End": end})
            module_timers[mod] = t
            progress = True
    if not progress:
        break

df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start'])

# === Display Cycle Counts ===
st.subheader("Cycle Counts")
cycle_counts = df_schedule.groupby('Module')['Phase'].apply(lambda x: x.str.fullmatch('Cooling').sum()).reset_index()
cycle_counts.columns = ['Module Pair', 'Complete Cycles']
st.dataframe(cycle_counts)

# === Steam Demand Breakdown ===
steam_breakdown = pd.DataFrame(0, index=range(TOTAL_MINUTES), columns=steam_demand_per_phase.keys())
for _, row in df_schedule.iterrows():
    if row['Phase'] in steam_demand_per_phase:
        for t in range(row['Start'], row['End']):
            steam_breakdown.loc[t, row['Phase']] += steam_demand_per_phase[row['Phase']]

steam_profile = steam_breakdown.sum(axis=1)

# === Gantt Chart ===
st.subheader("Gantt Chart")
colors = {
    'Adsorption': '#4B9CD3',
    'Evacuation': '#FFB347',
    'NCG Purging': '#FFD700',
    'Heating': '#E97451',
    'CO2 Purging': '#90EE90',
    'Cooling': '#9370DB'
}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
for _, row in df_schedule.iterrows():
    ax1.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
             color=colors[row['Phase']], edgecolor='black')
ax1.set_ylabel('Module Pair')
ax1.set_title('Process Control Schedule')
ax1.set_xlim(0, TOTAL_MINUTES)
ax1.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors.values()], colors.keys(), loc='upper right', fontsize=8)

ax2.plot(steam_profile.index, steam_profile.values, color='red', linewidth=2)
ax2.set_title('Overall Steam Demand (kg/hr)')
ax2.set_ylabel('Steam Demand (kg/hr)')
ax2.set_xlabel('Time (minutes)')
ax2.grid(True)
st.pyplot(fig)


