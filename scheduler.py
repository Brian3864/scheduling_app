# === Streamlit GUI for Flexible Scheduling with Overlap Analysis ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === CONFIGURATION INPUTS ===
st.title("Flexible Module Scheduler")

modules_count = st.slider("Number of Modules", 1, 6, 4)
ad_duration = st.number_input("Adsorption Duration (min)", 10, 120, 60)
evac_duration = st.number_input("Evacuation Duration (min)", 1, 60, 5)
ncg_duration = st.number_input("NCG Purging Duration (min)", 5, 60, 10)
heat_duration = st.number_input("Heating Duration (min)", 10, 60, 20)
co2_duration = st.number_input("CO₂ Purging Duration (min)", 5, 60, 10)
cool_duration = st.number_input("Cooling Duration (min)", 10, 60, 30)

st.markdown("### Resource Constraints")
ads_limit = st.slider("Adsorption Capacity", 1, 4, 2)
evac_limit = st.slider("Evacuation Capacity", 1, 2, 1)
ncg_limit = st.slider("NCG Purging Capacity", 1, 2, 1)
heat_limit = st.slider("Heating Capacity", 1, 2, 1)
co2_limit = st.slider("CO₂ Purging Capacity", 1, 2, 1)
cool_limit = st.slider("Cooling Capacity", 1, 2, 1)

# === SETUP ===
PHASE_DURATIONS = {
    'Adsorption': ad_duration,
    'Evacuation': evac_duration,
    'NCG Purging': ncg_duration,
    'Heating': heat_duration,
    'CO2 Purging': co2_duration,
    'Cooling': cool_duration
}

RESOURCE_LIMITS = {
    'Adsorption': ads_limit,
    'Evacuation': evac_limit,
    'NCG Purging': ncg_limit,
    'Heating': heat_limit,
    'CO2 Purging': co2_limit,
    'Cooling': cool_limit
}

PHASES = list(PHASE_DURATIONS.keys())
MODULES = [f"M{i+1}" for i in range(modules_count)]
TOTAL_MINUTES = 1440

resource_usage = {phase: [0] * TOTAL_MINUTES for phase in PHASES}
module_timers = {mod: 0 for mod in MODULES}
schedule = []

# === CORE SCHEDULING FUNCTION ===
def can_allocate(phase, start, duration):
    return all(resource_usage[phase][t] < RESOURCE_LIMITS[phase] for t in range(start, start + duration))

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
            t += duration
        if len(cycle_phases) == len(PHASES):
            for phase, start, end in cycle_phases:
                schedule.append({"Module": mod, "Phase": phase, "Start": start, "End": end})
            module_timers[mod] = t
            progress = True
    if not progress:
        break

# === DATAFRAME & CHART ===
df_schedule = pd.DataFrame(schedule).sort_values(by=['Module', 'Start'])

st.subheader("Schedule Table")
st.dataframe(df_schedule)

# === CYCLE COUNT ===
cycle_counts = df_schedule.groupby('Module')['Phase'].apply(lambda x: x.str.fullmatch('Cooling').sum()).reset_index()
cycle_counts.columns = ['Module', 'Complete Cycles']
st.subheader("Cycle Counts")
st.dataframe(cycle_counts)

# === GANTT CHART ===
colors = {
    'Adsorption': '#4B9CD3',
    'Evacuation': '#FFB347',
    'NCG Purging': '#FFD700',
    'Heating': '#E97451',
    'CO2 Purging': '#90EE90',
    'Cooling': '#9370DB'
}

fig, ax = plt.subplots(figsize=(12, 5))
for _, row in df_schedule.iterrows():
    ax.barh(row['Module'], row['End'] - row['Start'], left=row['Start'],
            color=colors[row['Phase']], edgecolor='black')

ax.set_xlabel('Time (minutes)')
ax.set_title('Gantt Chart - Module Schedule')
ax.set_xlim(0, TOTAL_MINUTES)
legend_handles = [mpatches.Patch(color=color, label=phase) for phase, color in colors.items()]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
st.pyplot(fig)

# === OVERLAP PLOT (Steam Demand) ===
steam_phases = ['Heating', 'CO2 Purging']
overlap_series = pd.DataFrame({phase: [0]*TOTAL_MINUTES for phase in steam_phases})
for _, row in df_schedule.iterrows():
    if row['Phase'] in steam_phases:
        overlap_series.loc[row['Start']:row['End']-1, row['Phase']] += 1

st.subheader("Steam Demand Over Time")
fig2, ax2 = plt.subplots(figsize=(12, 3))
overlap_series.plot(ax=ax2)
ax2.set_ylabel("# Modules Using Resource")
ax2.set_xlabel("Time (min)")
ax2.set_title("Steam-Intensive Phase Overlaps")
ax2.set_xlim(0, TOTAL_MINUTES)
st.pyplot(fig2)
