"""
Quantified analysis: Baseline vs Concurrent vs Interleaved.
Configurable cycle times, 4 modules, caps=2. Run: python run_analysis.py
"""
import numpy as np
import pandas as pd

# === CONFIG: Edit cycle times here ===
PHASES = ['Adsorption', 'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging', 'Cooling']
GRP_A = {'Adsorption': 60, 'Evacuation': 3, 'NCG Purging': 10, 'Heating': 0, 'CO2 Purging': 45, 'Cooling': 30}
GRP_B = {'Adsorption': 60, 'Evacuation': 6, 'NCG Purging': 7, 'Heating': 0, 'CO2 Purging': 30, 'Cooling': 35}
# ===

DESORPTION_PHASES = {'Evacuation', 'NCG Purging', 'Heating', 'CO2 Purging'}
PHASE_GAPS = {('NCG Purging', 'Heating'): 0, ('Heating', 'CO2 Purging'): 0}
TOTAL_MINUTES, MODULES = 1440, [1, 2, 3, 4]
GROUP_OF, GROUP_IDS = {1: "A", 2: "B", 3: "A", 4: "B"}, ["A", "B"]
adsorption_cap = shared_evac_cooling_cap = shared_purge_heat_co2_cap = concurrent_desorption_cap = 2
enforce_evac_cool = True


def build_delays(offset):
    return {m: (0 if GROUP_OF[m] == "A" else offset) for m in MODULES}


def run_sim(delays, mode, phase_by_group, modules=None, baseline=False):
    mods = modules or MODULES
    ru = {p: np.zeros(TOTAL_MINUTES, dtype=int) for p in PHASES}
    mt = {m: delays.get(m, 0) for m in mods}
    sched = []
    order = sorted(mods, key=lambda m: (GROUP_OF[m] != "A", m))
    while True:
        progress = False
        for mod in order:
            t, group = mt[mod], GROUP_OF[mod]
            tmp = {p: np.copy(ru[p]) for p in PHASES}
            cyc, ok = [], True
            for ph in PHASES:
                dur = max(0, int(phase_by_group[group][ph]))
                att = t
                while att + dur <= TOTAL_MINUTES:
                    conflict = False
                    if not baseline and mode == "Interleaved":
                        if ph == "Cooling" and np.any(tmp["Evacuation"][att:att + dur] > 0):
                            conflict = True
                        elif ph == "Evacuation" and not enforce_evac_cool and np.any(tmp["Cooling"][att:att + dur] > 0):
                            conflict = True
                    if conflict:
                        att += 1
                        continue
                    ok_alloc = True
                    if ph == "Adsorption":
                        ok_alloc = np.all(tmp[ph][att:att + dur] < adsorption_cap)
                    elif baseline and mode == "Interleaved":
                        ok_alloc = np.all(tmp[ph][att:att + dur] < shared_evac_cooling_cap) if ph in ("Evacuation", "Cooling") else np.all(tmp[ph][att:att + dur] < shared_purge_heat_co2_cap)
                    elif mode == "Interleaved":
                        if ph in ("Evacuation", "Cooling"):
                            for tt in range(att, min(att + dur, TOTAL_MINUTES)):
                                if tmp["Evacuation"][tt] + tmp["Cooling"][tt] >= shared_evac_cooling_cap:
                                    ok_alloc = False
                                    break
                        elif ph in ("NCG Purging", "Heating", "CO2 Purging"):
                            for tt in range(att, min(att + dur, TOTAL_MINUTES)):
                                if tmp["NCG Purging"][tt] + tmp["Heating"][tt] + tmp["CO2 Purging"][tt] >= shared_purge_heat_co2_cap:
                                    ok_alloc = False
                                    break
                    elif mode == "Concurrent" and ph == "Evacuation":
                        des = list(DESORPTION_PHASES) + ["Cooling"]
                        for tt in range(att, min(att + dur, TOTAL_MINUTES)):
                            if sum(tmp[x][tt] for x in des) >= concurrent_desorption_cap:
                                ok_alloc = False
                                break
                    if not ok_alloc:
                        att += 1
                        continue
                    break
                if att + dur > TOTAL_MINUTES:
                    ok = False
                    break
                es = min(att + dur, TOTAL_MINUTES)
                tmp[ph][att:es] += 1
                cyc.append((ph, att, att + dur))
                t = att + dur
                ni = PHASES.index(ph) + 1
                if ni < len(PHASES):
                    t += PHASE_GAPS.get((ph, PHASES[ni]), 0)
            if ok:
                for p in PHASES:
                    ru[p] = np.copy(tmp[p])
                for ph, st, en in cyc:
                    sched.append({"Module": mod, "Phase": ph, "Start": st, "End": en})
                mt[mod], progress = t, True
        if not progress:
            break
    return pd.DataFrame(sched)


def count_cycles(df):
    phase_order = {p: i for i, p in enumerate(PHASES)}
    tot = 0
    for m in MODULES:
        md = df[df["Module"] == m].copy()
        md["_order"] = md["Phase"].map(phase_order)
        md = md.sort_values(["Start", "_order"]).drop(columns=["_order"]).reset_index(drop=True)
        i = 0
        while i <= len(md) - len(PHASES):
            w = md.iloc[i : i + len(PHASES)]
            if list(w["Phase"]) == PHASES:
                tot += 1
                i += len(PHASES)
            else:
                i += 1
    return tot


def run_analysis(grp_a, grp_b, mods_A=None):
    """Run full analysis and return results."""
    mods_A = mods_A or [1, 3]
    phase_by_group = {"A": grp_a, "B": grp_b}
    best_d, best_i = 0, -1
    for d in range(0, 120, 4):
        delays = build_delays(d)
        si = run_sim(delays, "Interleaved", phase_by_group)
        c = count_cycles(si)
        if c > best_i:
            best_i, best_d = c, d
    delays = build_delays(best_d)
    cb = count_cycles(run_sim({m: 0 for m in mods_A}, "Interleaved", phase_by_group, mods_A, baseline=True))
    cc = count_cycles(run_sim(delays, "Concurrent", phase_by_group))
    ci = count_cycles(run_sim(delays, "Interleaved", phase_by_group))
    return cb, cc, ci, best_d


if __name__ == "__main__":
    cycle_a = sum(GRP_A.values())
    cycle_b = sum(GRP_B.values())
    print("=== Quantified Analysis (4 modules, caps=2, 1440 min) ===")
    print(f"Group A cycle: {cycle_a} min | Group B cycle: {cycle_b} min")
    theo_a = 1440 / cycle_a
    theo_b = 1440 / cycle_b
    print(f"Theoretical: {theo_a:.1f} cycles/module (A) | {theo_b:.1f} cycles/module (B) | 2 modules baseline max ~{int(2*theo_a)} (A only)")
    print()
    results = []
    # 1. Both equal to A
    cb, cc, ci, best_d = run_analysis(GRP_A, GRP_A)
    results.append(("Both = A (equal, 148 min)", cb, cc, ci, best_d))
    # 2. Both equal to B
    cb, cc, ci, best_d = run_analysis(GRP_B, GRP_B)
    results.append(("Both = B (equal, 138 min)", cb, cc, ci, best_d))
    # 3. Different (A and B)
    cb, cc, ci, best_d = run_analysis(GRP_A, GRP_B)
    results.append(("Different (A=148, B=138 min)", cb, cc, ci, best_d))
    print("Scenario                      | Baseline | Concurrent | Interleaved | Conc vs base | Int vs base | Int vs Conc | Best delay")
    print("-" * 105)
    for name, cb, cc, ci, bd in results:
        pct_conc = 100 * (cc - cb) / cb if cb > 0 else 0
        pct_int = 100 * (ci - cb) / cb if cb > 0 else 0
        pct_int_conc = 100 * (ci - cc) / cc if cc > 0 else 0
        print(f"{name:30} | {cb:8} | {cc:10} | {ci:11} | {pct_conc:+10.1f}% | {pct_int:+10.1f}% | {pct_int_conc:+10.1f}% | {bd} min")
