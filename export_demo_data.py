"""
Export K-Secretary Demo Data to JSON for Web Integration

This script pre-computes all necessary data for the interactive web demo
and exports it to JSON files that can be loaded by JavaScript.
"""

import numpy as np
from scipy.special import comb
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from itertools import combinations
import json
import os
from tqdm import tqdm

# Configuration
n_demo = 30
K_demo = 2
COEFF_THRESHOLD = 1e-3
DATA_DIR = "data"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("="*60)
print("K-SECRETARY DEMO DATA EXPORT")
print("="*60)
print(f"Configuration: n={n_demo}, K={K_demo}")
print(f"Output directory: {DATA_DIR}/")
print("="*60)

# ================================
# HELPER FUNCTIONS (from notebook)
# ================================

def compute_delta(n, i, k, ell):
    """Compute delta_{k|ell}(i)"""
    if i < 1 or k < 1 or ell < k or k > i:
        return 0.0

    numerator = comb(n - i, ell - k, exact=True) * comb(i - 1, k - 1, exact=True)
    denominator = comb(n - 1, ell - 1, exact=True)

    return numerator / denominator if denominator > 0 else 0.0

def get_meaningful_vars(n, K, coeff_threshold=COEFF_THRESHOLD):
    """Identify variables with non-negligible objective coefficients."""
    delta = np.zeros((n, K, K))
    for i in range(1, n + 1):
        for k in range(1, K + 1):
            for ell in range(k, K + 1):
                delta[i-1, k-1, ell-1] = compute_delta(n, i, k, ell)

    c_dict = {}
    for j in range(K):
        for k in range(K):
            for i in range(n):
                c_val = 0.0
                for ell in range(k, K):
                    c_val += (1.0 / n) * delta[i, k, ell]
                c_dict[(j, k, i)] = c_val

    meaningful_vars = []
    for j in range(K):
        for k in range(K):
            for i in range(n):
                if i < k or i == 0:
                    continue
                if abs(c_dict[(j, k, i)]) > coeff_threshold:
                    meaningful_vars.append((j, k, i))

    return meaningful_vars, c_dict

def create_k_secretary_lp_model_restricted(n, K, meaningful_vars):
    """Creates LP with only meaningful variables."""
    delta = np.zeros((n, K, K))
    for i in range(1, n + 1):
        for k in range(1, K + 1):
            for ell in range(k, K + 1):
                delta[i-1, k-1, ell-1] = compute_delta(n, i, k, ell)

    model = pyo.ConcreteModel()
    model.VARS = pyo.Set(initialize=meaningful_vars)
    model.z = pyo.Var(model.VARS, bounds=(0.0, 1.0))

    c_dict = {}
    for j, k, i in model.VARS:
        c_val = 0.0
        for ell in range(k, K):
            c_val += (1.0 / n) * delta[i, k, ell]
        c_dict[(j,k,i)] = c_val

    model.objective = pyo.Objective(
        expr=sum(c_dict[v] * model.z[v] for v in model.VARS),
        sense=pyo.maximize
    )

    model.lp_constrs = pyo.ConstraintList()

    for j in range(K - 1):
        for k in range(K):
            for i in range(1, n):
                if (j, k, i) not in meaningful_vars:
                    continue

                rhs = 0
                for m in range(i):
                    for ell in range(K):
                        if (j+1, ell, m) in meaningful_vars:
                            rhs += (1.0 / (m + 1)) * model.z[(j+1, ell, m)]
                        if (j, ell, m) in meaningful_vars:
                            rhs -= (1.0 / (m + 1)) * model.z[(j, ell, m)]

                model.lp_constrs.add(model.z[(j, k, i)] <= rhs)

    j = K - 1
    for k in range(K):
        for i in range(1, n):
            if (j, k, i) not in meaningful_vars:
                continue

            rhs = 1.0
            for m in range(i):
                for ell in range(K):
                    if (j, ell, m) in meaningful_vars:
                        rhs -= (1.0 / (m + 1)) * model.z[(j, ell, m)]

            model.lp_constrs.add(model.z[(j, k, i)] <= rhs)

    return model

def solve_v_worst_enumeration(n, K, B, vars_to_enumerate, all_meaningful_vars):
    """Solve V_worst(B) by enumeration."""
    v_worst = float('inf')
    worst_vars = None
    all_performances = []
    opt = SolverFactory('gurobi')

    # Calculate total combinations for progress bar
    from math import comb as math_comb
    total_combinations = math_comb(len(vars_to_enumerate), B)

    with tqdm(total=total_combinations, desc=f"  Solving B={B}", leave=False) as pbar:
        for vars_to_fix in combinations(vars_to_enumerate, B):
            model = create_k_secretary_lp_model_restricted(n, K, all_meaningful_vars)

            for v in vars_to_fix:
                model.z[v].fix(0.0)

            results = opt.solve(model, tee=False)

            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                obj_val = pyo.value(model.objective)
                all_performances.append((obj_val, vars_to_fix))
                if obj_val < v_worst:
                    v_worst = obj_val
                    worst_vars = vars_to_fix

            for v in vars_to_fix:
                model.z[v].unfix()

            pbar.update(1)

    all_performances.sort(key=lambda x: x[0])
    return v_worst, list(worst_vars) if worst_vars else [], all_performances

# ================================
# COMPUTE BASE SOLUTION
# ================================

print("\n[1/4] Computing base solution (B=0)...")

delta = np.zeros((n_demo, K_demo, K_demo))
for i in range(1, n_demo + 1):
    for k in range(1, K_demo + 1):
        for ell in range(k, K_demo + 1):
            delta[i-1, k-1, ell-1] = compute_delta(n_demo, i, k, ell)

model = pyo.ConcreteModel()
model.J = pyo.Set(initialize=range(K_demo))
model.K = pyo.Set(initialize=range(K_demo))
model.I = pyo.Set(initialize=range(n_demo))
model.VARS = model.J * model.K * model.I
model.z = pyo.Var(model.VARS, bounds=(0.0, 1.0))

c_dict = {}
for j, k, i in model.VARS:
    c_val = 0.0
    for ell in range(k, K_demo):
        c_val += (1.0 / n_demo) * delta[i, k, ell]
    c_dict[(j,k,i)] = c_val

model.objective = pyo.Objective(
    expr=sum(c_dict[v] * model.z[v] for v in model.VARS),
    sense=pyo.maximize
)

model.lp_constrs = pyo.ConstraintList()

for j in range(K_demo - 1):
    for k in range(K_demo):
        for i in range(1, n_demo):
            rhs = 0
            for m in range(i):
                for ell in range(K_demo):
                    rhs += (1.0 / (m + 1)) * (model.z[j+1, ell, m] - model.z[j, ell, m])
            model.lp_constrs.add(model.z[j, k, i] <= rhs)

j = K_demo - 1
for k in range(K_demo):
    for i in range(1, n_demo):
        rhs = 1.0
        for m in range(i):
            for ell in range(K_demo):
                rhs -= (1.0 / (m + 1)) * model.z[j, ell, m]
        model.lp_constrs.add(model.z[j, k, i] <= rhs)

opt = SolverFactory('gurobi')
results = opt.solve(model, tee=False)
obj_val_base = pyo.value(model.objective)

print(f"  ✓ V_best = {obj_val_base:.6f} (CR = {obj_val_base/K_demo:.6f})")

# Get meaningful variables
all_meaningful_vars, c_dict = get_meaningful_vars(n_demo, K_demo)
print(f"  ✓ Meaningful variables: {len(all_meaningful_vars)}")

# Export base configuration
base_config = {
    "n": n_demo,
    "K": K_demo,
    "v_best": obj_val_base,
    "cr_best": obj_val_base / K_demo,
    "num_meaningful_vars": len(all_meaningful_vars)
}

with open(f"{DATA_DIR}/base_config.json", 'w') as f:
    json.dump(base_config, f, indent=2)
print(f"  ✓ Saved: {DATA_DIR}/base_config.json")

# ================================
# EXPORT V_WORST DATA
# ================================

print("\n[2/4] Computing V_worst for all configurations...")

v_worst_data = {}

# Create list of all configurations to process
configs_to_process = [(q, k) for q in range(K_demo) for k in range(K_demo)]

for quota_idx, k_pot_idx in tqdm(configs_to_process, desc="Configurations", position=0):
    quota_name = f"Q{quota_idx+1}"
    k_pot_name = f"k{k_pot_idx+1}"
    config_key = f"{quota_name}_{k_pot_name}"

    # Filter variables
    vars_to_enum = [v for v in all_meaningful_vars
                    if v[0] == quota_idx and v[1] == k_pot_idx]

    print(f"\n  Computing {config_key}: {len(vars_to_enum)} variables")

    v_worst_data[config_key] = {
        "quota": quota_idx,
        "k_potential": k_pot_idx,
        "num_vars": len(vars_to_enum),
        "budgets": {}
    }

    # Compute for B = 1 to 3 (or max available)
    max_B = min(3, len(vars_to_enum))

    for B in tqdm(range(1, max_B + 1), desc=f"  {config_key} budgets", leave=False, position=1):
        v_worst, worst_vars, _ = solve_v_worst_enumeration(
            n_demo, K_demo, B, vars_to_enum, all_meaningful_vars
        )

        v_worst_data[config_key]["budgets"][str(B)] = {
            "v_worst": v_worst,
            "cr_worst": v_worst / K_demo,
            "gap_pct": (obj_val_base - v_worst) / obj_val_base * 100,
            "worst_vars": [[int(v[0]), int(v[1]), int(v[2])] for v in worst_vars],
            "worst_vars_readable": [
                f"Q{v[0]+1}, {v[1]+1}-pot, step {v[2]+1}" for v in worst_vars
            ]
        }

        print(f"    B={B}: V_worst={v_worst:.6f}, CR={v_worst/K_demo:.6f}")

# Save V_worst data
with open(f"{DATA_DIR}/v_worst_all.json", 'w') as f:
    json.dump(v_worst_data, f, indent=2)
print(f"  ✓ Saved: {DATA_DIR}/v_worst_all.json")

# ================================
# EXPORT SPECTRUM DATA
# ================================

print("\n[3/4] Computing α-spectrum data...")

spectrum_data = {}

for quota_idx, k_pot_idx in tqdm(configs_to_process, desc="Spectrum configs", position=0):
    vars_to_enum = [v for v in all_meaningful_vars
                    if v[0] == quota_idx and v[1] == k_pot_idx]

    quota_name = f"Q{quota_idx+1}"
    k_pot_name = f"k{k_pot_idx+1}"
    config_key = f"{quota_name}_{k_pot_name}"

    print(f"\n  Computing spectrum for {config_key}")

    spectrum_data[config_key] = {}

    # Compute for B = 1, 2, 3
    budgets_to_process = [B for B in [1, 2, 3] if len(vars_to_enum) >= B]
    for B in tqdm(budgets_to_process, desc=f"  {config_key} spectrum", leave=False, position=1):
        print(f"    B={B}...")

        # Get all performances
        v_worst, worst_vars, all_performances = solve_v_worst_enumeration(
            n_demo, K_demo, B, vars_to_enum, all_meaningful_vars
        )

        v_best = obj_val_base
        alpha_step = 0.05
        alpha_grid = np.arange(0, 1.0 + alpha_step, alpha_step)

        spectrum_results = {
            "alpha": [],
            "performance": [],
            "cr": [],
            "threshold": [],
            "steps": []
        }

        for alpha in alpha_grid:
            threshold = (1 - alpha) * v_best + alpha * v_worst

            perf_found = None
            vars_found = None

            for perf, vars_fixed in all_performances:
                if perf >= threshold:
                    perf_found = perf
                    vars_found = vars_fixed
                    break

            if perf_found is not None:
                spectrum_results["alpha"].append(float(alpha))
                spectrum_results["performance"].append(float(perf_found))
                spectrum_results["cr"].append(float(perf_found / K_demo))
                spectrum_results["threshold"].append(float(threshold))

                if vars_found:
                    steps = [int(v[2] + 1) for v in vars_found]
                    spectrum_results["steps"].append(steps)
                else:
                    spectrum_results["steps"].append([])

        spectrum_data[config_key][f"B{B}"] = {
            "v_worst": float(v_worst),
            "v_best": float(v_best),
            "spectrum": spectrum_results
        }

# Save spectrum data
with open(f"{DATA_DIR}/spectrum_all.json", 'w') as f:
    json.dump(spectrum_data, f, indent=2)
print(f"  ✓ Saved: {DATA_DIR}/spectrum_all.json")

# ================================
# EXPORT 3D SURFACE DATA
# ================================

print("\n[4/4] Computing 3D surface data...")

surface_data = {}

for quota_idx, k_pot_idx in tqdm(configs_to_process, desc="Surface configs", position=0):
    vars_to_enum = [v for v in all_meaningful_vars
                    if v[0] == quota_idx and v[1] == k_pot_idx]

    quota_name = f"Q{quota_idx+1}"
    k_pot_name = f"k{k_pot_idx+1}"
    config_key = f"{quota_name}_{k_pot_name}"

    print(f"\n  Computing 3D surface for {config_key}")

    B_max = min(3, len(vars_to_enum))
    alpha_step = 0.1
    alpha_grid = np.arange(0, 1.0 + alpha_step, alpha_step)
    B_range = range(1, B_max + 1)

    # Precompute all performances for each B
    all_performances_by_B = {}
    v_worst_by_B = {}

    for B in tqdm(B_range, desc=f"  {config_key} surface", leave=False, position=1):
        v_worst, worst_vars, all_performances = solve_v_worst_enumeration(
            n_demo, K_demo, B, vars_to_enum, all_meaningful_vars
        )
        all_performances_by_B[B] = all_performances
        v_worst_by_B[B] = v_worst
        print(f"    B={B}: V_worst={v_worst:.6f}")

    # Build CR surface
    cr_surface = []

    for B in B_range:
        v_worst = v_worst_by_B[B]
        all_performances = all_performances_by_B[B]
        cr_row = []

        for alpha in alpha_grid:
            threshold = (1 - alpha) * obj_val_base + alpha * v_worst

            perf_found = obj_val_base
            for perf, _ in all_performances:
                if perf >= threshold:
                    perf_found = perf
                    break

            cr_row.append(float(perf_found / K_demo))

        cr_surface.append(cr_row)

    surface_data[config_key] = {
        "alpha_grid": [float(a) for a in alpha_grid],
        "B_range": list(B_range),
        "cr_surface": cr_surface,
        "v_worst_by_B": {str(B): float(v_worst_by_B[B]) for B in B_range}
    }

# Save surface data
with open(f"{DATA_DIR}/surface_all.json", 'w') as f:
    json.dump(surface_data, f, indent=2)
print(f"  ✓ Saved: {DATA_DIR}/surface_all.json")

# ================================
# SUMMARY
# ================================

print("\n" + "="*60)
print("EXPORT COMPLETE!")
print("="*60)
print(f"\nGenerated files in {DATA_DIR}/:")
print("  - base_config.json       : Base problem configuration")
print("  - v_worst_all.json       : V_worst values for all configs")
print("  - spectrum_all.json      : α-spectrum data")
print("  - surface_all.json       : 3D surface data")
print("\nReady for web integration!")
print("="*60)
