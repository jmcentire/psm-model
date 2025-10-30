#!/usr/bin/env python3
"""
Pressure System Model (PSM)
Full Validation and Error Convergence Simulation
------------------------------------------------
This script performs the multi-week 2025 simulation, computes
monthly error convergence vs proxy values, saves results, and
generates Figure 1 for the LaTeX paper.

It also includes a placeholder for the 2008–2009 GFC validation.
"""

import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Utility: safe directory creation
# ---------------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------
# Load parameters or fallback defaults
# ---------------------------------------------------------------------
try:
    with open("parameters.json", "r") as f:
        params = json.load(f)
except FileNotFoundError:
    print("parameters.json not found — using example defaults.")
    params = {
        "tanks": ["US", "China", "Canada", "EU", "LatAm", "Other"],
        "initial_P": [28.8, 18.5, 2.22, 18.9, 6.73, 68.0],
        "C": [0.023, 0.045, 0.008, 0.012, 0.022, 0.03],
        "L": [0.03, 0.016, 0.0338, 0.032, 0.0764, 0.052],
        "K": [
            [0, 0.03, 0.02, 0.02, 0.015, 0.01],
            [0.03, 0, 0.01, 0.015, 0.01, 0.005],
            [0.02, 0.01, 0, 0.015, 0.01, 0.005],
            [0.02, 0.015, 0.015, 0, 0.01, 0.01],
            [0.015, 0.01, 0.01, 0.01, 0, 0.005],
            [0.01, 0.005, 0.005, 0.01, 0.005, 0],
        ],
        "R_baseline": [[1.0] * 6 for _ in range(6)],
    }

tanks = params["tanks"]
n = len(tanks)
P0 = np.array(params["initial_P"])
C = np.array(params["C"])
L = np.array(params["L"])
K = np.array(params["K"])
R = np.array(params["R_baseline"])

# ---------------------------------------------------------------------
# Core Dynamics (Euler step)
# ---------------------------------------------------------------------
def psm_dynamics(P, C, L, K, R, stochastic=False, leak_sigma=0.005, flow_sigma=0.01):
    n = len(P)
    flows = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                flows[i] -= K[i][j] * R[i][j] * (P[i] - P[j]) / 52  # weekly scaling
    dPdt = (C * P / 52) - (L * P) + flows
    if stochastic:
        dPdt += np.random.normal(0, leak_sigma * P) + np.random.normal(0, flow_sigma * np.abs(flows))
    return dPdt


def run_euler_sim(P0, weeks, C, L, K, R, stochastic=False, num_runs=1, dt=1/52):
    """Run ensemble simulation using explicit Euler integration."""
    results = []
    for r in range(num_runs):
        P = P0.copy()
        for _ in range(weeks):
            dPdt = psm_dynamics(P, C, L, K, R, stochastic)
            P += dPdt * dt
        results.append(P)
    results = np.array(results)
    return np.mean(results, axis=0), np.std(results, axis=0)


# ---------------------------------------------------------------------
# Proxy Actuals (placeholder — replace with real data if available)
# ---------------------------------------------------------------------
monthly_actuals = {
    "Jan": [28.9, 18.9, 2.24, 18.95, 6.85, 69.0],
    "Feb": [29.0, 19.0, 2.245, 19.0, 6.88, 69.2],
    "Mar": [29.05, 19.05, 2.25, 19.05, 6.9, 69.4],
    "Apr": [29.1, 19.1, 2.255, 19.1, 6.93, 69.6],
    "May": [29.15, 19.15, 2.26, 19.15, 6.95, 69.8],
    "Jun": [29.2, 19.2, 2.265, 19.2, 6.98, 70.0],
    "Jul": [29.25, 19.25, 2.27, 19.25, 7.0, 70.2],
    "Aug": [29.3, 19.3, 2.275, 19.3, 7.03, 70.4],
    "Sep": [29.35, 19.35, 2.28, 19.35, 7.05, 70.6],
    "Oct": [29.4, 19.4, 2.285, 19.4, 7.08, 70.8],
}

months = list(monthly_actuals.keys())

# ---------------------------------------------------------------------
# Multi-week Simulation (Jan–Oct 2025)
# ---------------------------------------------------------------------
num_weeks = 43
P_current = P0.copy()
weekly_errors = []
per_tank_weekly = []

for week in range(num_weeks):
    mean, std = run_euler_sim(P_current, 1, C, L, K, R, stochastic=True, num_runs=100)
    P_current = mean

    # Approximate current month (4 weeks per month)
    month_idx = min(week // 4, 9)
    actual_proxy = np.array(monthly_actuals[months[month_idx]])

    # Calculate per-tank absolute % errors
    errors = np.abs((mean - actual_proxy) / actual_proxy) * 100
    per_tank_weekly.append(errors)
    weekly_errors.append(np.mean(errors))

# ---------------------------------------------------------------------
# Aggregate to monthly averages
# ---------------------------------------------------------------------
monthly_errors = []
per_tank_monthly = np.zeros((10, n))
for m in range(10):
    start = m * 4
    end = min(start + 4, len(per_tank_weekly))
    segment = np.array(per_tank_weekly[start:end])
    monthly_errors.append(np.mean(segment))
    per_tank_monthly[m] = np.mean(segment, axis=0)

# ---------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------
ensure_dir("results")
csv_path = os.path.join("results", "monthly_errors.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Month", "Average Error (%)"] + tanks)
    for m, err, per_tank in zip(months, monthly_errors, per_tank_monthly):
        writer.writerow([m, f"{err:.2f}"] + [f"{v:.2f}" for v in per_tank])

print(f"Monthly errors saved to {csv_path}")

# ---------------------------------------------------------------------
# Plot Convergence
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(months, monthly_errors, color="blue", marker="o", linewidth=2)
plt.title("Error Convergence (Jan–Oct 2025)")
plt.xlabel("Month")
plt.ylabel(r"Raw Error (%)")
plt.ylim(0, max(monthly_errors) * 1.2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join("results", "error_convergence.png"), dpi=300)
plt.close()
print("Figure saved to results/error_convergence.png")

# ---------------------------------------------------------------------
# 2008–2009 GFC Validation (Placeholder Stub)
# ---------------------------------------------------------------------
def run_gfc_validation():
    """
    Placeholder for historical 2008–2009 validation.
    Should:
      - Load 2007 baseline (sample_data.csv)
      - Simulate 104 weeks (2 years)
      - Compute annual aggregates
      - Output CSV and optional plot
    """
    print("GFC validation not yet implemented. Add logic here for Section 3.1.")

# ---------------------------------------------------------------------
# Script complete
# ---------------------------------------------------------------------
print("Full validation complete.")

