import numpy as np
from scipy.integrate import odeint
import json
import csv
import matplotlib.pyplot as plt
import os

# Load parameters (with fallback if file missing)
try:
    with open('parameters.json', 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("parameters.json not found - using example defaults")
    params = {
        "tanks": ["US", "China", "Canada", "EU", "LatAm", "Other"],
        "initial_P": [28.8, 18.5, 2.22, 18.9, 6.73, 68.0],  # Start from end-2024 for 2025 sim
        "C": [0.023, 0.045, 0.008, 0.012, 0.022, 0.03],
        "L": [0.03, 0.016, 0.0338, 0.032, 0.0764, 0.052],
        "K": [[0, 0.03, 0.02, 0.02, 0.015, 0.01],
              [0.03, 0, 0.01, 0.015, 0.01, 0.005],
              [0.02, 0.01, 0, 0.015, 0.01, 0.005],
              [0.02, 0.015, 0.015, 0, 0.01, 0.01],
              [0.015, 0.01, 0.01, 0.01, 0, 0.005],
              [0.01, 0.005, 0.005, 0.01, 0.005, 0]],
        "R_baseline": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    }

tanks = params['tanks']
P0 = np.array(params['initial_P'])
C = np.array(params['C'])
L = np.array(params['L'])
K = np.array(params['K'])
R = np.array(params['R_baseline'])

# PSM dynamics function (corrected for pairwise differences with weekly scaling)
def psm_dynamics(P, t, C, L, K, R, stochastic=False, leak_sigma=0.005, flow_sigma=0.01):
    n = len(P)
    flows = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                flows[i] -= K[i][j] * R[i][j] * (P[i] - P[j]) / 52  # Weekly scale
    dPdt = (C * P / 52) - L * P + flows  # Weekly scale for C
    if stochastic:
        dPdt += np.random.normal(0, leak_sigma * P) + np.random.normal(0, flow_sigma * np.abs(flows))  # Abs to avoid negative scale
    return dPdt

# Run simulation (configurable stochastic/ensembles)
def run_sim(P0, t, C, L, K, R, stochastic=False, num_runs=1, leak_sigma=0.005, flow_sigma=0.01):
    if stochastic and num_runs > 1:
        results = [odeint(psm_dynamics, P0, t, args=(C, L, K, R, True, leak_sigma, flow_sigma), atol=1e-6, rtol=1e-6)[-1] for _ in range(num_runs)]
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        return mean, std
    else:
        return odeint(psm_dynamics, P0, t, args=(C, L, K, R, stochastic, leak_sigma, flow_sigma), atol=1e-6, rtol=1e-6)[-1], np.zeros_like(P0)

# Monthly "actual" proxy series for error calculation (placeholder values from paper trends; replace with real proxies)
monthly_actuals = {
    'Jan': [28.9, 18.9, 2.24, 18.95, 6.85, 69.0],
    'Feb': [29.0, 19.0, 2.245, 19.0, 6.88, 69.2],
    'Mar': [29.05, 19.05, 2.25, 19.05, 6.9, 69.4],
    'Apr': [29.1, 19.1, 2.255, 19.1, 6.93, 69.6],
    'May': [29.15, 19.15, 2.26, 19.15, 6.95, 69.8],
    'Jun': [29.2, 19.2, 2.265, 19.2, 6.98, 70.0],
    'Jul': [29.25, 19.25, 2.27, 19.25, 7.0, 70.2],
    'Aug': [29.3, 19.3, 2.275, 19.3, 7.03, 70.4],
    'Sep': [29.35, 19.35, 2.28, 19.35, 7.05, 70.6],
    'Oct': [29.4, 19.4, 2.285, 19.4, 7.08, 70.8]
}

months = list(monthly_actuals.keys())
weekly_errors = []  # List to store weekly average errors

# Simulate 43 weeks (Jan-Oct 2025, approx 4.3 weeks/month)
num_weeks = 43
P_current = P0
for week in range(num_weeks):
    t = np.linspace(0, 1/52, 2)
    mean, std = run_sim(P_current, t, C, L, K, R, stochastic=True, num_runs=100)
    P_current = mean  # Update for next week

    # Compute weekly error (avg % across tanks vs interpolated monthly proxy)
    month_idx = min(week // 4, 9)  # Approximate month (4 weeks/month)
    actual_proxy = np.array(monthly_actuals[months[month_idx]])
    error = np.abs((mean - actual_proxy) / actual_proxy) * 100
    avg_error = np.mean(error)
    weekly_errors.append(avg_error)

# Aggregate to monthly averages
monthly_errors = []
for m in range(10):
    start = m * 4
    end = min(start + 4, len(weekly_errors))
    monthly_errors.append(np.mean(weekly_errors[start:end]))

# Save to CSV
with open('monthly_errors.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Month', 'Average Raw Error (%)'])
    for m, err in zip(months, monthly_errors):
        writer.writerow([m, err])

# Generate plot
plt.plot(months, monthly_errors, color='blue', linewidth=2)
plt.title('Error Convergence')
plt.xlabel('Month')
plt.ylabel(r'Raw Error (\%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_convergence.png', dpi=300)
plt.close()

print("Monthly errors saved to monthly_errors.csv")
print("Figure saved to error_convergence.png")
</parameter
</xai:function_call
