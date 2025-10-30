import numpy as np
from scipy.integrate import odeint
import json
import csv
import matplotlib.pyplot as plt

# Load parameters (with fallback if file missing)
try:
    with open('parameters.json', 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("parameters.json not found - using example defaults")
    params = {
        "tanks": ["US", "China", "Canada", "EU", "LatAm", "Other"],
        "initial_P": [29.15, 19.28, 2.25, 19.45, 7.12, 70.8],
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
        dPdt += np.random.normal(0, leak_sigma * P) + np.random.normal(0, flow_sigma * flows)  # Configurable noise
    return dPdt

# Run simulation (configurable stochastic/ensembles)
def run_sim(P0, t, C, L, K, R, stochastic=False, num_runs=1, leak_sigma=0.005, flow_sigma=0.01):
    if stochastic and num_runs > 1:
        results = [odeint(psm_dynamics, P0, t, args=(C, L, K, R, True, leak_sigma, flow_sigma))[-1] for _ in range(num_runs)]
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        return mean, std
    else:
        return odeint(psm_dynamics, P0, t, args=(C, L, K, R, stochastic, leak_sigma, flow_sigma))[-1], np.zeros_like(P0)

# Example: 1 week sim
t = np.linspace(0, 1/52, 2)
mean, std = run_sim(P0, t, C, L, K, R, stochastic=True, num_runs=100, leak_sigma=0.005, flow_sigma=0.01)
print("Mean:", mean)
print("Std:", std)

# Plot example figure (error convergence placeholder)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
errors = [15.2, 11.8, 9.4, 7.1, 5.8, 4.5, 3.7, 3.0, 2.6, 2.2]
plt.plot(months, errors)
plt.title('Error Convergence')
plt.savefig('error_convergence.png')

# Sensitivity plot placeholder
shifts = ['+10\% K', '-10\% K', '+10\% L', '-10\% L']
variance = [18, -15, 12, -10]
plt.bar(shifts, variance)
plt.title('Sensitivity Analysis')
plt.savefig('sensitivity_analysis.png')

# Load sample data (placeholder for full)
with open('sample_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
print(data)
