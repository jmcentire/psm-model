import numpy as np
from scipy.integrate import odeint
import json
import csv
import matplotlib.pyplot as plt

# Load parameters
with open('parameters.json', 'r') as f:
    params = json.load(f)

tanks = params['tanks']
P0 = np.array(params['initial_P'])
C = np.array(params['C'])
L = np.array(params['L'])
K = np.array(params['K'])
R = np.array(params['R_baseline'])

# PSM dynamics function (corrected for pairwise differences)
def psm_dynamics(P, t, C, L, K, R, stochastic=False):
    n = len(P)
    flows = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                flows[i] -= K[i][j] * R[i][j] * (P[i] - P[j])
    dPdt = C * P - L * P + flows
    if stochastic:
        dPdt += np.random.normal(0, 0.005 * P)  # Leak noise
    return dPdt

# Run simulation
def run_sim(P0, t, C, L, K, R, stochastic=False, num_runs=1):
    if stochastic and num_runs > 1:
        results = [odeint(psm_dynamics, P0, t, args=(C, L, K, R, True))[-1] for _ in range(num_runs)]
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        return mean, std
    else:
        return odeint(psm_dynamics, P0, t, args=(C, L, K, R, stochastic))[-1], np.zeros_like(P0)

# Example: 1 week sim
t = np.linspace(0, 1/52, 2)
mean, std = run_sim(P0, t, C, L, K, R, stochastic=True, num_runs=100)
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
