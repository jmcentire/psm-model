# PSM Model Repository

## Overview
This repository contains the code, data, and parameters for the Pressure System Model (PSM) as described in the arXiv paper "Pressure Systems: A Modular Stochastic Analogy for Multi-Scale Economic Forecasting".

## Setup
1. Clone the repo: `git clone https://github.com/jmcentire/psm-model.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python psm_sim.py`

## Parameters
- parameters.json: Initial values for C, L, K, R. Edit to customize tanks. Guidance: 
  - initial_P: From IMF/World Bank GDP data.
  - C: Annual growth rates from IMF WEO (scale weekly as C / 52).
  - L: Inflation from CPI + corruption drag = (100 - CPI_score)/100 * 0.02; CPI from Transparency International.
  - K: Trade imbalances as \% GDP from WTO (e.g., US-China deficit / US GDP).
  - R: Start at 1.0; adjust for tariffs (e.g., 0.75 for 25\% reduction).
  - For sectors: Add sub-tanks with connected K (e.g., microchips initial_P from SIA sales).

## Stochastic and Ensemble Options
- Set `stochastic=True` to enable noise (configurable: leak_sigma=0.005, flow_sigma=0.01).
- Set `num_runs=100` for ensembles (default 1 for deterministic).
- Run without: `stochastic=False, num_runs=1`.

## Data
- sample_data.csv: Historical GDP proxies. For full replication, fetch latest from:
  - GDP: IMF WEO[](https://www.imf.org/en/Publications/WEO/weo-database/2025/October)
  - Microchips: SIA reports[](https://www.semiconductors.org/resources/data/)
  - Rare Earths: Trading Economics[](https://tradingeconomics.com/commodity/neodymium)
  - Energy: EIA STEO[](https://www.eia.gov/outlooks/steo/)
  - Interpolate monthly to weekly linearly + noise in code.

## Running Simulations
- For 1 week: Use t = np.linspace(0, 1/52, 2)
- Stochastic: Set `stochastic=True`, adjust sigmas, `num_runs>1` for CI.
- Disable: `stochastic=False` for deterministic runs.
- Example output: Mean and Std for pressures.

## Replication
To replicate 2008 GFC:
1. Load 2007 from sample_data.csv as P0.
2. Run with baseline params.
3. For diagnostics, apply adjustments manually (e.g., C[1] += 0.05 for China).

For 2025: Use weekly loop with proxies (interpolate monthly data linearly in code).

Contact: j.andrew.mcentire@gmail.com
