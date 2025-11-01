# Pressure System Model (PSM)

**Fast Economic Diagnostics Through Physics-Inspired Modeling**

This repository contains the implementation, data, and LaTeX manuscript for the *Pressure System Model (PSM)*—a rapid diagnostic tool that models economic systems as networks of interconnected "pressure tanks" governed by compressors (growth drivers), leaks (inflation/corruption drags), flows (trade balances), and regulators (policy interventions).

---

## 🧠 What is PSM?

PSM is a **diagnostic screening tool** for economic policy analysis, not a replacement for established models like DSGE or CGE. Think of it as an economic "X-ray"—fast, transparent, and useful for initial diagnosis before deploying more detailed modeling.

**Key Value Proposition:**
- ⚡ **Speed**: Generate forecasts in seconds instead of hours
- 🔍 **Diagnostics**: Prediction errors reveal hidden forces (stimulus, shocks, behavioral shifts)
- 🎯 **Hypothesis Generation**: Test 100 scenarios quickly to identify high-priority cases for deeper analysis
- 📚 **Transparency**: Intuitive pressure/flow metaphor makes results interpretable for non-experts

**What PSM is NOT:**
- ❌ Not a micro-founded welfare analysis tool (use DSGE for that)
- ❌ Not a precision forecaster (<5% errors; PSM targets 5-15% for screening)
- ❌ Not suitable for high-frequency financial forecasting
- ❌ Not a replacement for detailed sectoral CGE analysis

Paper: [psm_paper/main.tex](psm_paper/main.tex)  
Code: [psm_sim.py](psm_sim.py)

---

## 🎯 When to Use PSM

### ✅ **Use PSM For:**

1. **Rapid Policy Screening**
   - Testing dozens of tariff scenarios in minutes
   - Initial impact assessments before detailed CGE modeling
   - Exploring parameter sensitivity across wide ranges

2. **Anomaly Detection**
   - Identifying when models systematically miss (errors signal hidden forces)
   - Discovering unannounced interventions via prediction deltas
   - Flagging supply shocks or behavioral regime changes

3. **Hypothesis Generation**
   - "What-if" exploration to prioritize deeper analysis
   - Generating testable predictions for empirical validation
   - Brainstorming policy alternatives quickly

4. **Teaching & Communication**
   - Pedagogical tool for teaching macro dynamics
   - Stakeholder presentations (intuitive pressure/flow metaphor)
   - Demonstrating trade-offs transparently

### ❌ **Do NOT Use PSM For:**

1. **Precision Forecasting** - Central banks needing <5% errors should use DSGE
2. **Welfare Analysis** - Requires micro-foundations PSM lacks
3. **Agent-Based Scenarios** - Labor markets, consumer behavior need specialized tools
4. **High-Frequency Finance** - Daily/hourly forecasting where agents dominate
5. **Definitive Policy Evaluation** - PSM screens; DSGE/CGE validate

---

## 🔄 Recommended Workflow: PSM → DSGE/CGE Pipeline

PSM works best as part of a **complementary modeling workflow**:

```
Step 1: RAPID SCREENING (PSM - minutes)
├─ Run 50-100 policy scenarios (tariff rates, subsidy levels, etc.)
├─ Identify top 3-5 scenarios by impact magnitude
└─ Flag anomalies where errors are large (signals missing mechanisms)

Step 2: HYPOTHESIS REFINEMENT (PSM diagnostics - minutes)
├─ Analyze prediction deltas to infer hidden forces
├─ Update parameters based on diagnostic insights
└─ Generate testable predictions for validation

Step 3: DETAILED ANALYSIS (DSGE/CGE - hours)
├─ Model the 3-5 high-priority scenarios in depth
├─ Calibrate sectoral linkages PSM errors flagged as important
└─ Perform welfare analysis and counterfactuals

Step 4: SENSITIVITY CHECKS (PSM - minutes)
├─ Return to PSM for quick robustness tests
├─ Verify DSGE/CGE results hold across parameter ranges
└─ Iterate if major discrepancies emerge
```

**Example Use Case:**
> A policy analyst needs to evaluate 100 potential tariff structures on semiconductors. 
> - **With PSM**: Screen all 100 in 30 minutes → identify 5 high-impact cases → model those 5 in CGE over 2 days
> - **Without PSM**: Model all 100 in CGE over 50 days → miss deadline

---

## ⚙️ Quick Start

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/jmcentire/psm-model.git
cd psm-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Base Simulation
```bash
python3 psm_sim.py
```

**Outputs:**
- Ensemble mean & standard deviation of 1-week forecast  
- Diagnostic plots (if missing, auto-generated):
  - `psm_paper/error_convergence.png` - shows diagnostic learning over time
  - `psm_paper/sensitivity_analysis.png` - parameter robustness tests
- CSV preview from `sample_data.csv`

These plots correspond to Figures 1–2 in the paper.

---

## 📊 Reproducing the Paper's Results

| Section | Dataset | Script | Output |
|----------|----------|--------|---------|
| **3.1 – 2008 GFC Validation** | sample_data.csv | (planned) run_full_validation.py | Table 1 (predictions vs actuals) |
| **3.2 – 2025 Weekly Validation** | proxy data (IMF / World Bank / SIA) | (planned) run_full_validation.py | Table 2 (error convergence) |
| **3.3 – Sensitivity Analysis** | model only | psm_sim.py (auto-generated figure) | Table 3 + sensitivity_analysis.png |
| **4 – Tariff Forecast** | 2025 baseline params | psm_sim.py | Tables 4–5 summaries |

### 🔁 Planned Script

A forthcoming `run_full_validation.py` will:

1. Run weekly simulations across Jan–Oct 2025 (~44 weeks)
2. Aggregate absolute error metrics into monthly averages
3. Reproduce the GFC and sensitivity tables from the paper
4. Save results in `/results/*.csv` for transparency

---

## 🧩 Repository Structure

```text
psm-model/
├─ LICENSE                 ← AGPL-3.0 (code)
├─ README.md               ← this file
├─ requirements.txt        ← NumPy, SciPy, Matplotlib
├─ parameters.json         ← baseline model parameters
├─ sample_data.csv         ← historical validation data
├─ psm_sim.py              ← main simulation script (Euler method)
├─ psm_paper/
│   ├─ main.tex            ← LaTeX source for the paper
│   ├─ references.bib      ← BibTeX file
│   ├─ error_convergence.png    ← Figure 1 (diagnostic learning)
│   └─ sensitivity_analysis.png ← Figure 2 (robustness tests)
└─ results/ (optional)     ← generated validation output
```

---

## 📘 Replication Guide for the Paper

### Build the LaTeX Paper
```bash
cd psm_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This produces `main.pdf` ready for arXiv upload.

### arXiv Packaging
Zip only the contents of `psm_paper/` so that `main.tex`, `references.bib`, and the figures are at the archive root:
```bash
cd psm_paper
zip -r ../psm_arxiv.zip *
```

Upload `psm_arxiv.zip` to arXiv under license **CC BY-NC-SA 4.0**.

---

## 🧪 Parameter Details

Baseline parameters (`parameters.json`):

- **Tanks**: `["US", "China", "Canada", "EU", "LatAm", "Other"]`
- **Initial Pressures (P)**: GDP values in trillions USD  
- **C (Compressors)**: Annualized growth rates (e.g., US 2.3%, China 4.5%)
- **L (Leaks)**: Inflation + corruption drags (e.g., US 3.0%, China 1.6%)
- **K (Conductance)**: Trade flow coefficients (6×6 matrix)
- **R (Regulators)**: Policy/tariff scaling (identity matrix = no intervention)

All values correspond to those used in Section 4 of the paper.

**Calibration Notes:**
- Parameters derived from historical averages (IMF, World Bank, WTO)
- Sensitivity: ±10% leak changes → ±3% output variance
- See paper Section 3.3 for full robustness analysis

---

## 🔬 Understanding Diagnostic Adjustments

PSM's key innovation is using **prediction errors as diagnostic signals**:

1. **Initial Forecast**: Run model with baseline parameters
2. **Measure Deltas**: Compare to proxy data (industrial production, sector reports)
3. **Interpret Errors**: Large misses suggest unmodeled forces:
   - Stimulus programs (compressor adjustments)
   - Supply shocks (conductance shifts)
   - Policy changes (regulator updates)
4. **Adjust & Rerun**: Update parameters based on economic interpretation
5. **Validate**: Check if adjusted model explains past anomalies

**Example from Paper:**
- 2008 GFC: PSM under-predicted China GDP by 60% in 2009
- **Diagnosis**: Large error flagged missing \$586B stimulus package
- **Adjustment**: Increased China compressor by +5%
- **Result**: Error dropped to 5%, confirmed stimulus as key driver

This process generates hypotheses for deeper DSGE/CGE investigation.

---

## ⚠️ Important Limitations

### Data Quality
- **2025 weekly validation uses interpolated proxies** (industrial production → GDP)
- Proxy measurement error ~3-5% before model forecasting
- True validation requires actual 2025 data (available mid-2026)
- We commit to rerunning and reporting any systematic biases

### Model Scope
- **No micro-foundations**: Unsuitable for welfare analysis or behavioral counterfactuals
- **Proportional dynamics**: Breaks down in hyperinflation or sudden stops
- **Flow-dominated**: Underperforms in agent-heavy scenarios (labor markets)
- **Parameter sensitivity**: Moderate (5-7% output variance from 10-20% input changes)

### Validation Caveats
- Small sample sizes (2 validation periods: 2008-2009, 2025)
- Hindsight adjustments (GFC benefited from known outcomes)
- November 2025 tariff forecasts are first true out-of-sample test

**Users should:**
- ✅ Treat PSM as hypothesis generator, not definitive forecaster
- ✅ Verify results with domain expertise
- ✅ Use complementary modeling (PSM screens → DSGE validates)
- ❌ Avoid "tuning until it fits" without economic interpretation

---

## 🪶 Licensing

| Component | License | Notes |
|------------|----------|-------|
| **Code** (psm_sim.py, etc.) | **GNU AGPL v3.0** | Keeps derived/hosted versions open-source |
| **Paper** (psm_paper/main.tex, main.pdf) | **CC BY-NC-SA 4.0** | Free for research/teaching; no commercial use |
| **Data** (sample_data.csv, proxies) | Public domain / aggregated sources | IMF, World Bank, SIA, EIA, etc. |

If you wish to use the model commercially, please contact the author for dual-licensing.

---

## 🧍 Author

**Jeremy McEntire**  
Independent Researcher  |  [j.andrew.mcentire@gmail.com](mailto:j.andrew.mcentire@gmail.com)  
Primary Paper: *Pressure Systems: A Modular Stochastic Analogy for Multi-Scale Economic Forecasting (2025)*  
[https://github.com/jmcentire/psm-model](https://github.com/jmcentire/psm-model)

---

## 🧩 Citation

If you use this model or paper in academic work, please cite:

```bibtex
@article{mcentire2025psm,
  title={Pressure Systems: A Modular Stochastic Analogy for Multi-Scale Economic Forecasting},
  author={McEntire, Jeremy},
  year={2025},
  note={arXiv preprint arXiv:2501.xxxxx},
  url={https://github.com/jmcentire/psm-model}
}
```

---

## 🧭 Roadmap

**Near-Term (1 Month):**
- [ ] Add `run_full_validation.py` for complete Table 1–3 reproduction
- [ ] CLI options for weeks, stochastic variance, and seed
- [ ] Real-validation follow-up when Q4 2025 data available (mid-2026)
- [ ] "When to Use PSM" examples with real case studies

**Medium-Term (3-6 Months):**
- [ ] Proxy data download utilities (IMF API / Trading Economics)
- [ ] Interactive dashboard for scenario exploration
- [ ] Machine learning auto-tuning for diagnostic adjustments

**Long-Term (1+ Years):**
- [ ] Agent-based CWT integration (family cohesion as inverse leaks)
- [ ] Hybrid PSM-DSGE framework
- [ ] Pedagogical materials for teaching macro with PSM

---

## 💬 Contact & Contributions

Pull requests for extensions are welcome under AGPL v3:
- Agent-based modeling integrations
- ML auto-tuning for parameter adjustments
- Visualization dashboards
- Additional validation datasets
- Pedagogical examples

**Particularly Interested In:**
- Real-world case studies (policy offices using PSM for screening)
- Failure analysis (where PSM systematically misses)
- Hybrid PSM-DSGE workflows
- Teaching materials (lecture slides, problem sets)

---

## 📚 Additional Resources

**Related Papers:**
- Phillips (1950) - Original MONIAC hydraulic computer
- Smets & Wouters (2007) - DSGE benchmark for comparison
- Dixon et al. (2013) - CGE validation challenges

**Complementary Tools:**
- [Dynare](https://www.dynare.org/) - DSGE modeling platform (use after PSM screening)
- [GTAP](https://www.gtap.agecon.purdue.edu/) - CGE for detailed trade analysis
- [FRED](https://fred.stlouisfed.org/) - Data sources for calibration

**Suggested Reading Order:**
1. Paper Section 1-2 (Introduction & Model) - 15 min
2. Run `psm_sim.py` - explore outputs - 30 min
3. Paper Section 3 (Validation) - understand diagnostic approach - 20 min
4. Paper Section 5 (Comparison & Limitations) - scope understanding - 15 min
5. Try modifying `parameters.json` - test sensitivity - 30 min

---

*Last updated: November 2025*
