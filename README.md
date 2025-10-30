# Pressure System Model (PSM)

**A Modular Stochastic Analogy for Multi-Scale Economic Forecasting**

This repository contains the implementation, data, and LaTeX manuscript for the *Pressure System Model (PSM)*—a physics-inspired framework that models economic systems as networks of interconnected “pressure tanks” governed by compressors (growth drivers), leaks (inflation/corruption drags), flows (trade balances), and regulators (policy interventions).

---

## 🧠 Overview

The PSM is a hybrid of **econophysics** and **macroeconomic simulation**.  
It demonstrates that a simple fluid-dynamic analogy can replicate the behavior of large-scale economic models such as DSGE and CGE with far less calibration overhead, while remaining interpretable and modular.

Paper: [psm_paper/main.tex](psm_paper/main.tex)  
Code: [psm_sim.py](psm_sim.py)

---

## ⚙️ Quick Start

### 1. Clone and install dependencies
\`\`\`bash
git clone https://github.com/jmcentire/psm-model.git
cd psm-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 2. Run the base simulation
\`\`\`bash
python3 psm_sim.py
\`\`\`

**Outputs**
- Ensemble mean & standard deviation of 1-week forecast  
- Two figures generated automatically if missing:
  - error_convergence.png
  - sensitivity_analysis.png
- CSV preview from sample_data.csv

These plots correspond to Figures 1–2 in the paper.

---

## 📊 Reproducing the Paper’s Results

| Section | Dataset | Script | Output |
|----------|----------|--------|---------|
| **3.1 – 2008 GFC Validation** | sample_data.csv | (planned) run_full_validation.py | Table 1 (predictions vs actuals) |
| **3.2 – 2025 Weekly Validation** | proxy data (IMF / World Bank / SIA) | (planned) run_full_validation.py | Table 2 (error convergence) |
| **3.3 – Sensitivity Analysis** | model only | psm_sim.py (auto-generated figure) | Table 3 + /psm_paper/sensitivity_analysis.png |
| **4 – Tariff Forecast** | 2025 baseline params | psm_sim.py | Tables 4–5 summaries |

---

### 🔁 Planned Script

A forthcoming run_full_validation.py will:

1. Run weekly simulations across Jan–Oct 2025 (~44 weeks).  
2. Aggregate absolute error metrics into monthly averages.  
3. Reproduce the GFC and sensitivity tables from the paper.  
4. Save results in /results/*.csv for transparency.

---

## 🧩 Repository Structure
\`\`\`text
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
│   ├─ error_convergence.png
│   └─ sensitivity_analysis.png
└─ results/ (optional)     ← generated validation output
\`\`\`

---

## 📘 Replication Guide for the Paper

### Build the LaTeX Paper
\`\`\`bash
cd psm_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
\`\`\`

This produces main.pdf ready for arXiv upload.

### arXiv Packaging
Zip only the contents of psm_paper/ so that main.tex, references.bib, and the figures are at the archive root:
\`\`\`bash
cd psm_paper
zip -r ../psm_arxiv.zip *
\`\`\`

Upload psm_arxiv.zip to arXiv under license **CC BY-NC-SA 4.0**.

---

## 🧪 Parameter Details

Baseline parameters (parameters.json):

- Tanks: ["US","China","Canada","EU","LatAm","Other"]
- Initial pressures (P): GDP values in trillions USD  
- C – compressors: annualized growth rates  
- L – leaks: inflation + corruption drags  
- K – conductance: trade flow coefficients  
- R – regulator matrix (policy/tariff scaling)  

All values correspond to those used in Section 4 of the paper.

---

## 🪶 Licensing

| Component | License | Notes |
|------------|----------|-------|
| **Code** (psm_sim.py, run_full_validation.py, etc.) | **GNU AGPL v3.0** | Keeps derived or hosted versions open-source. |
| **Paper** (psm_paper/main.tex, main.pdf) | **CC BY-NC-SA 4.0** | Freely share & adapt for research / teaching; no commercial use. |
| **Data** (sample_data.csv, proxies) | Public-domain / aggregated data sources | IMF, World Bank, SIA, EIA, etc. |

If you wish to use the model commercially, please contact the author for dual-licensing.

---

## 🧍‍ Author

**Jeremy McEntire**  
Independent Researcher  |  [j.andrew.mcentire@gmail.com](mailto:j.andrew.mcentire@gmail.com)  
Primary paper: *Pressure Systems: A Modular Stochastic Analogy for Multi-Scale Economic Forecasting (2025)*  
[https://github.com/jmcentire/psm-model](https://github.com/jmcentire/psm-model)

---

## 🧩 Citation

If you use this model or paper in academic work, please cite:

\`\`\`bibtex
@article{mcentire2025psm,
  title={Pressure Systems: A Modular Stochastic Analogy for Multi-Scale Economic Forecasting},
  author={McEntire, Jeremy},
  year={2025},
  note={arXiv preprint arXiv:2501.xxxxx}
}
\`\`\`

---

## 🧭 Roadmap
- [ ] Add run_full_validation.py for complete Table 1–3 reproduction.  
- [ ] Add CLI options for number of weeks, stochastic variance, and seed.  
- [ ] Add proxy data download utilities (IMF API / Trading Economics).  
- [ ] Extend model to agent-based CWT integration.

---

## 💬 Contact & Contributions
Pull requests for extensions (agent-based modeling, ML auto-tuning, visualization dashboards) are welcome under AGPL v3.

---

*Last updated: October 2025*
