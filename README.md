# Advanced LBO Monte Carlo Analysis

A modular, extensible Leveraged Buyout (LBO) Monte Carlo simulation framework built in Python.

The project combines a core stochastic LBO engine with an advanced analytics layer for scenario analysis, stress testing, sensitivity diagnostics, and what-if exploration. This repository is designed to mirror how sophisticated investment teams evaluate risk, return distributions, and downside exposure under uncertainty.

## 📌 Project Overview

Traditional LBO models rely on single-point assumptions. This project instead:

- Simulates thousands of LBO outcomes
- Models correlated operating and valuation drivers
- Quantifies return distributions (IRR, MOIC)
- Explicitly measures downside risk and tail outcomes
- Supports scenario and stress testing

The architecture separates:

- Core financial logic (reusable, testable)
- Advanced analytics & visualization (optional overlay)

## 🧱 Repository Structure
```
.
├── lbo_monte_carlo.py          # Core Monte Carlo LBO engine
├── advanced_lbo_analysis.py    # Advanced analytics layer (inherits core)
├── README.md                   # Project documentation
└── outputs/
    ├── lbo_monte_carlo_results.png
    ├── scenario_comparison.png
    ├── tornado_chart.png
    └── what_if_*.png
```

## ⚙️ Core Engine: `lbo_monte_carlo.py`

The base engine is responsible for:

- Defining base deal assumptions
- Generating correlated random inputs
- Building yearly operating projections
- Modeling multi-tranche debt structures
- Calculating:
  - Equity cash flows
  - IRR
  - MOIC
  - Default outcomes
- Running large-scale Monte Carlo simulations
- Producing summary statistics and visual dashboards

### Key Outputs

- IRR distribution
- MOIC distribution
- Probability of hitting hurdle rates
- Probability of default
- Operating and capital structure diagnostics

This file can be run independently.

## 🔍 Advanced Analytics Layer: `advanced_lbo_analysis.py`

The advanced module extends the base engine via inheritance:
```python
class AdvancedLBOAnalysis(LBOMonteCarlo):
```

No core financial logic is duplicated.

### Features Added

#### 1. Scenario Analysis

Evaluate bull, base, and bear cases by shifting key assumptions:

- Revenue growth
- EBITDA margin
- Exit multiple

Produces:

- Scenario-specific IRR/MOIC distributions
- Comparative visualizations

#### 2. Stress Testing

Tests extreme but plausible downside environments:

- Recession
- Margin compression
- Multiple contraction
- High interest rate environments

Measures:

- Probability of loss
- Probability of IRR below hurdle
- Tail risk exposure

#### 3. Sensitivity (Tornado) Analysis

Quantifies which assumptions drive returns most:

- Exit multiple
- Revenue growth
- EBITDA margin

Outputs a tornado chart ranked by IRR impact.

#### 4. What-If Analysis

Explores the effect of changing a single assumption across a range of values:

- IRR vs assumption
- MOIC vs assumption
- Interquartile uncertainty bands

## 📊 Example Outputs

The framework generates publication-ready visuals, including:

- Monte Carlo IRR histograms
- Scenario comparison charts
- Stress test summaries
- Tornado sensitivity charts
- What-if response curves

All outputs are automatically saved as `.png` files.

## 🚀 How to Run

### Requirements
```bash
pip install numpy pandas matplotlib seaborn
```

### Run Core Simulation Only
```bash
python lbo_monte_carlo.py
```

### Run Full Advanced Analysis
```bash
python advanced_lbo_analysis.py
```

## 🧠 Design Philosophy

**Separation of concerns**  
Core modeling and analytics are decoupled.

**Extensibility**  
New scenarios, variables, or structures can be added without touching the engine.

**Real-world realism**  
Correlated drivers, capital structure detail, and downside metrics reflect institutional workflows.

## ⚠️ Disclaimer

This model is for educational and analytical purposes only. It does not constitute investment advice or a recommendation to buy or sell any security.
