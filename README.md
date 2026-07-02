# ISLP Interactive Learning & Visualization Tool 🏎️

> **Envisioned by Aadi Singh** — A learning and visualization tool for *An Introduction to Statistical Learning with Applications in Python/R* (ISLP), built for the Applied AI/ML course under the Business Data Analytics (BDA) program. Find it on "".

---

## Overview

This Streamlit web application transforms the dense mathematical theory of ISLP Chapters 1–3 into an **interactive, hands-on experience**. Every equation is derived step-by-step — from school-level algebra to graduate-level proofs — and paired with real-time visualizations powered by custom business datasets.

### Key Features

- **Complete Mathematical Derivations** — From $y = mx + c$ to the Bias-Variance decomposition, every equation is broken down into clear, sequential steps.
- **6 Custom Business Datasets** — Real Estate, Customer LTV, Factory Output, Employee Productivity, Loan Approval, and Customer Churn (100 rows each).
- **Interactive Model Simulators** — Adjustable polynomial degrees, KNN neighbor counts, cluster sliders, and side-by-side parametric vs non-parametric comparisons.
- **10 Real-World Case Studies** — Target's Pregnancy Predictor, Netflix Prize, Amazon Dynamic Pricing, AlphaGo, the 2010 Flash Crash, and more.
- **150 Management-Style Assessment Questions** — 50 questions per chapter with instant grading and per-question explanations on submit.
- **Clean Minimalist UI** — F1 / Dust II–inspired aesthetic with Carbon Black backgrounds, Sand/Terracotta accents, and Racing Red highlights.

---

## Modules

| Module | Topic | Highlights |
|--------|-------|------------|
| **1 — Data Foundations** | Introduction to Data & Statistical Learning | Dataset exploration, Netflix $1M Prize Case Study, 50-question quiz |
| **2 — Statistical Learning Engine** | Core Theory (Ch. 2) | Master Equation, Prediction vs Inference, Parametric vs Non-Parametric, Bias-Variance Tradeoff (full 7-step derivation), Classification (KNN + Bayes), 10 Case Studies, 50-question quiz |
| **3 — Linear Regression** | The Parametric Workhorse (Ch. 3) | OLS derivation (RSS → calculus → matrix form), interactive β-tuning minigame, multivariate inference, Zillow Case Study, 50-question quiz |

---

## Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/)
- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, scikit-learn
- **Styling:** Custom CSS (Inter + JetBrains Mono fonts, dark-mode radial gradients)

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation
For local Install(why??)
```bash
# Clone the repository
git clone https://github.com/aadis12/ISLP-Interactive-WebApp.git
cd ISLP-Interactive-WebApp

# Install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn

# Generate the custom datasets
python data_generator.py

# Run the application
python -m streamlit run app.py
```

The app will open at `http://localhost:xxxx`(depends!).

---

## Project Structure

```
islp_webapp/
├── app.py                              # Main entry point & global UI/CSS
├── data_generator.py                   # Generates 6 custom business datasets
├── quiz_engine.py                      # Shared quiz rendering utility
├── pages/
│   ├── 1_Chapter_1_Intro.py            # Module 1: Data Foundations
│   ├── 2_Chapter_2_Statistical_Learning.py  # Module 2: Core Theory
│   └── 3_Chapter_3_Linear_Regression.py     # Module 3: Linear Regression
└── README.md
```

---

## Screenshots

After launching the app, navigate through the sidebar to explore:
- **Interactive U-Curve Simulator** (Bias-Variance tab with 3 adjustable sliders)
- **KNN Decision Boundary Visualizer** (toggle between Churn and Loan datasets)
- **Be the Algorithm Minigame** (manually tune β₀ and β₁ vs the optimal fit)

---

## License

This project is for educational purposes as part of the Applied AI/ML for BDA coursework.

---

<p align="center"><strong>Envisioned by Aadi</strong></p>
