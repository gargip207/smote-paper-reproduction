# SMOTE: Synthetic Minority Over-sampling Technique — Paper Reproduction

A from-scratch Python reproduction of **Chawla, Bowyer, Hall, and Kegelmeyer (2002)**, *"SMOTE: Synthetic Minority Over-sampling Technique"* (*Journal of Artificial Intelligence Research*, Vol. 16, pp. 321–357).

## Overview
Standard classifiers trained on imbalanced datasets maximize overall accuracy by favoring the majority class, resulting in poor recall on rare but critical events. Simply duplicating minority instances causes decision trees to overfit narrow, hyper-specific regions in feature space.

This repository implements the core algorithms and evaluation geometry from the paper without black-box resampling wrappers:
- **SMOTE ($k$-NN Feature-Space Interpolation):** Generates synthetic minority samples along hyper-rectangles between each minority instance and its $k=5$ nearest minority neighbors (`src/samplers.py`).
- **Combined Over/Under-Sampling:** Pairs SMOTE oversampling ($50\%$–$500\%$) with random majority undersampling following Chawla et al.'s minority-to-majority percentage convention (`src/samplers.py`).
- **ROC Geometry & Convex Hull:** Computes Trapezoidal AUC with $(100\%, 100\%)$ endpoint extrapolation and extracts the upper-left ROC Convex Hull via Andrew's monotone chain variant of Graham's Scan (`src/metrics.py`).
- **Leak-Free 10-Fold Cross-Validation:** Performs all resampling strictly inside training folds across C4.5-style entropy decision trees and prior-adjusted Naive Bayes baselines (`src/models.py`, `src/experiment.py`).

## Repository Structure
```text
smote-paper-reproduction/
├── data/                                      # Cached OpenML benchmark datasets
├── notebooks/
│   ├── 01_decision_regions_and_trees.ipynb    # Reproduces Figs. 3–5 (tree node counts & minority recall)
│   └── 02_roc_and_auc_benchmarks.ipynb        # Reproduces Figs. 7, 9, 11, 17 (ROC Convex Hulls) & Table 3
├── src/
│   ├── datasets.py                            # OpenML loaders & binarization (Pima, Phoneme, Satimage, Mammography)
│   ├── experiment.py                          # Leak-free 10-fold stratified CV grid runner
│   ├── metrics.py                             # %FP/%TP rates, Trapezoidal AUC, and Graham's Scan Convex Hull
│   ├── models.py                              # Pruned entropy Decision Tree (C4.5 proxy) & prior-tuned Naive Bayes
│   └── samplers.py                            # Pure NumPy SMOTE, random undersampling, and minority replication
├── tests/                                     # Pytest unit tests verifying sampling invariants, AUC, and CV pipeline
├── pyproject.toml
└── requirements.txt