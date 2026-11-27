import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.samplers import resample_dataset, random_oversample_minority
from src.models import (
    build_c45_proxy,
    predict_with_threshold,
    build_naive_bayes_with_prior_ratio,
)
from src.metrics import compute_rates, trapezoidal_auc


DEFAULT_UNDER_GRID = [
    10, 15, 25, 50, 75, 100, 125, 150, 175,
    200, 300, 400, 500, 600, 700, 800, 1000, 2000,
]

DEFAULT_PRIOR_MULTIPLIERS = [
    1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50,
]


def evaluate_resampling_curve(
    X: np.ndarray,
    y: np.ndarray,
    smote_pct: int | None = None,
    under_grid: list[int] = DEFAULT_UNDER_GRID,
    k: int = 5,
    ccp_alpha: float = 0.002,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sweep majority undersampling levels (paired with optional SMOTE oversampling)
    across 10-fold stratified CV. Includes the raw unresampled dataset as the
    lower-left anchor point (Section 5.2).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    # Include raw dataset (under_pct=None) first, followed by the undersampling sweep
    sweep_levels: list[int | None] = [None] + list(under_grid)
    records = []

    for under_pct in sweep_levels:
        fold_fps, fold_tps, fold_nodes = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_res, y_res = resample_dataset(
                X_train,
                y_train,
                smote_pct=smote_pct,
                under_pct=under_pct,
                k=k,
                random_state=random_state + fold_idx,
            )

            clf = build_c45_proxy(ccp_alpha=ccp_alpha, random_state=random_state + fold_idx)
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)

            fp, tp = compute_rates(y_test, y_pred)
            fold_fps.append(fp)
            fold_tps.append(tp)
            fold_nodes.append(clf.tree_.node_count)

        records.append({
            "smote_pct": smote_pct or 0,
            "under_pct": under_pct if under_pct is not None else 0,
            "fp": float(np.mean(fold_fps)),
            "tp": float(np.mean(fold_tps)),
            "tree_nodes": float(np.mean(fold_nodes)),
        })

    df = pd.DataFrame(records)
    return df


def evaluate_naive_bayes_curve(
    X: np.ndarray,
    y: np.ndarray,
    multipliers: list[float] = DEFAULT_PRIOR_MULTIPLIERS,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sweep Naive Bayes minority class prior multipliers from 1x to 50x (Section 5.2).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    records = []

    for mult in multipliers:
        fold_fps, fold_tps = [], []
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            n_min = int(np.sum(y_train == 1))
            n_maj = int(np.sum(y_train == 0))

            clf = build_naive_bayes_with_prior_ratio(n_maj, n_min, prior_multiplier=mult)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            fp, tp = compute_rates(y_test, y_pred)
            fold_fps.append(fp)
            fold_tps.append(tp)

        records.append({
            "prior_multiplier": mult,
            "fp": float(np.mean(fold_fps)),
            "tp": float(np.mean(fold_tps)),
        })

    return pd.DataFrame(records)


def evaluate_threshold_curve(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: list[float] | None = None,
    ccp_alpha: float = 0.002,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Vary C4.5 leaf decision thresholds from 0.5 down to 0.0 on raw data (Section 5.4).
    """
    if thresholds is None:
        thresholds = [0.5, 0.45, 0.42, 0.4, 0.35, 0.32, 0.3, 0.27, 0.25, 0.22, 0.2, 0.17, 0.15, 0.12, 0.1, 0.05, 0.0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    records = []

    for thresh in thresholds:
        fold_fps, fold_tps = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            clf = build_c45_proxy(ccp_alpha=ccp_alpha, random_state=random_state + fold_idx)
            clf.fit(X_train, y_train)
            y_pred = predict_with_threshold(clf, X_test, threshold=thresh)

            fp, tp = compute_rates(y_test, y_pred)
            fold_fps.append(fp)
            fold_tps.append(tp)

        records.append({
            "threshold": thresh,
            "fp": float(np.mean(fold_fps)),
            "tp": float(np.mean(fold_tps)),
        })

    return pd.DataFrame(records)


def compute_table3_auc_row(
    X: np.ndarray,
    y: np.ndarray,
    smote_levels: list[int] = [50, 100, 200, 300, 400, 500],
    ccp_alpha: float = 0.002,
    random_state: int = 42,
) -> dict[str, int]:
    """
    Compute the 4-digit scaled AUCs (AUC * 10000) matching Table 3 of Chawla et al. (2002).
    """
    row: dict[str, int] = {}

    under_df = evaluate_resampling_curve(X, y, smote_pct=None, ccp_alpha=ccp_alpha, random_state=random_state)
    row["Under"] = int(round(trapezoidal_auc(under_df["fp"], under_df["tp"]) * 10000))

    for s_pct in smote_levels:
        smote_df = evaluate_resampling_curve(X, y, smote_pct=s_pct, ccp_alpha=ccp_alpha, random_state=random_state)
        row[f"{s_pct} SMOTE"] = int(round(trapezoidal_auc(smote_df["fp"], smote_df["tp"]) * 10000))

    return row