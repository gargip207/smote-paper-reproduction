import numpy as np
from src.experiment import (
    evaluate_resampling_curve,
    evaluate_naive_bayes_curve,
    compute_table3_auc_row,
)


def test_experiment_pipeline_end_to_end():
    rng = np.random.default_rng(42)
    X_maj = rng.normal(0.0, 1.0, size=(120, 4))
    X_min = rng.normal(1.2, 1.0, size=(30, 4))
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * 120 + [1] * 30)

    res_df = evaluate_resampling_curve(X, y, smote_pct=100, under_grid=[50, 100], n_splits=3)
    assert len(res_df) == 3  # raw + 2 undersampling levels
    assert {"fp", "tp", "tree_nodes"}.issubset(res_df.columns)

    nb_df = evaluate_naive_bayes_curve(X, y, multipliers=[1, 5], n_splits=3)
    assert len(nb_df) == 2

    auc_row = compute_table3_auc_row(X, y, smote_levels=[100], random_state=42)
    assert 0 <= auc_row["Under"] <= 10000
    assert 0 <= auc_row["100 SMOTE"] <= 10000