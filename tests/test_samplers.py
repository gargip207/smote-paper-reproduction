import numpy as np
from src.samplers import (
    smote,
    random_undersample_majority,
    random_oversample_minority,
    resample_dataset,
)


def test_smote_output_shapes():
    X_min = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    assert smote(X_min, N=50, k=2, random_state=42).shape == (2, 2)
    assert smote(X_min, N=100, k=2, random_state=42).shape == (4, 2)
    assert smote(X_min, N=300, k=2, random_state=42).shape == (12, 2)


def test_smote_within_bounding_box():
    X_min = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 5.0]], dtype=float)
    syn = smote(X_min, N=500, k=2, random_state=7)
    assert np.all(syn >= X_min.min(axis=0))
    assert np.all(syn <= X_min.max(axis=0))


def test_undersample_paper_convention():
    # Paper Section 4.3 example: 50 minority, 200 majority, 200% undersampling -> 25 majority
    X_maj = np.zeros((200, 3))
    X_maj_sub = random_undersample_majority(X_maj, n_minority_orig=50, under_pct=200, random_state=1)
    assert len(X_maj_sub) == 25


def test_resample_dataset_combined():
    X = np.vstack([np.ones((20, 2)), np.zeros((100, 2))])
    y = np.array([1] * 20 + [0] * 100)
    X_res, y_res = resample_dataset(X, y, smote_pct=200, under_pct=100, k=5, random_state=0)
    # 20 original + 40 synthetic = 60 minority; 100% undersampling -> 20 majority
    assert np.sum(y_res == 1) == 60
    assert np.sum(y_res == 0) == 20