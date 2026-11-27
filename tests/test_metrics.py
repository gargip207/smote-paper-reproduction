import numpy as np
from src.metrics import compute_rates, trapezoidal_auc, roc_convex_hull


def test_compute_rates():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    fp, tp = compute_rates(y_true, y_pred)
    assert np.isclose(tp, 50.0)
    assert np.isclose(fp, 25.0)


def test_trapezoidal_auc_extrapolation():
    # Curve from (0, 0) -> (0, 100); automatically extrapolates to (100, 100) -> AUC = 1.0
    auc = trapezoidal_auc([0.0, 0.0], [0.0, 100.0])
    assert np.isclose(auc, 1.0)


def test_roc_convex_hull_filters_dominated_points():
    # (50, 40) lies below the diagonal (0,0)-(100,100) and must be excluded;
    # (20, 80) and (40, 90) form strict upper-left convex vertices.
    pts = [(20.0, 80.0), (50.0, 40.0), (40.0, 90.0)]
    hull = roc_convex_hull(pts)
    hull_tuples = [tuple(np.round(row, 2)) for row in hull]
    assert (0.0, 0.0) in hull_tuples
    assert (20.0, 80.0) in hull_tuples
    assert (40.0, 90.0) in hull_tuples
    assert (100.0, 100.0) in hull_tuples
    assert (50.0, 40.0) not in hull_tuples