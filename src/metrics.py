import numpy as np


def compute_rates(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> tuple[float, float]:
    """
    Compute (%FP, %TP) in [0, 100] from binary predictions (Section 2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pos_mask = y_true == positive_label
    neg_mask = ~pos_mask

    tp = np.sum((y_pred == positive_label) & pos_mask)
    fn = np.sum((y_pred != positive_label) & pos_mask)
    fp = np.sum((y_pred == positive_label) & neg_mask)
    tn = np.sum((y_pred != positive_label) & neg_mask)

    pct_tp = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pct_fp = 100.0 * fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(pct_fp), float(pct_tp)


def trapezoidal_auc(fp_points: np.ndarray | list[float], tp_points: np.ndarray | list[float]) -> float:
    """
    Compute Area Under the ROC Curve using the trapezoidal rule (Section 5.3).
    Expects %FP and %TP in [0, 100]. Automatically appends (100.0, 100.0) if missing
    and returns normalized AUC in [0, 1] (multiply by 10000 to match Table 3's 4-digit integers).
    """
    pts = sorted(zip(fp_points, tp_points), key=lambda p: (p[0], p[1]))

    # Extrapolate (100, 100) if curve does not naturally end there
    if not pts or not (np.isclose(pts[-1][0], 100.0) and np.isclose(pts[-1][1], 100.0)):
        pts.append((100.0, 100.0))

    fps = np.array([p[0] for p in pts], dtype=np.float64) / 100.0
    tps = np.array([p[1] for p in pts], dtype=np.float64) / 100.0

    # np.trapezoid (NumPy 2.0+) replaces deprecated np.trapz
    return float(np.trapezoid(tps, fps))


def roc_convex_hull(points: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    """
    Compute the upper-left ROC Convex Hull using Andrew's monotone chain variant of
    Graham's Scan (Section 5.2, Provost & Fawcett 2001).

    Parameters
    ----------
    points : array-like of shape (M, 2)
        Collection of (%FP, %TP) points across one or more ROC curves.

    Returns
    -------
    hull : np.ndarray of shape (H, 2)
        Ordered vertices forming the upper-left ROC convex hull from (0, 0) to (100, 100).
    """
    pts = np.asarray(points, dtype=np.float64)
    # Include standard ROC anchors (0, 0) and (100, 100)
    anchors = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float64)
    all_pts = np.vstack([pts, anchors]) if len(pts) > 0 else anchors

    # Deduplicate and sort by increasing %FP, then increasing %TP
    unique_pts = np.unique(np.round(all_pts, decimals=8), axis=0)
    sorted_pts = unique_pts[np.lexsort((unique_pts[:, 1], unique_pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # 2D cross product (OA x OB): > 0 means left turn (concave), <= 0 means right/collinear (convex upper hull)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    upper_hull: list[np.ndarray] = []
    for p in sorted_pts:
        # Keep only highest %TP for identical %FP
        if upper_hull and np.isclose(upper_hull[-1][0], p[0]):
            upper_hull.pop()
        while len(upper_hull) >= 2 and cross(upper_hull[-2], upper_hull[-1], p) >= 0:
            upper_hull.pop()
        upper_hull.append(p)

    return np.array(upper_hull, dtype=np.float64)