import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def build_c45_proxy(ccp_alpha: float = 0.002, random_state: int = 42) -> DecisionTreeClassifier:
    """
    Pruned entropy-based decision tree approximating C4.5 Release 8 (Section 5).
    """
    return DecisionTreeClassifier(
        criterion="entropy",
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )


def predict_with_threshold(
    clf: DecisionTreeClassifier,
    X_test: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict positive class (1) if P(y=1 | x) >= threshold (Section 5.4 leaf bias tuning).
    """
    probs = clf.predict_proba(X_test)[:, 1]
    if threshold <= 0.0:
        return np.ones(len(X_test), dtype=int)
    return (probs >= threshold).astype(int)


def build_naive_bayes_with_prior_ratio(
    n_majority: int,
    n_minority: int,
    prior_multiplier: float = 1.0,
) -> GaussianNB:
    """
    Cost-sensitive Naive Bayes where the minority class prior is scaled by
    `prior_multiplier` (swept from 1x to 50x in Section 5 & 5.2).
    """
    w_maj = float(n_majority)
    w_min = float(n_minority) * prior_multiplier
    total = w_maj + w_min
    priors = [w_maj / total, w_min / total]
    return GaussianNB(priors=priors)