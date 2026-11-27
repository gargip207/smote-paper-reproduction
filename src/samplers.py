import numpy as np


def smote(
    X_minority: np.ndarray,
    N: int = 100,
    k: int = 5,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate synthetic minority samples following Chawla et al. (2002), Section 4.2.

    Parameters
    ----------
    X_minority : np.ndarray of shape (T, num_attrs)
        Original minority class feature vectors.
    N : int
        Amount of SMOTE in percent (e.g., 50, 100, 200, 300, 400, 500).
        If N < 100, a random subset of floor((N / 100) * T) samples is SMOTEd at 100%.
        If N >= 100, N is treated as an integral multiple of 100.
    k : int, default=5
        Number of nearest minority neighbors to consider.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    synthetic : np.ndarray of shape (n_synthetic, num_attrs)
        Generated synthetic minority samples (does NOT include original X_minority).
    """
    rng = np.random.default_rng(random_state)
    X_min = np.asarray(X_minority, dtype=np.float64)
    T, num_attrs = X_min.shape

    if T < 2:
        raise ValueError("SMOTE requires at least 2 minority class samples.")

    # Step 1-6 of paper pseudocode: handle N < 100%
    if N < 100:
        n_sub = int((N / 100.0) * T)
        if n_sub == 0:
            return np.empty((0, num_attrs), dtype=np.float64)
        sub_indices = rng.choice(T, size=n_sub, replace=False)
        X_target = X_min[sub_indices]
        n_generations_per_sample = 1
    else:
        X_target = X_min
        n_generations_per_sample = int(N // 100)

    # Effective k cannot exceed T - 1
    k_eff = min(k, T - 1)

    # Step 13-15: Compute k nearest neighbors strictly within X_min (excluding self)
    # Pairwise squared Euclidean distances between X_target and X_min
    diff = X_target[:, np.newaxis, :] - X_min[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)

    # Mask self-distances so a sample never selects itself as a neighbor
    if N < 100:
        dist_sq[np.arange(len(X_target)), sub_indices] = np.inf
    else:
        np.fill_diagonal(dist_sq, np.inf)

    # Indices of k_eff nearest neighbors in X_min for each sample in X_target
    nn_indices = np.argsort(dist_sq, axis=1)[:, :k_eff]

    # Step 17-27: Populate synthetic samples
    n_target = len(X_target)
    total_synthetic = n_target * n_generations_per_sample
    synthetic = np.zeros((total_synthetic, num_attrs), dtype=np.float64)

    new_idx = 0
    for i in range(n_target):
        for _ in range(n_generations_per_sample):
            nn_choice = rng.integers(0, k_eff)
            neighbor = X_min[nn_indices[i, nn_choice]]
            # Paper's inner loop draws a fresh random gap in [0, 1] per attribute
            gap = rng.uniform(0.0, 1.0, size=num_attrs)
            synthetic[new_idx] = X_target[i] + gap * (neighbor - X_target[i])
            new_idx += 1

    return synthetic


def random_undersample_majority(
    X_majority: np.ndarray,
    n_minority_orig: int,
    under_pct: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Undersample the majority class using Chawla et al.'s percentage convention (Section 4.3):
    Undersampling majority at U% means the minority class becomes U% of the remaining
    majority class (i.e., n_maj_new = floor(n_minority_orig / (U / 100))).
    """
    rng = np.random.default_rng(random_state)
    X_maj = np.asarray(X_majority)
    n_maj = len(X_maj)

    target_maj = int(n_minority_orig / (under_pct / 100.0))
    target_maj = max(1, min(target_maj, n_maj))

    chosen_idx = rng.choice(n_maj, size=target_maj, replace=False)
    return X_maj[chosen_idx]


def random_oversample_minority(
    X_minority: np.ndarray,
    N: int,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Oversample minority class with replacement by N% (baseline in Section 4.1 & Fig 4, 5, 19).
    Returns ONLY the newly replicated samples to append to X_minority.
    """
    rng = np.random.default_rng(random_state)
    X_min = np.asarray(X_minority)
    n_new = int((N / 100.0) * len(X_min))
    if n_new == 0:
        return np.empty((0, X_min.shape[1]), dtype=X_min.dtype)
    idx = rng.choice(len(X_min), size=n_new, replace=True)
    return X_min[idx]


def resample_dataset(
    X: np.ndarray,
    y: np.ndarray,
    smote_pct: int | None = None,
    under_pct: int | None = None,
    k: int = 5,
    minority_label: int = 1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply combined SMOTE oversampling and majority undersampling to a training fold.
    """
    rng = np.random.default_rng(random_state)
    X_min = X[y == minority_label]
    X_maj = X[y != minority_label]
    n_min_orig = len(X_min)

    if smote_pct is not None and smote_pct > 0:
        syn = smote(X_min, N=smote_pct, k=k, random_state=int(rng.integers(1e9)))
        X_min_resampled = np.vstack([X_min, syn])
    else:
        X_min_resampled = X_min

    if under_pct is not None and under_pct > 0:
        X_maj_resampled = random_undersample_majority(
            X_maj,
            n_minority_orig=n_min_orig,
            under_pct=under_pct,
            random_state=int(rng.integers(1e9)),
        )
    else:
        X_maj_resampled = X_maj

    X_res = np.vstack([X_min_resampled, X_maj_resampled])
    y_res = np.concatenate([
        np.full(len(X_min_resampled), minority_label, dtype=y.dtype),
        np.full(len(X_maj_resampled), 1 - minority_label, dtype=y.dtype),
    ])

    perm = rng.permutation(len(y_res))
    return X_res[perm], y_res[perm]