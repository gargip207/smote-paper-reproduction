from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _normalize_targets(raw_target: np.ndarray) -> np.ndarray:
    return np.array([str(val).strip().removesuffix(".0") for val in raw_target])


def load_benchmark_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and binarize benchmark datasets matching Table 2 of Chawla et al. (2002).
    Returns (X, y) where y == 1 is the minority (positive) class and y == 0 is majority.
    """
    name = name.lower().strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if name == "pima":
        # 768 samples: 500 majority (0), 268 minority (1)
        data = fetch_openml("diabetes", version=1, as_frame=False, parser="liac-arff", data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (_normalize_targets(data.target) == "tested_positive").astype(int)

    elif name == "phoneme":
        # 5404 samples, 5 features: 3818 majority ('1'), 1586 minority ('2')
        data = fetch_openml("phoneme", version=1, as_frame=False, parser="liac-arff", data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (_normalize_targets(data.target) == "2").astype(int)

    elif name == "satimage":
        # 6430 samples: smallest class (class 4, 626 samples) vs remaining 5 classes
        data = fetch_openml("satimage", version=1, as_frame=False, parser="liac-arff", data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        targets = _normalize_targets(data.target)
        classes, counts = np.unique(targets, return_counts=True)
        minority_cls = classes[np.argmin(counts)]
        y = (targets == minority_cls).astype(int)
    elif name == "mammography":
        # 11183 samples: 10923 majority ('-1'), 260 minority ('1')
        data = fetch_openml("mammography", version=1, as_frame=False, parser="liac-arff", data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (_normalize_targets(data.target) == "1").astype(int)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: pima, phoneme, satimage, mammography.")

    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        raise RuntimeError(f"Dataset '{name}' binarization failed (min={np.sum(y==1)}, maj={np.sum(y==0)}).")

    return X, y