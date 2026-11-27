from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_benchmark_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and binarize benchmark datasets matching Table 2 of Chawla et al. (2002).
    Returns (X, y) where y == 1 is the minority (positive) class and y == 0 is majority.
    """
    name = name.lower().strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if name == "pima":
        # 768 samples: 500 majority (0), 268 minority (1)
        data = fetch_openml("diabetes", version=1, as_frame=False, data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (np.asarray(data.target) == "tested_positive").astype(int)

    elif name == "phoneme":
        # 5404 samples, 5 features: 3818 majority (class 1 in OpenML), 1586 minority (class 2)
        data = fetch_openml("phoneme", version=1, as_frame=False, data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (np.asarray(data.target).astype(str) == "2").astype(int)

    elif name == "satimage":
        # 6435 samples: smallest class ('4', 626 samples) vs remaining 5 classes (5809 samples)
        data = fetch_openml("satimage", version=1, as_frame=False, data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (np.asarray(data.target).astype(str) == "4").astype(int)

    elif name == "mammography":
        # 11183 samples: 10923 majority ('-1'), 260 minority ('1')
        data = fetch_openml("mammography", version=1, as_frame=False, data_home=str(DATA_DIR))
        X = np.asarray(data.data, dtype=np.float64)
        y = (np.asarray(data.target).astype(str) == "1").astype(int)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: pima, phoneme, satimage, mammography.")

    return X, y