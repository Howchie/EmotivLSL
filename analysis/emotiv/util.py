"""Small helpers shared by the analysis scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["to_jsonable"]


def to_jsonable(obj):
    """Recursively convert numpy/pandas objects to JSON types, dropping _private keys."""

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj
