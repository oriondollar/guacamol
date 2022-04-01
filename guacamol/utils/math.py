from typing import List

import numpy as np
from scipy.spatial.distance import cosine as cos_distance


def arithmetic_mean(values: List[float]) -> float:
    """
    Computes the arithmetic mean of a list of values.
    """
    return sum(values) / len(values)


def geometric_mean(values: List[float]) -> float:
    """
    Computes the geometric mean of a list of values.
    """
    a = np.array(values)
    return a.prod() ** (1.0 / len(a))

def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:
     sim = <r, g> / ||r|| / ||g||

     https://github.com/molecularsets/moses
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)
