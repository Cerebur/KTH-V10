import numpy as np
import numpy.typing as npt

NestedArray = npt.NDArray[np.object_] | npt.NDArray[np.float64] | list


# Added by MB on 16.4.2026 for cleaning nested arrays, so that plotting will work
def clean_array(arr: NestedArray) -> np.ndarray:
    """Flattens arrays.

    Used for the trace_fieldline_v10 function to convert arrays from type
    np.array([np.array([value]),np.array([value]), ...) to np.array(value, value, ...).

    Args:
        arr: nested arrays to be flattened

    Returns:
        Flattened array
    """
    cleaned = []
    for v in arr:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                continue  # skip empty arrays
            cleaned.append(v.item())  # extract scalar
        else:
            cleaned.append(v)
    return np.array(cleaned, dtype=float)
