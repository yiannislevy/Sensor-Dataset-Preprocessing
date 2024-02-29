import numpy as np


def calculate_intervals(bite_events_array):
    """
    Calculate intervals between successive bites.

    Parameters:
    - bite_events_array: numpy.ndarray, array of bite events with start and end times.

    Returns:
    - numpy.ndarray, array of intervals between bites.
    """
    return np.diff(bite_events_array[:, 0])
