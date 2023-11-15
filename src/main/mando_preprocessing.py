import numpy as np
from scipy.signal import savgol_filter

import pandas as pd
from scipy.signal import savgol_filter
# TODO cleanup code

def preprocess_mandometer_data(file_path, plate_weight=200, window_size=5, polyorder=2):
    """
    Preprocesses raw Mandometer data from a text file. This involves removing jitter,
    subtracting plate weight, setting negative values to zero, and optionally classifying
    data points as stable or unstable based on weight constancy.

    Args:
        file_path (str): Path to the raw Mandometer data file.
        plate_weight (int, optional): Weight of the empty plate to be subtracted. Defaults to 200.
        window_size (int, optional): Window size for Savitzky-Golay filter to smooth the data. Defaults to 5.
        polyorder (int, optional): Polynomial order for Savitzky-Golay filter. Defaults to 2.

    Returns:
        pandas.DataFrame: DataFrame containing preprocessed Mandometer data.

    Raises:
        IOError: If the file could not be read.
        ValueError: If the file content is not as expected.
    """
    try:
        # Read the raw data from the file into a DataFrame
        data = pd.read_csv(file_path, header=None, names=['weight'])

        # Remove jitter using Savitzky-Golay filter
        data['smoothed_weight'] = savgol_filter(data['weight'], window_size, polyorder)

        # Subtract plate weight and set negative values to zero
        data['adjusted_weight'] = data['smoothed_weight'].apply(lambda x: max(x - plate_weight, 0))

        return data[['adjusted_weight']]
    except IOError as e:
        print(f"Error reading the file: {e}")
        raise
    except ValueError as e:
        print(f"Error processing the file content: {e}")
        raise


def classify_and_segment_data(data, stability_threshold=0.5):
    """
    Classifies data points as stable or unstable based on weight constancy and
    segments the data into intervals corresponding to different eating events.

    Args:
        data (numpy.ndarray): Preprocessed Mandometer data.
        stability_threshold (float, optional): Threshold for classifying data stability. Defaults to 0.5.

    Returns:
        numpy.ndarray: Array with classified and segmented data.
    """
    # Calculate the rate of change in weight
    rate_of_change = np.abs(np.diff(data, prepend=data[0]))

    # Classify as stable (0) or unstable (1) based on the rate of change
    stability_classification = (rate_of_change > stability_threshold).astype(int)

    # TODO: Implement specific logic for segmenting data into eating events

    return stability_classification


def resample_mando(data):
    # TODO docstring
    # TODO implement
    print("Resampling...")