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


def linear_resample(data, target_freq):
    """
    Linearly resample raw data to a target frequency.

    Args:
        data (pandas.Series): Series containing the raw data.
        target_freq (int): The target frequency in Hz to resample the data to.

    Returns:
        pandas.DataFrame: The resampled data DataFrame.
    """
    # Assuming the data is sampled at 1Hz originally, create a timestamp index
    timestamps = pd.date_range(start='2023-01-01', periods=len(data), freq='S')

    # Create a DataFrame with the original data
    df = pd.DataFrame(data=data.values, index=timestamps, columns=['weight'])

    # Determine the number of samples in the resampled data
    total_samples = int(np.ceil((timestamps[-1] - timestamps[0]).total_seconds() * target_freq))

    # Create a new timestamp index for the target frequency
    target_timestamps = pd.date_range(start=timestamps[0], periods=total_samples, freq=pd.DateOffset(seconds=1/target_freq))

    # Reindex the original DataFrame to the target timestamps with linear interpolation
    resampled_df = df.reindex(target_timestamps).interpolate(method='linear')

    return resampled_df

# TODO evaluate its ability to detect bites and then keep it or not
def process_meal_data(data, stability_range=3, max_decrease=70):
    """
    Process the meal weight data to ensure decreases are within specified limits and stable.

    Parameters:
    data (list or np.array): The raw meal weight data.
    stability_range (int): The allowed fluctuation range for considering a decrease as stable (default 3 grams).
    max_decrease (int): Maximum allowed decrease in weight between two consecutive measurements (default 40 grams).

    Returns:
    np.array: The processed meal weight data with filtered decreases.
    """
    if len(data) < 2:
        return np.array(data)  # Not enough data to process

    processed_data = np.copy(data)
    for i in range(1, len(data)):
        current_decrease = processed_data[i - 1] - processed_data[i]

        # Check if decrease is more than the maximum allowed
        if current_decrease > max_decrease:
            processed_data[i] = processed_data[i - 1]
        else:
            # Check for stability in the next measurement (if exists)
            if i < len(data) - 1:
                next_decrease = processed_data[i] - data[i + 1]
                if abs(next_decrease) > stability_range:
                    processed_data[i] = processed_data[i - 1]

    return processed_data
