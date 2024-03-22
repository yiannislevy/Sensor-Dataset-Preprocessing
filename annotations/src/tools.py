import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from src.utils.tools import save_data


def calculate_intervals(bite_events_array):
    """
    Calculate intervals between successive bites.

    Parameters:
    - bite_events_array: numpy.ndarray, array of bite events with start and end times.

    Returns:
    - numpy.ndarray, array of intervals between bites.
    """
    return np.diff(bite_events_array[:, 0])


def align_imu_to_video_timeline(imu, subject_id, save, processed_data_directory=None, filename="imu_relative_time",
                                time_sync_path="time_sync.json"):
    """
    Processes IMU data to synchronize with a video timeline by calculating the elapsed time
    in seconds from the start of the video, considering the specific subject ID. Optionally,
    saves the processed data to a designated directory.

    Parameters:
        imu (pandas.DataFrame): IMU data, expected to include a timestamp column 't'.
        subject_id (str): Identifier for the subject to filter synchronization information.
        save (bool): If True, processed IMU data is saved to disk.
        processed_data_directory (str, optional): Target directory for saving processed data. Required if `save` is True.
        filename (str, optional): Filename for the saved data. Default is "imu_relative_time".
        time_sync_path (str, optional): Path to the JSON file containing synchronization information. Default is "time_sync.json".

    Returns:
        numpy.ndarray: Processed IMU data as a NumPy array, with timestamps relative to the video start time.

    Raises:
        ValueError: If subject ID is not found in the synchronization information or if `processed_data_directory` is not provided when `save` is True.
    """
    # Load synchronization information
    with open(time_sync_path, "r") as file:
        time_sync = json.load(file)

    # Extract synchronization information for the given subject ID
    sync_info = next((item for item in time_sync['sync_info'] if item['subject_id'] == subject_id), None)
    if sync_info is None:
        raise ValueError(f"Subject ID {subject_id} not found in synchronization information.")

    # Calculate video start time as datetime
    video_start_time = datetime.strptime(sync_info['video_start_time'], '%H:%M:%S.%f').time()
    video_start_datetime = datetime.combine(imu['t'].iloc[0].date(), video_start_time)

    # Filter and adjust IMU data based on video start time
    imu_data_filtered = imu[imu['t'] >= video_start_datetime].copy()
    imu_data_filtered['t'] = (imu_data_filtered['t'] - video_start_datetime).dt.total_seconds()
    imu_data_filtered.reset_index(drop=True, inplace=True)

    # Convert to NumPy array for processing efficiency
    imu_data_filtered_np = np.array(imu_data_filtered)

    # Save processed data if requested
    if save:
        if not processed_data_directory:
            raise ValueError("A directory must be specified to save the processed data.")
        save_data(imu_data_filtered_np, processed_data_directory, filename)

    return imu_data_filtered_np


def extract_event_windows(signal_data, ground_truth):
    """
    Extracts signal data corresponding to specified event windows from ground truth.

    Parameters:
    - signal_data (np.ndarray): A 2D NumPy array where the first column represents timestamps,
      and the subsequent columns represent sensor data at those timestamps. It is assumed
      that timestamps are in float64 format.
    - ground_truth (np.ndarray): A 2D NumPy array with each row representing an event,
      containing two columns for the start and end timestamps of the event.

    Returns:
    - event_windows (list of np.ndarray): A list where each element is a 2D NumPy array
      containing the signal data that falls within the start and end times of an event
      from the ground truth.

    This function iterates through each event window defined in the ground truth, extracts
    signal samples that fall within these windows (inclusive of start and end times),
    and returns a list of these samples grouped by event.
    """
    event_windows = []

    for start_time, end_time in ground_truth:
        indices = np.where((signal_data[:, 0] >= start_time) & (signal_data[:, 0] <= end_time))
        event_signal = signal_data[indices]
        event_windows.append(event_signal)

    return event_windows
