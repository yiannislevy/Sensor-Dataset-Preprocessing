import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt, medfilt
import json


# 1. Sync data based on their common time range
def sync(acc_data, gyro_data):
    """
    Synchronizes accelerometer and gyroscope data based on their time stamps.

    Parameters:
    acc_data (pd.DataFrame): DataFrame containing accelerometer data with columns ['time', 'x', 'y', 'z'].
    gyro_data (pd.DataFrame): DataFrame containing gyroscope data with columns ['time', 'x', 'y', 'z'].

    Returns:
    pd.DataFrame: Synchronized accelerometer data.
    pd.DataFrame: Synchronized gyroscope data.
    """
    # Ensure the time columns are in datetime format
    acc_data['time'] = pd.to_datetime(acc_data['time'])
    gyro_data['time'] = pd.to_datetime(gyro_data['time'])

    # Find the common time range
    common_start_time = max(acc_data['time'].min(), gyro_data['time'].min())
    common_end_time = min(acc_data['time'].max(), gyro_data['time'].max())

    # Filter data based on the common time range
    acc_synced = acc_data[(acc_data['time'] >= common_start_time) & (acc_data['time'] <= common_end_time)]
    gyro_synced = gyro_data[(gyro_data['time'] >= common_start_time) & (gyro_data['time'] <= common_end_time)]

    return acc_synced, gyro_synced


# 2. Resample data to a common frequency
def resample(acc_data, gyro_data, target_freq=100):
    """
    Resamples synchronized accelerometer and gyroscope data to a common frequency.

    Parameters:
    acc_data (pd.DataFrame): DataFrame containing synchronized accelerometer data.
    gyro_data (pd.DataFrame): DataFrame containing synchronized gyroscope data.
    target_freq (float): Target frequency for resampling.

    Returns:
    pd.DataFrame: Resampled accelerometer data.
    pd.DataFrame: Resampled gyroscope data.
    """
    def interpolate_to_common_timestamps(sensor_data, time_range):
        sensor_data = sensor_data.copy()
        sensor_data['time_numeric'] = sensor_data['time'].astype(np.int64)
        common_time_range_numeric = time_range.astype(np.int64)

        interpolated_data = {}
        for axis in ['x', 'y', 'z']:
            f = interp1d(sensor_data['time_numeric'], sensor_data[axis], kind='linear', fill_value='extrapolate')
            interpolated_data[axis] = f(common_time_range_numeric)

        interpolated_data['time'] = time_range
        return pd.DataFrame(interpolated_data)

    # Generate new common timestamps at the target frequency
    common_time_range = pd.date_range(start=acc_data['time'].min(),
                                      end=acc_data['time'].max(),
                                      freq=pd.Timedelta(seconds=1/target_freq))

    # Interpolate accelerometer and gyroscope data
    acc_resampled = interpolate_to_common_timestamps(acc_data, common_time_range)
    gyro_resampled = interpolate_to_common_timestamps(gyro_data, common_time_range)

    return acc_resampled, gyro_resampled


# 3. Remove the gravity component from the accelerometer data
def remove_gravity(data, sample_rate=100, cutoff_hz=1):
    """
    Apply a custom high-pass filter to the accelerometer data.

    Args:
        data (pandas.DataFrame): DataFrame containing accelerometer data.
        sample_rate (int): Sample rate of the accelerometer data.
        cutoff_hz (int): Cutoff frequency for the high-pass filter.

    Returns:
        pandas.DataFrame: DataFrame with the filtered accelerometer data.
    """
    num_taps = sample_rate * 5 + 1

    hp_filter = firwin(num_taps, cutoff_hz / (sample_rate / 2), pass_zero=False)

    data['x'] = filtfilt(hp_filter, 1.0, data['x'])
    data['y'] = filtfilt(hp_filter, 1.0, data['y'])
    data['z'] = filtfilt(hp_filter, 1.0, data['z'])

    return data


# 4. Apply median filtering to smooth the data
def median_filter(acc_data, gyro_data, filter_order=5):
    """
    Apply a median filter to the accelerometer and gyroscope sensor data to smooth it.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer sensor data with 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope sensor data with 'x', 'y', 'z' columns.
        filter_order (int): The order of the median filter (kernel size). Defaults to 5.

    Returns:
        tuple: A tuple containing two DataFrames:
               - DataFrame containing the filtered accelerometer sensor data.
               - DataFrame containing the filtered gyroscope sensor data.

    Raises:
        ValueError: If the filtering fails due to an inappropriate filter order.
    """
    try:
        # Apply median filter to each axis for accelerometer data
        filtered_acc_x = medfilt(acc_data['x'], kernel_size=filter_order)
        filtered_acc_y = medfilt(acc_data['y'], kernel_size=filter_order)
        filtered_acc_z = medfilt(acc_data['z'], kernel_size=filter_order)

        # Create a new DataFrame for the filtered accelerometer data
        filtered_acc_data = acc_data.copy()
        filtered_acc_data['x'] = filtered_acc_x
        filtered_acc_data['y'] = filtered_acc_y
        filtered_acc_data['z'] = filtered_acc_z

        # Apply median filter to each axis for gyroscope data
        filtered_gyro_x = medfilt(gyro_data['x'], kernel_size=filter_order)
        filtered_gyro_y = medfilt(gyro_data['y'], kernel_size=filter_order)
        filtered_gyro_z = medfilt(gyro_data['z'], kernel_size=filter_order)

        # Create a new DataFrame for the filtered gyroscope data
        filtered_gyro_data = gyro_data.copy()
        filtered_gyro_data['x'] = filtered_gyro_x
        filtered_gyro_data['y'] = filtered_gyro_y
        filtered_gyro_data['z'] = filtered_gyro_z

        return filtered_acc_data, filtered_gyro_data
    except ValueError as e:
        print(f"Filtering failed due to inappropriate filter order. Error: {e}")
        raise


# 5. Mirror hand gestures based on the right hand (for left-handed subjects only)
def mirror_left_to_right(acc_data, gyro_data):
    """
    Mirror accelerometer and gyroscope data for left-handed subjects to match right-handed orientation.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'x', 'y', 'z' columns.

    Returns:
        tuple of pandas.DataFrame: Tuple containing the mirrored accelerometer and gyroscope DataFrames.
    """
    # No try-except block needed here as there is no file I/O or calculations likely to raise exceptions
    acc_data['x'] = -acc_data['x']
    gyro_data['y'] = -gyro_data['y']
    gyro_data['z'] = -gyro_data['z']
    return acc_data, gyro_data


# 6. Align data with Microsoft's Band 2 Watch orientation standard
def align_old_msft_watch(acc_data, gyro_data):
    """
    Align accelerometer and gyroscope data based on the orientation of the Microsoft Band 2 watch.

    This function is intended for data collected from the right hand. If data are collected from the left hand,
    use the 'mirror_left_to_right' function from imu_preprocessing.py before aligning the data. The alignment
    assumes the watch is worn on the wrist with the screen facing inwards towards the body.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with columns 'x', 'y', 'z'.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with columns 'x', 'y', 'z'.

    Returns:
        tuple of pandas.DataFrame: The aligned accelerometer data (acc_data) and gyroscope data (gyro_data).
    """
    acc_data['y'] = -acc_data['y']
    acc_data['z'] = -acc_data['z']
    gyro_data['y'] = -gyro_data['y']
    gyro_data['z'] = -gyro_data['z']

    return acc_data, gyro_data


# 7. Transform data units from m/s^2 to g and rad/s to deg/s
def transform_data(acc_data, gyro_data):
    """
    Transforms accelerometer data from m/s^2 to g-forces, and gyroscope data from rad/s to degrees/s.

    Parameters:
    acc_df (pd.DataFrame): DataFrame containing accelerometer data with columns ['time', 'x', 'y', 'z'].
    gyr_df (pd.DataFrame): DataFrame containing gyroscope data with columns ['time', 'x', 'y', 'z'].

    Returns:
    pd.DataFrame: Transformed accelerometer data in g-forces.
    pd.DataFrame: Transformed gyroscope data in degrees/s.
    """
    # Constants for conversion
    g_force_conversion = 9.81
    degrees_conversion = 57.2958

    # Create copies of the DataFrames to avoid modifying the original data
    acc_transformed = acc_data.copy()
    gyro_transformed = gyro_data.copy()

    # Transform accelerometer data from m/s^2 to g-forces
    for axis in ['x', 'y', 'z']:
        acc_transformed[axis] = acc_transformed[axis] / g_force_conversion

    # Transform gyroscope data from rad/s to degrees/s
    for axis in ['x', 'y', 'z']:
        gyro_transformed[axis] = gyro_transformed[axis] * degrees_conversion

    return acc_transformed, gyro_transformed


# 8. Standardize the data (subtract the mean and divide by the standard deviation)
def standardize_data(acc_data, gyro_data, mean_std_path):
    """
    Standardizes the accelerometer and gyroscope sensor data using pre-calculated mean and standard deviation values
    for each axis from a JSON file, while keeping the timestamp column intact.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer sensor data with timestamp, 'x', 'y', and 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope sensor data with timestamp, 'x', 'y', and 'z' columns.
        mean_std_path (str): Path to the JSON file containing pre-calculated mean and standard deviation values.

    Returns:
        tuple of pandas.DataFrame: Tuple containing the standardized accelerometer and gyroscope DataFrames.
    """
    try:
        # Load pre-calculated mean and standard deviation values
        with open(mean_std_path, 'r') as f:
            mean_std = json.load(f)

        # Extract means and standard deviations
        means = mean_std['means']
        std_devs = mean_std['std_devs']

        # The order of means and std_devs is assumed to be [a_x, a_y, a_z, g_x, g_y, g_z]
        axis_labels = ['x', 'y', 'z']

        standardized_acc_data = acc_data.copy()
        standardized_gyro_data = gyro_data.copy()

        # Standardize accelerometer data
        for i, axis in enumerate(axis_labels):
            standardized_acc_data[axis] = (acc_data[axis] - means[i]) / std_devs[i]

        # Standardize gyroscope data
        for i, axis in enumerate(axis_labels):
            standardized_gyro_data[axis] = (gyro_data[axis] - means[i + 3]) / std_devs[i + 3]

        return standardized_acc_data, standardized_gyro_data
    except ValueError as e:
        print(f"Standardization failed. Error: {e}")
        raise


# 9. Combine the data into a single array
def combine_sensor_data(standardized_acc, standardized_gyro):
    """
    Combine standardized accelerometer and gyroscope data into a single DataFrame.

    Args:
        standardized_acc (pandas.DataFrame): DataFrame with standardized accelerometer data and 'time' column.
        standardized_gyro (pandas.DataFrame): DataFrame with standardized gyroscope data and 'time' column.

    Returns:
        pandas.DataFrame: DataFrame with combined accelerometer and gyroscope data.

    Raises:
        ValueError: If the combining fails due to mismatched time columns.
    """
    try:
        # Check if the time columns in both dataframes are identical
        if not standardized_acc['time'].equals(standardized_gyro['time']):
            raise ValueError('Time columns do not match.')

        # Combine the dataframes
        combined_data = pd.DataFrame({
            't': standardized_acc['time'],
            'a_x': standardized_acc['x'],
            'a_y': standardized_acc['y'],
            'a_z': standardized_acc['z'],
            'g_x': standardized_gyro['x'],
            'g_y': standardized_gyro['y'],
            'g_z': standardized_gyro['z']
        })

        return combined_data
    except ValueError as e:
        print(f"Time columns do not match. Error: {e}")
        raise
