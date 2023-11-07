import numpy as np
import pandas as pd
from scipy.signal import resample_poly, medfilt, filtfilt, butter


# 1. Estimate sampling frequency of each sensor
def calculate_frequency(data):
    """
    Calculate the sampling frequency of sensor data based on the time differences between data points.

    Args:
        data (pandas.DataFrame): Dataframe containing sensor data with a 'time' column.

    Returns:
        float: The calculated average sampling frequency in Hertz (Hz).
    """
    try:
        time_diffs = data['time'].diff().dropna()
        time_diffs_ms = time_diffs.dt.total_seconds() * 1000
        avg_interval = time_diffs_ms.mean()
        return 1000 / avg_interval
    except Exception as e:
        print(f"An error occurred while calculating the frequency: {e}")
        raise


# 2. Resample the data to a common frequency, interpolate missing values, and synchronize the timestamps
def resample(acc_data, gyro_data, target_freq):
    """
    Resample accelerometer and gyroscope data to a common frequency, interpolate missing values, and synchronize timestamps.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'time' and 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'time' and 'x', 'y', 'z' columns.
        target_freq (int): The target frequency in Hz to resample the data to.

    Returns:
        tuple of pandas.DataFrame: Tuple containing the resampled accelerometer and gyroscope DataFrames.

    Raises:
        ValueError: If the resampling fails due to inconsistent data.
    """
    try:
        # Synchronize the timestamps
        start_time = max(acc_data['time'].iloc[0], gyro_data['time'].iloc[0])
        end_time = min(acc_data['time'].iloc[-1], gyro_data['time'].iloc[-1])

        acc_data_sync = acc_data[(acc_data['time'] >= start_time) & (acc_data['time'] <= end_time)]
        gyro_data_sync = gyro_data[(gyro_data['time'] >= start_time) & (gyro_data['time'] <= end_time)]

        # Calculate original frequency
        original_freq = int(round((calculate_frequency(acc_data_sync) + calculate_frequency(gyro_data_sync)) / 2))

        L = target_freq  # Up-sampling factor
        M = original_freq  # Down-sampling factor

        # Resample data for each axis using resample_poly
        acc_resampled = np.array([resample_poly(acc_data_sync[axis], L, M) for axis in ['x', 'y', 'z']]).T
        gyro_resampled = np.array([resample_poly(gyro_data_sync[axis], L, M) for axis in ['x', 'y', 'z']]).T

        # Generate timestamps
        resampled_len = min(len(acc_resampled), len(gyro_resampled))
        resampled_times = pd.date_range(start=start_time, periods=resampled_len,
                                        freq=pd.DateOffset(seconds=1 / target_freq))

        # Create DataFrames
        acc_resampled_df = pd.DataFrame(acc_resampled[:resampled_len], columns=['x', 'y', 'z'])
        acc_resampled_df['time'] = resampled_times

        gyro_resampled_df = pd.DataFrame(gyro_resampled[:resampled_len], columns=['x', 'y', 'z'])
        gyro_resampled_df['time'] = resampled_times

        return acc_resampled_df, gyro_resampled_df
    except ValueError as e:
        print(f"Resampling failed due to inconsistent data. Error: {e}")
        raise


# 3. Apply nth order median filtering to smooth the data
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


# 4. Remove the gravity component from the accelerometer data
def remove_gravity(data, cutoff_hz, sample_rate, order=5):
    """
    Remove the gravity component from accelerometer data using a high-pass Butterworth filter.

    Args:
        data (pandas.DataFrame): DataFrame containing accelerometer data with 'x', 'y', 'z' columns.
        cutoff_hz (float): The cutoff frequency of the high-pass filter in Hz.
        sample_rate (int): The sampling rate of the data in Hz.
        order (int, optional): The order of the Butterworth filter. Defaults to 5.

    Returns:
        pandas.DataFrame: DataFrame with the gravity component removed from the accelerometer data.

    Raises:
        ValueError: If the filter configuration is invalid.
    """
    try:
        # Calculate the Nyquist frequency
        nyquist = 0.5 * sample_rate

        # Normalize the frequency by the Nyquist frequency
        normal_cutoff = cutoff_hz / nyquist

        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='high', analog=False)

        # Apply the filter to each axis using filtfilt for zero phase delay
        filtered_x = filtfilt(b, a, data['x'])
        filtered_y = filtfilt(b, a, data['y'])
        filtered_z = filtfilt(b, a, data['z'])

        # Create a new DataFrame for the filtered data
        filtered_data = data.copy()
        filtered_data['x'] = filtered_x
        filtered_data['y'] = filtered_y
        filtered_data['z'] = filtered_z

        return filtered_data
    except ValueError as e:
        print(f"Filter configuration is invalid. Error: {e}")
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


# 6. Standardize the data (subtract the mean and divide by the standard deviation)
def standardize_data(acc_data, gyro_data):
    """
    Standardize the sensor data by subtracting the mean and dividing by the standard deviation for each axis.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer sensor data with 'x', 'y', and 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope sensor data with 'x', 'y', and 'z' columns.

    Returns:
        tuple of pandas.DataFrame: Tuple containing the standardized accelerometer and gyroscope DataFrames.

    Raises:
        ValueError: If standardization fails due to data issues.
    """
    try:
        standardized_acc_data = acc_data.copy()
        for axis in ['x', 'y', 'z']:
            axis_mean = acc_data[axis].mean()
            axis_std = acc_data[axis].std()
            standardized_acc_data[axis] = (acc_data[axis] - axis_mean) / axis_std

        standardized_gyro_data = gyro_data.copy()
        for axis in ['x', 'y', 'z']:
            axis_mean = gyro_data[axis].mean()
            axis_std = gyro_data[axis].std()
            standardized_gyro_data[axis] = (gyro_data[axis] - axis_mean) / axis_std

        return standardized_acc_data, standardized_gyro_data
    except ValueError as e:
        print(f"Standardization failed. Error: {e}")
        raise


# 7. Combine the data into a single array
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
