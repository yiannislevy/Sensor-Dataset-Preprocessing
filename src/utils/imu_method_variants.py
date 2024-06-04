import numpy as np
import pandas as pd
from scipy.signal import resample_poly, firwin, lfilter, filtfilt, butter, medfilt
from scipy.interpolate import interp1d
from src.utils.imu_data_visualization import plot_raw_sensor
from src.utils.tools import calculate_frequency


# If moving average filtering is required, use this version >
def moving_average_filter(acc_data, gyro_data, filter_length=25):
    """
    Apply a moving average filter with a specified length to accelerometer and gyroscope sensor data.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer sensor data with 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope sensor data with 'x', 'y', 'z' columns.
        filter_length (int): The length of the moving average filter. Defaults to 25.

    Returns:
        tuple: A tuple containing two DataFrames:
               - DataFrame containing the filtered accelerometer sensor data.
               - DataFrame containing the filtered gyroscope sensor data.
    """
    # Create the filter kernel (uniform weights)
    filter_kernel = np.ones(filter_length) / filter_length

    # Apply moving average filter to each axis for accelerometer and gyroscope data
    filtered_acc_data = acc_data.copy()
    filtered_gyro_data = gyro_data.copy()

    for axis in ['x', 'y', 'z']:
        filtered_acc_data[axis] = np.convolve(acc_data[axis], filter_kernel, mode='same')
        filtered_gyro_data[axis] = np.convolve(gyro_data[axis], filter_kernel, mode='same')

    return filtered_acc_data, filtered_gyro_data


# Latest version >
def remove_gravity_butter(data, sample_rate, cutoff_hz=1, order=5):
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


def remove_gravity_with_taps(data, cutoff_hz, sample_rate, numtaps=513):
    """
    Remove the gravity component from accelerometer data using a high-pass FIR filter.

    This function designs a Finite Impulse Response (FIR) high-pass filter with the specified number of taps
    and cutoff frequency to remove the gravity component from accelerometer data. The gravity component is
    considered to be the low-frequency signal that does not represent the dynamic movements of the sensor.

    Args:
        data (pandas.DataFrame): DataFrame containing accelerometer data with 'x', 'y', and 'z' columns.
        cutoff_hz (float): The cutoff frequency of the high-pass filter in Hz.
        sample_rate (int): The sampling rate of the data in Hz.
        numtaps (int): The number of taps in the filter, which determines its order. Default is 513.

    Returns:
        pandas.DataFrame: DataFrame with the gravity component removed from the accelerometer data.

    The function assumes that the 'x', 'y', and 'z' columns in the input DataFrame contain uniformly sampled
    time-series data. The use of a high-pass filter is to isolate the dynamic movements by filtering out the
    constant or slowly varying part of the signal, attributed to gravity.
    """
    # Design the high-pass filter
    hp_coefficients = firwin(numtaps, cutoff_hz, pass_zero=False, fs=sample_rate)

    # Apply the filter to each axis
    filtered_x = lfilter(hp_coefficients, 1.0, data['x'])
    filtered_y = lfilter(hp_coefficients, 1.0, data['y'])
    filtered_z = lfilter(hp_coefficients, 1.0, data['z'])

    # Create a new DataFrame for the filtered data
    filtered_data = data.copy()
    filtered_data['x'] = filtered_x
    filtered_data['y'] = filtered_y
    filtered_data['z'] = filtered_z

    return filtered_data


# This was the last version of the resampling before it was refactored
def resample_sensor_data_last(acc_data, gyro_data, target_freq='10L'):
    """
    Process accelerometer and gyroscope data to a common frequency using cubic interpolation.

    This function synchronizes accelerometer and gyroscope data within their overlapping timeframe and
    resamples the data points using cubic interpolation to match a specified target frequency. It handles
    any duplicates by averaging data points at the same timestamp.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'time', 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'time', 'x', 'y', 'z' columns.
        target_freq (str or pandas.DateOffset, optional): Frequency of the new timestamps, formatted as a string
        or pandas DateOffset. Defaults to '10L', which represents 10 milliseconds.

    Returns:
        tuple of pandas.DataFrame: A tuple containing the processed_micromovements accelerometer and gyroscope DataFrames.

    The function assumes the 'time' column in the input DataFrames is in a datetime-like format and that
    the input DataFrames are sorted in ascending time order with no large gaps.
    """
    # Find the common timeframe
    start_time = max(acc_data['time'].min(), gyro_data['time'].min())
    end_time = min(acc_data['time'].max(), gyro_data['time'].max())

    # Generate new timestamps at the target frequency within the common timeframe
    common_timestamps = pd.date_range(start=start_time, end=end_time, freq=target_freq)

    def interpolate_and_resample(sensor_df, new_timestamps):
        # Drop duplicates and average the data if any duplicates exist
        sensor_df = sensor_df.groupby('time', as_index=False).mean()

        # Convert to UNIX timestamp in seconds, avoiding deprecation warnings
        x = sensor_df['time'].view('int64') / 1e9
        y = sensor_df[['x', 'y', 'z']].values

        # Create polynomial interpolation functions for each axis
        interp_funcs = [interp1d(x, y[:, i], kind='cubic', fill_value='extrapolate') for i in range(3)]

        # Convert new timestamps to UNIX timestamp in seconds, avoiding deprecation warnings
        new_x = new_timestamps.view('int64') / 1e9

        # Interpolate data
        interpolated_data = np.array([func(new_x) for func in interp_funcs]).T  # Transpose to match original shape

        # Create new DataFrame with interpolated data
        interpolated_df = pd.DataFrame(interpolated_data, columns=['x', 'y', 'z'])
        interpolated_df['time'] = new_timestamps

        return interpolated_df

    # Interpolate and resample accelerometer and gyroscope data
    processed_acc_data = interpolate_and_resample(acc_data, common_timestamps)
    processed_gyro_data = interpolate_and_resample(gyro_data, common_timestamps)

    return processed_acc_data, processed_gyro_data


# This was the first version of resampling without common timestamps
def resample_sensor_data_first(acc_data, gyro_data, acc_freq, gyro_freq, target_freq):
    """
    Interpolate accelerometer and gyroscope data to a target frequency using polynomial resampling.

    This function independently resamples the accelerometer and gyroscope data to a common target frequency using
    polynomial interpolation. It calculates upsample and downsample factors based on the greatest common divisor
    between the current and target frequencies.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'time', 'x', 'y', 'z' columns.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'time', 'x', 'y', 'z' columns.
        acc_freq (int): The sampling frequency of the accelerometer data in Hz.
        gyro_freq (int): The sampling frequency of the gyroscope data in Hz.
        target_freq (int): The target frequency in Hz for the resampled data.

    Returns:
        tuple of pandas.DataFrame: A tuple containing the resampled accelerometer and gyroscope DataFrames.

    The function assumes that the input data's 'time' columns are datetime-like and that the input data
    are uniformly sampled at the specified current frequencies without any missing timestamps.
    """

    def interpolate_single_sensor(data, current_freq):
        # Determine the factor to convert current_freq to an integer
        current_freq_int = int(np.ceil(current_freq))

        # Calculate the greatest common divisor to simplify the ratio
        gcd = np.gcd(current_freq_int, target_freq)

        # Calculate upsample and downsample factors
        up = target_freq // gcd
        down = current_freq_int // gcd

        # Resample data for each axis
        x_resampled = resample_poly(data['x'].to_numpy(), up, down)
        y_resampled = resample_poly(data['y'].to_numpy(), up, down)
        z_resampled = resample_poly(data['z'].to_numpy(), up, down)

        # Create new timestamps
        duration = data['time'].iloc[-1] - data['time'].iloc[0]
        total_seconds = duration.total_seconds()
        num_samples = len(x_resampled)
        new_times = pd.date_range(start=data['time'].iloc[0], periods=num_samples,
                                  freq=pd.Timedelta(seconds=total_seconds / num_samples))

        # Create a new DataFrame for the resampled data
        resampled_data = pd.DataFrame({
            'x': x_resampled,
            'y': y_resampled,
            'z': z_resampled,
            'time': new_times
        })

        return resampled_data

    # Interpolate both sensors
    interpolated_acc = interpolate_single_sensor(acc_data, acc_freq)
    interpolated_gyro = interpolate_single_sensor(gyro_data, gyro_freq)

    return interpolated_acc, interpolated_gyro


# Other versions of resampling:


def resample_simple(data, target_freq):
    """
    Resample the sensor data to a specified target frequency using linear interpolation.

    This function uses linear interpolation to resample the data of each axis (x, y, z) to a new time
    grid defined by the target frequency. It creates a new DataFrame with the interpolated values and
    corresponding timestamps.

    Args:
        data (pandas.DataFrame): DataFrame containing the original sensor data with 'time', 'x', 'y', 'z' columns.
        target_freq (int): The target frequency in Hz to which the data will be resampled.

    Returns:
        pandas.DataFrame: DataFrame containing the resampled data with 'x', 'y', 'z', and 'time' columns.

    The function assumes that the time column in the input data is in a format convertible to milliseconds
    and that the data covers a continuous range with no large gaps.
    """
    # Ensure time is in milliseconds
    original_time = (data['time'].astype(np.int64) // 10 ** 6).values
    # Define the new time range based on the target frequency
    new_time = np.arange(original_time[0], original_time[-1], 1000 / target_freq)

    # Interpolate each axis separately
    x_interp = np.interp(new_time, original_time, data['x'].values)
    y_interp = np.interp(new_time, original_time, data['y'].values)
    z_interp = np.interp(new_time, original_time, data['z'].values)

    # Convert numeric timestamps back to datetime objects for the new times
    new_time = pd.to_datetime(new_time, unit='ms')

    # Create a new DataFrame with the interpolated values
    resampled_df = pd.DataFrame({
        'x': x_interp,
        'y': y_interp,
        'z': z_interp,
        'time': new_time
    })

    return resampled_df


def resample_simple_common_timestamp(acc_data, gyro_data, target_freq):
    """
    Interpolate accelerometer and gyroscope data to a common set of timestamps at a target frequency.

    This function first determines the overlapping time range between accelerometer and gyroscope data.
    Then, it creates a common set of timestamps at the target frequency and interpolates both sets of sensor
    data to these timestamps using linear interpolation.

    Args:
        acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'time' column.
        gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'time' column.
        target_freq (int): The target frequency in Hz for the common timestamps.

    Returns:
        tuple of pandas.DataFrame: A tuple containing the interpolated accelerometer and gyroscope DataFrames.

    The function assumes 'time' columns are datetime-like and that both datasets have overlapping time ranges.
    """

    def interpolate_single_sensor(data, timestamp):
        # Interpolate data to align with the common timestamps
        interpolated_data = data.set_index('time').reindex(timestamp).interpolate(method='time')
        interpolated_data['time'] = timestamp  # Assign the common timestamps to the 'time' column
        return interpolated_data

    # Determine the later start time and the earlier end time between the two datasets
    start_time = max(acc_data['time'].min(), gyro_data['time'].min())
    end_time = min(acc_data['time'].max(), gyro_data['time'].max())

    # Create a common set of timestamps at the target frequency within the start and end time bounds
    common_timestamps = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(seconds=1 / target_freq))

    # Interpolate both sensors to the common set of timestamps
    interpolated_acc = interpolate_single_sensor(acc_data, common_timestamps)
    interpolated_gyro = interpolate_single_sensor(gyro_data, common_timestamps)

    return interpolated_acc, interpolated_gyro


# Alternative sync and resample functions
def upsample_sensor_data(sensor_data, target_freq):
    """
        Upsample sensor data to a target frequency using linear interpolation.

        This function converts the 'time' column of the input sensor data to datetime format,
        generates a new time range at the desired target frequency, converts timestamps to a numeric
        format suitable for interpolation, and performs linear interpolation for each axis ('x', 'y', 'z').
        After interpolation, it converts the numeric times back to datetime timestamps.

        Args:
            sensor_data (pandas.DataFrame): DataFrame containing sensor data with columns 'time', 'x', 'y', and 'z'.
            target_freq (int): The target frequency in Hz to which the data will be upsampled.

        Returns:
            pandas.DataFrame: DataFrame containing upsampled sensor data with original columns including 'time'.

        The function assumes that the 'time' column is appropriately formatted to allow datetime operations.
    """
    # Ensure the time column is in datetime format
    sensor_data['time'] = pd.to_datetime(sensor_data['time'])

    # Determine the range of the timestamps
    time_range = pd.date_range(start=sensor_data['time'].min(),
                               end=sensor_data['time'].max(),
                               freq=pd.Timedelta(seconds=1 / target_freq))

    # Convert timestamp to numeric for interpolation
    sensor_data['time_numeric'] = sensor_data['time'].astype(np.int64)
    time_range_numeric = time_range.astype(np.int64)

    # Interpolate for each axis
    interpolated_data = {}
    for axis in ['x', 'y', 'z']:
        f = interp1d(sensor_data['time_numeric'], sensor_data[axis], kind='linear', fill_value='extrapolate')
        interpolated_data[axis] = f(time_range_numeric)

    # Convert numeric times back to timestamps
    interpolated_data['time'] = time_range

    return pd.DataFrame(interpolated_data)


def upsample_and_sync(acc_data, gyro_data, target_freq):
    """
        Upsample accelerometer and gyroscope data to a target frequency and synchronize their timestamps.

        This function first upsamples both accelerometer and gyroscope data to the specified target frequency.
        It then identifies the common timeframe between the upsampled datasets and trims them accordingly to ensure
        they are synchronized. The function also includes diagnostics by plotting the resampled accelerometer data
        and printing its frequency to verify the upsampling process.

        Args:
            acc_data (pandas.DataFrame): DataFrame containing accelerometer data with 'time' column.
            gyro_data (pandas.DataFrame): DataFrame containing gyroscope data with 'time' column.
            target_freq (int): The target frequency in Hz for upsampling.

        Returns:
            tuple of pandas.DataFrame: A tuple containing the synchronized, upsampled accelerometer and gyroscope DataFrames.

        The function assumes both input data sets are formatted with a 'time' column that can be manipulated as datetime.
    """
    # Upsample accelerometer data
    acc_resampled = upsample_sensor_data(acc_data, target_freq)

    plot_raw_sensor([acc_resampled], "Resampled inbetween step")
    print(calculate_frequency(acc_resampled))

    # Upsample gyroscope data
    gyro_resampled = upsample_sensor_data(gyro_data, target_freq)

    # Synchronize the timestamps
    # Since both are up-sampled to the same frequency, we can just align their timestamps
    common_start_time = max(acc_resampled['time'].iloc[0], gyro_resampled['time'].iloc[0])
    common_end_time = min(acc_resampled['time'].iloc[-1], gyro_resampled['time'].iloc[-1])

    # Trim the data to the common timeframe
    acc_resampled = acc_resampled[
        (acc_resampled['time'] >= common_start_time) & (acc_resampled['time'] <= common_end_time)]
    gyro_resampled = gyro_resampled[
        (gyro_resampled['time'] >= common_start_time) & (gyro_resampled['time'] <= common_end_time)]

    return acc_resampled, gyro_resampled
