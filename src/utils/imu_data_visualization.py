import pytz
from matplotlib import pyplot as plt
import pytz


def plot_raw_sensor(data, title):
    """
    Plot sensor data over time. This function can handle either a single sensor DataFrame or a tuple of DataFrames for both accelerometer and gyroscope.

    Args:
        data (tuple of pandas.DataFrame or pandas.DataFrame): Sensor data to be plotted.
            If a tuple is provided, it should contain two DataFrames (one for accelerometer and one for gyroscope).
            If a single DataFrame is provided, it will be considered as data for one sensor type.
        title (str, optional): Title for the plot. Defaults to "Sensor Data". Used as the sensor name when a single DataFrame is provided.

    Plots:
        Matplotlib figures displaying the sensor data for each axis over time.
    """
    # Determine if data is a tuple (both sensors) or a single DataFrame (one sensor)
    if isinstance(data, tuple):
        sensors = data
        sensor_names = ['Accelerometer', 'Gyroscope']
    else:
        sensors = [data]
        sensor_names = [title]  # Use the provided title as the sensor name

    for sensor, sensor_name in zip(sensors, sensor_names):
        x = sensor['x']
        y = sensor['y']
        z = sensor['z']
        time = sensor['time']
        # Ensure the time is timezone-aware before converting to naive
        time = time.dt.tz_localize(pytz.UTC).dt.tz_convert(None)

        plt.figure(figsize=(12, 8))
        plt.plot(time, x, label='x')
        plt.plot(time, y, label='y')
        plt.plot(time, z, label='z')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'{title}: {sensor_name} Data over Time')
        plt.legend()
        plt.show()


def plot_axis_comparison(original_data, interpolated_data, axis, title):
    """
    Plot a comparison of original and interpolated sensor data for a specified axis.

    The function converts timezone-naive 'time' column in the data to timezone-aware UTC
    for consistent plotting.

    Args:
        original_data (pandas.DataFrame): DataFrame containing the original sensor data with 'time' column.
        interpolated_data (pandas.DataFrame): DataFrame containing the interpolated sensor data with 'time' column.
        axis (str): The axis to be compared ('x', 'y', or 'z').
        title (str): A title for the plot to indicate the comparison context.

    Plots:
        Matplotlib figures comparing the original and interpolated data for the specified axis over time.
    """
    # Ensure the axis is one of the expected values
    if axis not in ['x', 'y', 'z']:
        raise ValueError("The axis argument must be 'x', 'y', or 'z'.")

    # Extract values and timestamps for the specified axis from original data
    original_values = original_data[axis]
    time_original = original_data['time']
    # Convert to timezone-aware UTC if needed
    time_original = time_original.dt.tz_localize(pytz.UTC).dt.tz_convert(None)

    # Extract values and timestamps for the specified axis from interpolated data
    interpolated_values = interpolated_data[axis]
    time_interpolated = interpolated_data['time']
    # Convert to timezone-aware UTC if needed
    time_interpolated = time_interpolated.dt.tz_localize(pytz.UTC).dt.tz_convert(None)

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(time_original, original_values, label=f'Original {axis.upper()}-axis', linestyle='-')
    plt.plot(time_interpolated, interpolated_values, label=f'Interpolated {axis.upper()}-axis', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel(f'{axis.upper()}-axis Values')
    plt.title(f'Comparison of Original and Interpolated {axis.upper()}-axis Data: ' + title)
    plt.legend()
    plt.show()


def plot_parquet(M, title_prefix='Subject'):
    """
    Plot the sensor data stored in a DataFrame.

    This function separates accelerometer and gyroscope data and plots each one's components
    over time. It utilizes a nested function to create individual plots for each sensor type.

    Args:
        M (pandas.DataFrame): The DataFrame containing the sensor data with time ('t') and the sensor
                              readings ('a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z').
        title_prefix (str): A prefix for the plot title to indicate the context or subject.

    No return value; this function outputs the plots directly.

    The function assumes the DataFrame is structured with specific column names and does not handle exceptions
    related to data formatting.
    """
    # Split the data into accelerometer and gyroscope parts
    acc_data = M[['t', 'a_x', 'a_y', 'a_z']]
    gyro_data = M[['t', 'g_x', 'g_y', 'g_z']]

    # Define a helper function to plot each sensor
    def plot_each_sensor(data, sensor_title):
        plt.figure(figsize=(12, 8))
        for axis in data.columns[1:]:
            plt.plot(data['t'], data[axis], label=axis)
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'{sensor_title} Data over Time for {title_prefix}')
        plt.legend()
        plt.show()

    # Plot accelerometer data
    plot_each_sensor(acc_data, 'Accelerometer')

    # Plot gyroscope data
    plot_each_sensor(gyro_data, 'Gyroscope')
