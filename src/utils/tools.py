import pandas as pd
import matplotlib.pyplot as plt
import os


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


def convert_time_to_seconds(time_str):
    """
    Convert a time string in the format "HH:MM:SS.SSSS" to seconds.

    Parameters:
    - time_str (str): Time string to convert.

    Returns:
    - float: Time in seconds.
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def plot_parquet(M, title_prefix):
    """
    Plot the time series data for accelerometer and gyroscope from a DataFrame.

    This function splits the combined accelerometer and gyroscope data into two parts,
    and plots each sensor's time series in separate figures.

    Args:
        M (pandas.DataFrame): DataFrame with combined sensor data including time ('t'),
                              accelerometer ('a_x', 'a_y', 'a_z'), and gyroscope ('g_x', 'g_y', 'g_z') columns.
        title_prefix (str): A prefix for the plot title, usually indicating the subject or condition of the data.

    The function does not return a value. It generates and displays plots.
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
        plt.ylabel('Sensor Values')
        plt.title(f'{sensor_title} Data over Time for {title_prefix}')
        plt.legend()
        plt.show()

    # Plot accelerometer data
    plot_each_sensor(acc_data, 'Accelerometer')

    # Plot gyroscope data
    plot_each_sensor(gyro_data, 'Gyroscope')


def plot_pickle(processed_data_directory, subject_id, title_prefix):
    """
    Load sensor data from a pickle file and plot the time series data for accelerometer and gyroscope.

    Args:
        processed_data_directory (str): The directory where the pickle files are stored.
        subject_id (str): The subject identifier used to locate the pickle file.
        title_prefix (str): A prefix for the plot title, usually indicating the subject or condition of the data.
    """
    # Define the file path
    file_path = os.path.join(processed_data_directory, subject_id, f"{subject_id}.pkl")

    # Load the data from pickle file
    M = pd.read_pickle(file_path)

    # Split the data into accelerometer and gyroscope parts
    acc_data = M[['t', 'a_x', 'a_y', 'a_z']]
    gyro_data = M[['t', 'g_x', 'g_y', 'g_z']]

    # Define a helper function to plot each sensor
    def plot_each_sensor(data, sensor_title):
        plt.figure(figsize=(12, 8))
        for axis in data.columns[1:]:
            plt.plot(data['t'], data[axis], label=axis)
        plt.xlabel('Time')
        plt.ylabel('Sensor Values')
        plt.title(f'{sensor_title} Data over Time for {title_prefix}')
        plt.legend()
        plt.show()

    # Plot accelerometer data
    plot_each_sensor(acc_data, 'Accelerometer')

    # Plot gyroscope data
    plot_each_sensor(gyro_data, 'Gyroscope')


def save_data(data, processed_data_directory, filename):
    """
    Save data to a pickle file.

    Args:
        data: Data to be saved, can be of any type.
        processed_data_directory (str): Directory to save the data.
        filename (str): Name of the file to save.
    """
    # Create the directory for the file if it doesn't exist
    file_dir = os.path.join(processed_data_directory)
    os.makedirs(file_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(file_dir, f"{filename}.pkl")

    # Save the data in pickle format
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
