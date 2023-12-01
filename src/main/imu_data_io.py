import os
import pandas as pd
import numpy as np

sensor_dtype = np.dtype([
    ("x", ">f"),
    ("y", ">f"),
    ("z", ">f"),
    ("time", ">i8"),
])


# Read in the data
def load_raw_sensor_data(path):
    """
    Load accelerometer and gyroscope sensor data from binary files, convert timestamps, and return as pandas DataFrames.

    Args:
        path (str): Path to the directory containing .bin files with 'accelerometer' and 'gyroscope' in their names.

    Returns:
        tuple: Returns two DataFrames, the first with accelerometer data and the second with gyroscope data.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the data cannot be loaded properly.
    """
    try:
        acc_files = [f for f in os.listdir(path) if f.endswith('.bin') and 'accelerometer' in f]
        acc_files = sorted(acc_files, key=lambda x: int(x.split("_")[0]))

        gyro_files = [f for f in os.listdir(path) if f.endswith('.bin') and 'gyroscope' in f]
        gyro_files = sorted(gyro_files, key=lambda x: int(x.split("_")[0]))

        all_acc_data = []
        for file in acc_files:
            boot_time_nanos = int(file.split("_")[0]) * 1e6
            file_path = os.path.join(path, file)
            acc_data = np.fromfile(file_path, dtype=sensor_dtype)
            first_event_time = acc_data['time'][0]
            corrected_timestamps = ((acc_data['time'] - first_event_time) + boot_time_nanos) / 1e9
            corrected_datetimes = pd.to_datetime(corrected_timestamps, unit='s')
            df = pd.DataFrame(acc_data[["x", "y", "z"]].byteswap().newbyteorder())
            df['time'] = corrected_datetimes
            all_acc_data.append(df)
        all_acc_data = pd.concat(all_acc_data)

        all_gyro_data = []
        for file in gyro_files:
            boot_time_nanos = int(file.split("_")[0]) * 1e6
            file_path = os.path.join(path, file)
            gyro_data = np.fromfile(file_path, dtype=sensor_dtype)
            first_event_time = gyro_data['time'][0]
            corrected_timestamps = ((gyro_data['time'] - first_event_time) + boot_time_nanos) / 1e9
            corrected_datetimes = pd.to_datetime(corrected_timestamps, unit='s')
            df = pd.DataFrame(gyro_data[["x", "y", "z"]].byteswap().newbyteorder())
            df['time'] = corrected_datetimes
            all_gyro_data.append(df)
        all_gyro_data = pd.concat(all_gyro_data)
        return all_acc_data, all_gyro_data
    except FileNotFoundError as e:
        print(f"The path {path} does not exist. Error: {e}")
        raise
    except ValueError as e:
        print(f"Data cannot be loaded properly. Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def save_data(data, processed_data_directory, subject_id, file_format='pkl'):
    """
    Save the processed_micromovements data to a file in the specified format.

    Args:
        data (pandas.DataFrame): The data to save.
        processed_data_directory (str): The root directory to save processed_micromovements data.
        subject_id (str): The subject identifier.
        file_format (str): The file format to save the data ('parquet', 'csv', 'pickle', etc.).
    """
    # Create a subdirectory for the subject if it doesn't exist
    subject_dir = os.path.join(processed_data_directory, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(subject_dir, f"{subject_id}.{file_format if file_format != 'pickle' else 'pkl'}")

    # Save the data in the specified format
    if file_format == 'parquet':
        data.to_parquet(file_path)
    elif file_format == 'csv':
        data.to_csv(file_path, index=False)
    elif file_format in ['pickle', 'pkl']:
        data.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def read_parquet(file_path):
    """
    Read a parquet file and return its contents as a pandas DataFrame.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        pandas.DataFrame: DataFrame containing data from the parquet file or None if an error occurs.

    Raises:
        IOError: If there is an error reading the file.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")
        return None


def check_already_processed(subject_id, processed_data_directory, file_format='parquet'):
    """
    Check if the sensor data for a given subject has already been processed_micromovements in the specified file format.

    Args:
        subject_id (str): The identifier for the subject.
        processed_data_directory (str): The directory path where processed_micromovements data is stored.
        file_format (str or tuple): The file format to check. Can be a single string or a tuple of strings representing multiple formats. Defaults to 'parquet'.

    Returns:
        bool: True if data has already been processed_micromovements in the specified file format, False otherwise.
    """
    if isinstance(file_format, str):
        file_formats = (file_format,)
    else:
        file_formats = file_format

    for fmt in file_formats:
        expected_file_path = os.path.join(processed_data_directory, subject_id, f'{subject_id}.{fmt}')
        if os.path.isfile(expected_file_path):
            return True
    return False
