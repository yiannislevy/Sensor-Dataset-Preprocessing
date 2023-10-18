import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from tkinter import filedialog
from tkinter import Tk
import pickle

# 1. Read IMU Data
def read_sensor_data(path, sensor_type):
    files = [f for f in os.listdir(path) if f.endswith('.bin') and sensor_type in f]
    files = sorted(files, key=lambda x: int(x.split("_")[0]))
    BEtype = np.dtype([
        ("x", ">f"),
        ("y", ">f"),
        ("z", ">f"),
        ("time", ">i8"),
    ])
    all_data = []
    for file in files:
        boot_time_nanos = int(file.split("_")[0]) * 1e6
        file_path = os.path.join(path, file)
        data = np.fromfile(file_path, dtype=BEtype)
        first_event_time = data['time'][0]
        corrected_timestamps = ((data['time'] - first_event_time) + boot_time_nanos) / 1e9
        corrected_datetimes = pd.to_datetime(corrected_timestamps, unit='s')
        df = pd.DataFrame(data[["x", "y", "z"]].byteswap().newbyteorder())
        df['time'] = corrected_datetimes
        # df = df.set_index('time')
        all_data.append(df)
    return pd.concat(all_data), corrected_datetimes[-1]


# 2. Read Scale Data
def read_scale_data(path, stop_time):
    files = [f for f in os.listdir(path) if f.startswith('weights_') and f.endswith('.txt')]
    files = sorted(files, key=lambda x: datetime.strptime(x.split('_')[2].replace('.txt', ''), '%H%M%S'))
    all_data = []
    for file in files:
        file_path = os.path.join(path, file)
        weights = pd.read_csv(file_path, header=None, names=['weight'])
        start_time = stop_time - timedelta(seconds=len(weights)-1)  # calculate start time based on stop time and number of weight data points
        time = pd.date_range(start=start_time, end=stop_time, periods=len(weights))  # create date range
        weights['time'] = time
        weights = weights.set_index('time')
        all_data.append(weights)
    return pd.concat(all_data)


# 3. Identify Eating Period
def identify_eating_period(imu_data, scale_data, scale_end_time):
    # Ensure time columns align
    imu_data = imu_data.rename(columns={'time': 'Timestamp'})

    # Extract time and weight from scale_data
    scale_data_with_time = scale_data.reset_index().rename(columns={'time': 'Timestamp'})

    # Calculate time offset
    imu_end_time = imu_data['Timestamp'].iloc[-1]
    time_offset = imu_end_time - scale_end_time

    # Calculate eating period in IMU time
    eating_start_time_imu = scale_data_with_time['Timestamp'].iloc[0] + time_offset
    eating_end_time_imu = imu_end_time

    # Filter IMU data for eating period
    eating_period_imu = imu_data[
        (imu_data['Timestamp'] >= eating_start_time_imu) &
        (imu_data['Timestamp'] <= eating_end_time_imu)
    ]

    return eating_period_imu


# 4. Compute Average Sampling Interval
def compute_avg_interval(eating_period_imu):
    # Calculate the time differences between consecutive samples
    time_diffs = eating_period_imu['Timestamp'].diff().dropna()

    # Convert time differences to milliseconds
    time_diffs_ms = time_diffs.dt.total_seconds() * 1000

    # Compute the average sampling interval
    avg_interval = time_diffs_ms.mean()

    return avg_interval

# 4.5 Downsample IMU Data with highest sampling rate to the average sampling interval of the lowest sampling rate
# def resample_imu_data(imu_data, target_avg_interval):
#     imu_data.set_index('Timestamp', inplace=True)
#     imu_data_resampled = imu_data.resample(f"{int(target_avg_interval)}ms").mean()
#     # imu_data_resampled.interpolate(inplace=True)  # Interpolate missing values
#     imu_data_resampled.dropna(inplace=True)  # Drop rows with NaN values, which may include NaT in the Timestamp
#     imu_data_resampled.reset_index(inplace=True)
#     return imu_data_resampled

# 5. Interpolate Scale Data
def interpolate_scale_data(scale_data, avg_interval):
    scale_data_with_time = scale_data.reset_index().rename(columns={'time': 'Timestamp'})
    original_time_vector_unix = scale_data_with_time['Timestamp'].astype('datetime64[s]').view('int64') // 1e9

    # Sort the time vector
    sorted_original_time_vector_unix = original_time_vector_unix.sort_values()

    scale_data_values = scale_data_with_time['weight'].values

    # Use sorted time vector for start and end times
    start_time = sorted_original_time_vector_unix.iloc[0]
    end_time = sorted_original_time_vector_unix.iloc[-1]

    new_time_vector = np.arange(start_time, end_time, avg_interval / 1000)
    new_time_vector_datetime = pd.to_datetime(new_time_vector, unit='s')

    interpolating_function = interp1d(sorted_original_time_vector_unix, scale_data_values, kind='linear', fill_value='extrapolate')
    interpolated_values = interpolating_function(new_time_vector)

    interpolated_scale_data = pd.DataFrame({'Timestamp': new_time_vector_datetime, 'Interpolated_Weight': interpolated_values})

    return interpolated_scale_data

# 6. Save Processed Data
def save_processed_data(output_path, subject_id, eating_period_acc, eating_period_gyro, interpolated_scale_data_acc, interpolated_scale_data_gyro):
    csv_dir = os.path.join(output_path, 'csv', subject_id)
    binary_dir = os.path.join(output_path, 'binary', subject_id)

    for directory in [csv_dir, binary_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save as CSV
    eating_period_acc.to_csv(os.path.join(csv_dir, 'accelerometer_processed.csv'))
    eating_period_gyro.to_csv(os.path.join(csv_dir, 'gyroscope_processed.csv'))
    interpolated_scale_data_acc.to_csv(os.path.join(csv_dir, 'scale_processed_accelerometer.csv'))
    interpolated_scale_data_gyro.to_csv(os.path.join(csv_dir, 'scale_processed_gyroscope.csv'))

    custom_dtype = np.dtype([
        ("x", ">f"),
        ("y", ">f"),
        ("z", ">f"),
        ("time", ">i8"),
    ])

    for df, filename in zip([eating_period_acc, eating_period_gyro], ['accelerometer_processed.bin', 'gyroscope_processed.bin']):
        array_to_save = np.array(
            [(row.x, row.y, row.z, row.Timestamp.timestamp()) for index, row in df.iterrows()],
            dtype=custom_dtype)
        array_to_save.tofile(os.path.join(binary_dir, filename))

    scale_array_to_save_acc = np.array(
        [(row.Interpolated_Weight, row.Timestamp.timestamp()) for index, row in interpolated_scale_data_acc.iterrows()],
        dtype=[("grams", ">f"), ("time", ">i8")])
    scale_array_to_save_acc.tofile(os.path.join(binary_dir, 'scale_processed_accelerometer.bin'))

    scale_array_to_save_gyro = np.array(
        [(row.Interpolated_Weight, row.Timestamp.timestamp()) for index, row in interpolated_scale_data_gyro.iterrows()],
        dtype=[("grams", ">f"), ("time", ">i8")])
    scale_array_to_save_gyro.tofile(os.path.join(binary_dir, 'scale_processed_gyroscope.bin'))



# Open folder dialog for selecting dataset folder
root = Tk()
root.withdraw()
dataset_path = filedialog.askdirectory(title="Select the dataset folder")

# Loop through each subject's folder
subject_dirs = [str(d) for d in os.listdir(dataset_path + '/raw') if os.path.isdir(os.path.join(dataset_path + '/raw', d))]
for subject_id in subject_dirs:

    output_path_csv = os.path.join(dataset_path, 'processed', 'csv', subject_id)
    output_path_binary = os.path.join(dataset_path, 'processed', 'binary', subject_id)

    # Check if processed files already exist
    acc_csv_file = os.path.join(output_path_csv, 'accelerometer_processed.csv')
    gyro_csv_file = os.path.join(output_path_csv, 'gyroscope_processed.csv')
    scale_csv_file = os.path.join(output_path_csv, 'scale_processed.csv')

    acc_binary_file = os.path.join(output_path_binary, 'accelerometer_processed.bin')
    gyro_binary_file = os.path.join(output_path_binary, 'gyroscope_processed.bin')
    scale_binary_file = os.path.join(output_path_binary, 'scale_processed.bin')

    if all(os.path.exists(file) for file in
           [acc_csv_file, gyro_csv_file, scale_csv_file, acc_binary_file, gyro_binary_file, scale_binary_file]):
        print(f"Processed data for subject {subject_id} already exists. Skipping.")
        continue

    subject_path = os.path.join(dataset_path, 'raw', subject_id)
    output_path = os.path.join(dataset_path, 'processed')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Step 1: Read IMU Data
    accelerometer_data, _ = read_sensor_data(subject_path, 'accelerometer')
    gyroscope_data, imu_end_time = read_sensor_data(subject_path, 'gyroscope')

    # Step 2: Read Scale Data
    scale_data = read_scale_data(subject_path, imu_end_time)

    # Step 3: Identify Eating Period
    eating_period_acc = identify_eating_period(accelerometer_data, scale_data, imu_end_time)
    eating_period_gyro = identify_eating_period(gyroscope_data, scale_data, imu_end_time)

    # Step 4: Compute Average Sampling Interval
    avg_interval_acc = compute_avg_interval(eating_period_acc)
    avg_interval_gyro = compute_avg_interval(eating_period_gyro)

    # # New Step: Resample the IMU data with the higher sampling rate to match the other
    # if avg_interval_acc < avg_interval_gyro:
    #     # Resample accelerometer data to match gyroscope interval
    #     eating_period_acc = resample_imu_data(eating_period_acc, avg_interval_gyro)
    # else:
    #     # Resample gyroscope data to match accelerometer interval
    #     eating_period_gyro = resample_imu_data(eating_period_gyro, avg_interval_acc)

    # Step 5: Interpolate Scale Data (twice, once for each IMU sensor)
    interpolated_scale_data_acc = interpolate_scale_data(scale_data, avg_interval_acc)
    interpolated_scale_data_gyro = interpolate_scale_data(scale_data, avg_interval_gyro)

    min_avg_interval = max(avg_interval_acc, avg_interval_gyro)
    interpolated_scale_data = interpolate_scale_data(scale_data, min_avg_interval)

    # Step 6: Save Processed Data
    save_processed_data(output_path, subject_id, eating_period_acc, eating_period_gyro, interpolated_scale_data_acc,
                        interpolated_scale_data_gyro)
