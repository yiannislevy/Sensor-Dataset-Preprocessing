# run_pipeline.py

import os
import json
from pathlib import Path

from src.main.imu_data_io import load_raw_sensor_data, save_data, check_already_processed
from src.main.imu_preprocessing import resample, median_filter, remove_gravity, mirror_left_to_right, standardize_data, \
    combine_sensor_data

# TODO add mandometer in the pipeline
# TODO add documentation
# TODO modify README
# TODO modify requirements.txt
# TODO add tests
# TODO remove unnecessary files like align_data or start_end, pipe_test etc [notebooks]
def main():
    # Load configuration
    with open('config/imu_config.json') as config_file:
        config = json.load(config_file)

    raw_data_directory = config['data_paths']['raw_data_directory']
    processed_data_directory = config['data_paths']['processed_data_directory']
    upsample_frequency = config['resampling']['upsample_frequency']
    median_filter_order = config['filtering']['median_filter_order']
    gravity_filter_cutoff_hz = config['filtering']['gravity_filter_cutoff_hz']
    left_handed_subjects = config['processing_options']['left_handed_subjects']
    saving_format = config['saving_options']['file_format']

    # Ensure processed data directory exists
    Path(processed_data_directory).mkdir(parents=True, exist_ok=True)

    # Iterate over subject directories in the raw data directory
    for subject_path in Path(raw_data_directory).iterdir():
        if subject_path.is_dir() and not subject_path.name.startswith('.'):
            subject_id = subject_path.name

            # Skip if the subject's directory name contains '**'
            if '**' in subject_id:
                print(f"Skipping subject {subject_id} because it contains '**'")
                continue

            # Construct the path to the subject's data
            subject_data_path = os.path.join(raw_data_directory, subject_id)

            # Check if the subject's data has already been processed
            if check_already_processed(subject_id, processed_data_directory, saving_format):
                print(f"Subject {subject_id} has already been processed with this file format. Skipping.")
                continue

            try:
                # Load raw sensor data
                acc_data, gyro_data = load_raw_sensor_data(subject_data_path)

                # Resample the data
                acc_data, gyro_data = resample(acc_data, gyro_data, upsample_frequency)

                # Apply median filter
                acc_data, gyro_data = median_filter(acc_data, gyro_data, median_filter_order)

                # Remove earth's gravity from accelerometer data
                acc_data = remove_gravity(acc_data, gravity_filter_cutoff_hz, upsample_frequency)

                # If subject is left-handed, mirror the data
                if int(subject_id) in left_handed_subjects:
                    acc_data, gyro_data = mirror_left_to_right(acc_data, gyro_data)

                # Standardize
                acc_data, gyro_data = standardize_data(acc_data, gyro_data)

                # Combine accelerometer and gyroscope data
                combined_data = combine_sensor_data(acc_data, gyro_data)

                # Save the processed data
                save_data(combined_data, processed_data_directory, subject_id, saving_format)

                print(f"Processing complete for subject {subject_id}")

            except Exception as e:
                print(f"An error occurred while processing subject {subject_id}: {e}")
                continue


if __name__ == "__main__":
    main()
