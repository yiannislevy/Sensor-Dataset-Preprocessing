# run_pipeline.py

import os
import json
from pathlib import Path

from src.main.imu_data_io import load_raw_sensor_data, save_data, check_already_processed
from src.main.imu_preprocessing import (sync, resample, remove_gravity, median_filter, mirror_left_to_right,
                                        align_old_msft_watch, standardize_data, transform_data, combine_sensor_data)


# TODO add mandometer in the pipeline
# TODO add documentation
# TODO modify README
# TODO remove unnecessary files like align_data or start_end, pipe_test etc [notebooks]

# NOTE : PAPER IN QUESTION --> Modeling Wrist Micromovements
def main():
    # Load configuration
    with open('config/imu_config.json') as config_file:
        config = json.load(config_file)

    raw_data_directory = config['data_paths']['raw_data_directory']
    processed_data_directory = config['data_paths']['processed_data_directory']
    upsample_frequency = config['resampling']['upsample_frequency']
    filter_length = config['filtering']['moving_average_filter_length']
    gravity_filter_cutoff_hz = config['filtering']['gravity_filter_cutoff_hz']
    left_handed_subjects = config['processing_options']['left_handed_subjects']
    saving_format = config['saving_options']['file_format']
    path_to_mean_std = config['standardization']['path_to_mean_std']

    # Ensure processed_micromovements data directory exists
    Path(processed_data_directory).mkdir(parents=True, exist_ok=True)

    # Iterate over subject directories in the raw data directory
    for subject_path in Path(raw_data_directory).iterdir():
        if subject_path.is_dir() and not subject_path.name.startswith('.'):
            subject_id = subject_path.name

            # Skip if the subject's directory name contains '**'
            if '**' in subject_id:
                print(f"Skipping subject {subject_id} because it contains '**'")
                continue

            meal_folders = [f for f in subject_path.iterdir() if f.is_dir() and f.name.startswith('meal_')]

            # If meal_x folders are found, process each one
            if meal_folders:
                for meal_folder in meal_folders:
                    process_session(meal_folder, subject_id + '_' + meal_folder.name, processed_data_directory, upsample_frequency, filter_length, gravity_filter_cutoff_hz, left_handed_subjects, saving_format, path_to_mean_std)
            else:
                # No meal_x folders found, process the subject's main directory
                process_session(subject_path, subject_id, processed_data_directory, upsample_frequency, filter_length, gravity_filter_cutoff_hz, left_handed_subjects, saving_format, path_to_mean_std)


def process_session(folder_path, identifier, processed_data_directory, upsample_frequency, filter_length, gravity_filter_cutoff_hz, left_handed_subjects, saving_format, path_to_mean_std):
    if check_already_processed(identifier, processed_data_directory, saving_format):
        print(f"{identifier} has already been processed. Skipping.")
        return

    try:
        acc_data, gyro_data = load_raw_sensor_data(folder_path)

        acc_data, gyro_data = sync(acc_data, gyro_data)

        acc_data, gyro_data = resample(acc_data, gyro_data, upsample_frequency)

        acc_data = remove_gravity(acc_data, upsample_frequency, gravity_filter_cutoff_hz)

        acc_data, gyro_data = median_filter(acc_data, gyro_data, filter_length)

        if int(identifier.split('_')[0]) in left_handed_subjects:
            acc_data, gyro_data = mirror_left_to_right(acc_data, gyro_data)

        acc_data, gyro_data = align_old_msft_watch(acc_data, gyro_data)

        acc_data, gyro_data = transform_data(acc_data, gyro_data)

        acc_data, gyro_data = standardize_data(acc_data, gyro_data, path_to_mean_std)

        combined_data = combine_sensor_data(acc_data, gyro_data)

        save_data(combined_data, processed_data_directory, identifier, saving_format)

        print(f"Processing complete for {identifier}")

    except Exception as e:
        print(f"An error occurred while processing {identifier}: {e}")


if __name__ == "__main__":
    main()
