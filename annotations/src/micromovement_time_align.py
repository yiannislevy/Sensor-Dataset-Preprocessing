import os
import json
from datetime import datetime, timedelta

# TODO test this script


def convert_to_datetime(time_str):
    return datetime.strptime(time_str, '%H:%M:%S.%f')


def align_to_imu_time(relative_time_str, imu_clap_time, video_clap_time, video_start_time):
    relative_time = datetime.strptime(relative_time_str, '%H:%M:%S.%f') - datetime(1900, 1, 1)
    absolute_video_time = video_start_time + relative_time
    aligned_time = absolute_video_time - (video_clap_time - imu_clap_time)
    return aligned_time.strftime('%H:%M:%S.%f')[:-3]


def process_subject_file(subject_id, subject_sync_info):
    filename = f'/manual_micromovements/subject_{subject_id}.json'
    aligned_filename = f'/processed_micromovements/subject_{subject_id}.json'

    with open(filename, 'r') as file:
        data = json.load(file)

    # Extract times from subject_sync_info
    video_start_time = convert_to_datetime(subject_sync_info["video_start_time"])
    imu_clap_time = convert_to_datetime(subject_sync_info["imu_clap_time"])
    video_clap_time = video_start_time + timedelta(seconds=float(subject_sync_info["relative_video_clap_time"].split(":")[-1]))

    # Align timestamps
    for meal in data.get('meals', []):
        for bite in meal.get('bites', []):
            for key in ['pick_food_t_relative', 'upwards_t_relative']:
                if key in bite:
                    bite[key]['start'] = align_to_imu_time(bite[key]['start'], imu_clap_time, video_clap_time, video_start_time)
                    bite[key]['end'] = align_to_imu_time(bite[key]['end'], imu_clap_time, video_clap_time, video_start_time)

    # Save the aligned data
    with open(aligned_filename, 'w') as file:
        json.dump(data, file, indent=4)


# Load the synchronization information
sync_filename = '../time_sync.json'
with open(sync_filename, 'r') as file:
    sync_data = json.load(file)

os.makedirs('../processed_micromovements', exist_ok=True)

# Process each subject file
for subject_sync_info in sync_data["sync_info"]:
    subject_id = subject_sync_info["subject_id"]
    process_subject_file(subject_id, subject_sync_info)
