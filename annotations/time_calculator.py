import json
from datetime import datetime, timedelta


# Helper function to convert string time format to datetime object
def convert_to_datetime(time_str):
    return datetime.strptime(time_str, '%H:%M:%S.%f')


def calculate_mando_start_time(sync_info):
    # Convert IMU clap time to datetime object
    imu_clap_time = convert_to_datetime(sync_info["imu_clap_time"])

    # Calculate time difference between IMU clap and Mandometer event in video
    video_clap_time = convert_to_datetime(sync_info["relative_video_clap_time"])
    video_mando_event = convert_to_datetime(sync_info["relative_video_mando_time"])
    video_clap_mando_diff = video_mando_event - video_clap_time

    # Parse the relative mando time as a duration
    hours, minutes, seconds_milliseconds = sync_info["relative_mando_time"].split(':')
    seconds, milliseconds = seconds_milliseconds.split('.')
    hours, minutes, seconds, milliseconds = int(hours), int(minutes), int(seconds), int(milliseconds)
    relative_mando_duration = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    # Adjusted Mandometer start time
    mando_start_time = imu_clap_time + video_clap_mando_diff - relative_mando_duration
    return mando_start_time.strftime('%H:%M:%S.%f')[:-2]


def calculate_video_imu_time_difference(sync_info):
    # Convert video start time to datetime object
    video_start_time = convert_to_datetime(sync_info["video_start_time"])

    # Convert relative video clap time to a timedelta
    relative_video_clap_time = convert_to_datetime(sync_info["relative_video_clap_time"]) - datetime(1900, 1, 1)

    # Calculate absolute video clap time
    absolute_video_clap_time = video_start_time + relative_video_clap_time

    # Convert IMU clap time to datetime object
    imu_clap_time = convert_to_datetime(sync_info["imu_clap_time"])

    # Calculate the time difference
    time_difference = abs(imu_clap_time - absolute_video_clap_time)
    return time_difference


# Load JSON data
path = "time_sync.json"
with open(path, 'r') as file:
    data = json.load(file)

# Process each subject using the simplified approach
for subject in data["sync_info"]:
    mando_real_start_time = calculate_mando_start_time(subject)
    subject["mando_real_start_time"] = mando_real_start_time
    dt = calculate_video_imu_time_difference(subject)
    subject["video_imu_time_difference"] = str(dt)[:-2]

# Save the updated data back to the JSON file
with open(path, 'w') as file:
    json.dump(data, file, indent=4)
