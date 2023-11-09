import os
import pandas as pd
import numpy as np

path = "../../data/processed"
# make a list of the folders in that path unless it is a .DS_Store file
dataset = [f for f in os.listdir(path) if not f.startswith('.')]
dataset.sort(key=lambda x: int(x))
durations = []

for subject in dataset:
    data = pd.read_parquet(os.path.join(path, subject))
    duration = data['t'].iloc[-1] - data['t'].iloc[0]  # Calculate time between successive rows
    durations.append(duration.total_seconds())  # Extend, not append, to flatten the list

# Convert list of durations to a NumPy array for statistical calculations
durations = np.array(durations)

# Calculate various statistics
mean_duration = np.mean(durations)
median_duration = np.median(durations)
std_dev_duration = np.std(durations)
max_duration = np.max(durations)
min_duration = np.min(durations)
q75, q25 = np.percentile(durations, [75, 25])
iqr = q75 - q25
total_sec = np.sum(durations)
total_hours = total_sec / 3600

# Print out the statistics
print(f"Mean duration: {mean_duration} seconds")
print(f"Standard Deviation of duration: {std_dev_duration} seconds")
print(f"Median duration: {median_duration} seconds")
print(f"Min duration: {min_duration} seconds")
print(f"Max duration: {max_duration} seconds")
print(f"Interquartile Range: {iqr} seconds")
print(f"Total duration: {total_sec} seconds")
print(f"Total duration: {total_hours} hours")
