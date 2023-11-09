import os
import pandas as pd
import numpy as np

path = "../../data/processed"
dataset = [f for f in os.listdir(path) if not f.startswith('.')]
dataset.sort(key=lambda x: int(x))
durations = []
for subjects in dataset:
    data = pd.read_parquet(os.path.join(path, subjects))
    duration = data['t'].iloc[-1] - data['t'].iloc[0]
    durations.append(duration.total_seconds())

mean_durations = np.mean(durations)


print(f"The durations of the subjects are (in seconds): {durations}")
print(f"Mean duration is {mean_durations} seconds!")
print(f"Minimum duration is {np.min(durations)} seconds!")
print(f"Maximum duration is {np.max(durations)} seconds!")
