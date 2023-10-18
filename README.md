# Sensor Dataset Preprocessing

A robust data preprocessing pipeline designed to convert raw accelerometer, gyroscope, and scale sensor data into a machine learning-ready dataset.

## Features

- **Data Reading**: Reads raw sensor data files (accelerometer and gyroscope) and scale data.
- **Eating Period Identification**: Identifies eating periods based on accelerometer, gyroscope, and scale data, discarding eating-irrelevant previous activity.
- **Average Sampling Interval Computation**: Calculates the average sampling interval for accelerometer and gyroscope data.
- **Data Interpolation**: Interpolates scale data based on given interval, one for accelerometer's and one for gyroscope's.
- **Data Saving**: Saves all processed data in both CSV and binary formats in a structured directory.
- **Folder Navigation**: Allows for folder selection through a GUI.
- **Data Check**: Skips processing for subjects whose data have already been processed.

## Directory Structure

## Directory Structure

- `dataset_folder/`
  - `raw/`
    - `subject_id/`
      - `accelerometer_data.bin` (can be multiple, they will be concatenated)
      - `gyroscope_data.bin` (can be multiple, they will be concatenated)
      - `scale_data_accelerometer.txt`
      - `scale_data_gyroscope.txt`



## Installation

Clone the repository in the dataset folder:

\```bash
git clone https://github.com/yiannislevy/Sensor-Dataset-Preprocessing.git 
\```

## Usage

1. Launch the script. A GUI window will appear to select the dataset folder.
2. The script will process each subject's data in the dataset, saving it in a structured directory under `processed/csv` and `processed/binary`.
3. If processed data for a subject already exists, the script will skip to the next subject.

Run the main pipeline script:

\```bash
python main_pipeline.py
\```

## Dependencies

- Python 3.x
- Pandas
- NumPy
- SciPy
- Tkinter
