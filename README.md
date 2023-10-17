# Sensor Dataset Preprocessing
A robust data preprocessing pipeline designed to convert raw accelerometer, gyroscope, and scale sensor data into a machine learning-ready dataset.

## Features

- **Data Reading**: Reads raw sensor data files (accelerometer and gyroscope) and scale data.
- **Eating Period Identification**: Identifies eating periods based on accelerometer, gyroscope, and scale data.
- **Average Sampling Interval Computation**: Calculates the average sampling interval for accelerometer and gyroscope data.
- **Data Interpolation**: Interpolates scale data based on the minimum average interval from accelerometer and gyroscope data.
- **Data Saving**: Saves all processed data in a structured directory.

## Installation

Clone the repository:

\`\`\`bash
git clone https://github.com/yiannislevy/Sensor-Dataset-Preprocessing.git 
\`\`\`

## Usage

1. Place the raw sensor data and scale data in the designated folders.
2. Modify the `subject_path` and `output_path` variables in the main pipeline script to point to the raw data folder and the folder where you want the processed data saved.
3. Run the main pipeline script:

\`\`\`bash
python main_pipeline.py
\`\`\`

## Dependencies

- Python 3.x
- Pandas
- NumPy
- SciPy
