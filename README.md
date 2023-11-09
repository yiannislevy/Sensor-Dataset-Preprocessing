# Sensor Dataset Preprocessing

This repository contains a comprehensive pipeline for preprocessing sensor and weight data, specifically designed for accelerometer and gyroscope binary data and weight from a scale under plate, for use in machine learning models.

## Features

- **Data Loading and Saving**: Efficiently reads and writes sensor data.
- **Preprocessing**: Includes resampling, median filtering, and gravity component removal from accelerometer data, standardization, optional mirroring for left handed subjects.
- **Inspection and Analysis**: Offers statistical analysis and evaluation tools for quality assurance.
- **Visualization**: Provides utilities for visualizing data trends and anomalies.
- **Variants**: Contains alternative methods for preprocessing steps, allowing for method comparison and selection.

## Directory Structure

- `config/`: Configuration files in JSON format.
- `data/`: Directory for raw and processed data storage.
  - `raw/`: Contains raw data files organized by subject.
    - `1/`, `2/`, `3/`, ...: Folders for each subject, named by a unique identifier.
      - `timestamp_accelerometer.bin`: Raw accelerometer data files, where 'timestamp' is the creation time in Unix time.
      - `timestamp_gyroscope.bin`: Raw gyroscope data files, where 'timestamp' is the creation time in Unix time.
      - `weight_time.txt`: Weight measurements in text format, where 'time' is in the format YYYYMMDD_HHMMSS.
  - `processed/`: Contains processed data files organized by subject.
    - `1/`, `2/`, `3/`, ...: Folders for each subject, named by a unique identifier.
      - `combined_data.parquet`: Processed data in Parquet format.
- `docs/`: Additional documentation, guidelines, and notes.
- `notebooks/`: Jupyter notebooks for demonstrations, experiments, and tutorials.
- `src/`: Source code directory for the dataset preprocessing pipeline.
  - `analysis/`: Scripts for data analysis and generating statistics about subjects and the dataset.
    - `__init__.py`
    - `data_inspection.py`
    - `dataset_statistics.py`
  - `main/`: Main scripts for dataset creation and preprocessing.
    - `__init__.py`
    - `data_io.py`
    - `data_preprocessing.py`
  - `utils/`: Utility scripts with helper functions.
    - `__init__.py`
    - `data_visualization.py`
    - `method_variants.py`
- `.gitignore`: File specifying untracked files that Git should ignore.
- `LICENSE`: The MIT License detailing the terms under which the software is provided.
- `README.md`: This file, providing an overview and documentation for the project.
- `requirements.txt`: List of Python dependencies required to run the project.
- `run_pipeline.py`: Master script to initiate and run the preprocessing pipeline.
- `setup.py`: Script for setting up the project environment.

## Installation

Clone the repository:

```bash
git clone https://github.com/yiannislevy/Sensor-Dataset-Preprocessing.git
cd Sensor-Dataset-Preprocessing
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the preprocessing pipeline, execute the following command:

```bash
python run_pipeline.py
```
The pipeline will process data in the data/raw directory and output the results to data/processed.

## Data Folder Structure

The `data` directory should be structured as follows:

- Each subject's data is contained in a separate folder named with a unique ID.
- The subject folder contains raw binary files for accelerometer and gyroscope data. There may be multiple binary files for each sensor, which will be concatenated during processing.
- A text file containing weight data is also included in the subject's folder.
- Processed data will be saved in the `data/processed` directory, within a folder corresponding to the subject's ID, in a Parquet file that contains all processed data for that subject.

## Configuration

Adjust the configuration file `config/config.json` to specify the following parameters:

- `data_dir`: The directory containing the raw data.
- `processed_dir`: The directory to save the processed data.
- `upsample_frequency`: The frequency to upsample the data to, in Hz.
- `median_filter_order`: The order of the median filter to apply to the data.
- `gravity_filter_cutoff_hz`: The cutoff frequency for the low-pass filter used.
- `left_handed_subjects`: A list of subject IDs that are left handed.

## Documentation

Documentation for the project can be found in the `docs` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
