import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from pathlib import Path
import re


class IMUDatasetAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.sensor_dtype = np.dtype([
            ("x", ">f4"),
            ("y", ">f4"),
            ("z", ">f4"),
            ("time", ">i8"),
        ])
        self.sampling_rate = 100  # Hz
        self.all_sessions = []
        self.results = {}

    def parse_subject_info(self, dir_name):
        """Extract subject ID and meal number from directory name"""
        parts = str(dir_name).split('_')
        subject_id = int(re.findall(r'\d+', parts[0])[0])
        meal_num = int(parts[-1]) if len(parts) > 2 else 1
        return subject_id, meal_num

    def load_raw_sensor_data(self, path):
        """Load and process sensor data from binary files"""
        acc_files = sorted([f for f in os.listdir(path) if f.endswith('.bin') and 'accelerometer' in f])
        gyro_files = sorted([f for f in os.listdir(path) if f.endswith('.bin') and 'gyroscope' in f])

        all_acc_data = []
        all_gyro_data = []

        # Load accelerometer data
        for file in acc_files:
            boot_time_nanos = int(file.split("_")[0]) * 1e6
            file_path = os.path.join(path, file)
            acc_data = np.fromfile(file_path, dtype=self.sensor_dtype)
            first_event_time = acc_data['time'][0]
            corrected_timestamps = ((acc_data['time'] - first_event_time) + boot_time_nanos) / 1e9
            corrected_datetimes = pd.to_datetime(corrected_timestamps, unit='s')
            df = pd.DataFrame(acc_data[["x", "y", "z"]].byteswap().newbyteorder())
            df['time'] = corrected_datetimes
            all_acc_data.append(df)

        # Load gyroscope data
        for file in gyro_files:
            boot_time_nanos = int(file.split("_")[0]) * 1e6
            file_path = os.path.join(path, file)
            gyro_data = np.fromfile(file_path, dtype=self.sensor_dtype)
            first_event_time = gyro_data['time'][0]
            corrected_timestamps = ((gyro_data['time'] - first_event_time) + boot_time_nanos) / 1e9
            corrected_datetimes = pd.to_datetime(corrected_timestamps, unit='s')
            df = pd.DataFrame(gyro_data[["x", "y", "z"]].byteswap().newbyteorder())
            df['time'] = corrected_datetimes
            all_gyro_data.append(df)

        return pd.concat(all_acc_data), pd.concat(all_gyro_data)

    def sync_sensors(self, acc_data, gyro_data):
        """Synchronize accelerometer and gyroscope data"""
        common_start_time = max(acc_data['time'].min(), gyro_data['time'].min())
        common_end_time = min(acc_data['time'].max(), gyro_data['time'].max())

        acc_synced = acc_data[(acc_data['time'] >= common_start_time) &
                              (acc_data['time'] <= common_end_time)]
        gyro_synced = gyro_data[(gyro_data['time'] >= common_start_time) &
                                (gyro_data['time'] <= common_end_time)]

        return acc_synced.reset_index(drop=True), gyro_synced.reset_index(drop=True)

    def calculate_signal_stats(self, data):
        """Calculate comprehensive signal statistics"""
        stats_dict = {
            'range': {
                'min': data[['x', 'y', 'z']].min().to_dict(),
                'max': data[['x', 'y', 'z']].max().to_dict()
            },
            'mean': data[['x', 'y', 'z']].mean().to_dict(),
            'std': data[['x', 'y', 'z']].std().to_dict(),
            'skewness': data[['x', 'y', 'z']].apply(stats.skew).to_dict(),
            'kurtosis': data[['x', 'y', 'z']].apply(stats.kurtosis).to_dict(),
            'snr': self.calculate_snr(data[['x', 'y', 'z']].values)
        }
        return stats_dict

    def calculate_snr(self, data):
        """Calculate Signal-to-Noise Ratio for each axis"""
        # Using Welch's method for better noise estimation
        f, Pxx = signal.welch(data, fs=self.sampling_rate, axis=0)
        signal_power = np.mean(Pxx, axis=0)
        noise_power = np.std(Pxx, axis=0)
        snr = 10 * np.log10(signal_power / noise_power)
        return {'x': snr[0], 'y': snr[1], 'z': snr[2]}

    def analyze_spectrum(self, data):
        """Perform spectral analysis"""
        freqs, Pxx = signal.welch(data[['x', 'y', 'z']].values,
                                  fs=self.sampling_rate,
                                  nperseg=1024)

        # Calculate power in different frequency bands
        low_mask = (freqs >= 0.1) & (freqs < 3)
        mid_mask = (freqs >= 3) & (freqs < 10)
        high_mask = freqs >= 10

        power_dist = {
            'low_freq': {
                'x': np.sum(Pxx[low_mask, 0]),
                'y': np.sum(Pxx[low_mask, 1]),
                'z': np.sum(Pxx[low_mask, 2])
            },
            'mid_freq': {
                'x': np.sum(Pxx[mid_mask, 0]),
                'y': np.sum(Pxx[mid_mask, 1]),
                'z': np.sum(Pxx[mid_mask, 2])
            },
            'high_freq': {
                'x': np.sum(Pxx[high_mask, 0]),
                'y': np.sum(Pxx[high_mask, 1]),
                'z': np.sum(Pxx[high_mask, 2])
            }
        }

        return freqs, Pxx, power_dist

    def calculate_sampling_stats(self, data):
        """Calculate sampling rate statistics"""
        time_diffs = np.diff(data['time'].astype(np.int64)) / 1e9
        return {
            'mean_interval': np.mean(time_diffs),
            'std_interval': np.std(time_diffs),
            'actual_rate': 1 / np.mean(time_diffs),
            'rate_stability': np.std(1 / time_diffs)
        }

    def plot_session_durations(self, save_path):
        """Plot distribution of session durations"""
        durations = [session['duration'] for session in self.all_sessions]
        plt.figure(figsize=(10, 6))
        sns.histplot(durations, bins=20)
        plt.title('Session Duration Distribution')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Count')
        plt.savefig(save_path / 'session_durations.png')
        plt.close()

    def plot_spectral_analysis(self, save_path):
        """Plot spectral analysis results"""
        # Implementation for spectral analysis plots
        for sensor_type in ['accelerometer', 'gyroscope']:
            power_dist = self.results[f'{sensor_type}_spectrum']['power_dist']

            plt.figure(figsize=(12, 6))
            x = np.arange(3)
            width = 0.25

            for i, axis in enumerate(['x', 'y', 'z']):
                powers = [power_dist[band][axis] for band in ['low_freq', 'mid_freq', 'high_freq']]
                plt.bar(x + i * width, powers, width, label=f'{axis}-axis')

            plt.title(f'{sensor_type.capitalize()} Spectral Power Distribution')
            plt.xlabel('Frequency Band')
            plt.ylabel('Power')
            plt.xticks(x + width, ['Low (0.1-3Hz)', 'Mid (3-10Hz)', 'High (>10Hz)'])
            plt.legend()
            plt.savefig(save_path / f'{sensor_type}_spectrum.png')
            plt.close()

    def analyze_dataset(self, output_path):
        """Perform complete dataset analysis"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        # Analyze each session
        for session_dir in self.base_path.glob("*"):
            if not session_dir.is_dir():
                continue

            subject_id, meal_num = self.parse_subject_info(session_dir.name)

            # Load and process data
            acc_data, gyro_data = self.load_raw_sensor_data(session_dir)
            acc_synced, gyro_synced = self.sync_sensors(acc_data, gyro_data)

            duration = (acc_synced['time'].max() - acc_synced['time'].min()).total_seconds() / 60

            session_info = {
                'subject_id': subject_id,
                'meal_num': meal_num,
                'duration': duration,
                'acc_data': acc_synced,
                'gyro_data': gyro_synced
            }

            self.all_sessions.append(session_info)

        # Calculate dataset-wide statistics
        self.results['general'] = {
            'total_sessions': len(self.all_sessions),
            'unique_subjects': len(set(s['subject_id'] for s in self.all_sessions)),
            'total_duration': sum(s['duration'] for s in self.all_sessions),
            'mean_duration': np.mean([s['duration'] for s in self.all_sessions]),
            'std_duration': np.std([s['duration'] for s in self.all_sessions])
        }

        # Combine all sensor data for overall analysis
        all_acc = pd.concat([s['acc_data'] for s in self.all_sessions])
        all_gyro = pd.concat([s['gyro_data'] for s in self.all_sessions])

        # Calculate comprehensive statistics
        self.results['accelerometer_stats'] = self.calculate_signal_stats(all_acc)
        self.results['gyroscope_stats'] = self.calculate_signal_stats(all_gyro)

        # Sampling rate analysis
        self.results['sampling_stats'] = {
            'accelerometer': self.calculate_sampling_stats(all_acc),
            'gyroscope': self.calculate_sampling_stats(all_gyro)
        }

        # Spectral analysis
        self.results['accelerometer_spectrum'] = {
            'freqs': None,
            'power': None,
            'power_dist': self.analyze_spectrum(all_acc)[2]
        }
        self.results['gyroscope_spectrum'] = {
            'freqs': None,
            'power': None,
            'power_dist': self.analyze_spectrum(all_gyro)[2]
        }

        # Generate plots
        self.plot_session_durations(output_path)
        self.plot_spectral_analysis(output_path)

        # Save results to JSON
        with open(output_path / 'analysis_results.json', 'w') as f:
            import json
            json.dump({k: v for k, v in self.results.items()
                       if not isinstance(v, pd.DataFrame)}, f, indent=4)

        return self.results


# Usage example
if __name__ == "__main__":
    analyzer = IMUDatasetAnalyzer("path/to/raw")
    results = analyzer.analyze_dataset("path/to/output")

    # Print summary
    print("\nDataset Summary:")
    print(f"Total Sessions: {results['general']['total_sessions']}")
    print(f"Unique Subjects: {results['general']['unique_subjects']}")
    print(f"Total Duration: {results['general']['total_duration']:.2f} minutes")
    print(f"Mean Session Duration: {results['general']['mean_duration']:.2f} minutes")

    # Print detailed statistics as needed...