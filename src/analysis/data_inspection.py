import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


def frequency_analysis(original_data, interpolated_data, sampling_rate):
    """
    Perform and plot the Fast Fourier Transform (FFT) of original and interpolated data.

    This function computes the FFT of two sets of signal data, the original and the interpolated,
    to analyze their frequency components. The result is plotted to compare the frequency spectra
    of both signals.

    Args:
        original_data (pandas.DataFrame): DataFrame containing the original signal data with 'x' column.
        interpolated_data (pandas.DataFrame): DataFrame containing the interpolated signal data with 'x' column.
        sampling_rate (int): The sampling rate of the signals in Hertz.

    No return value or exceptions raised; this function outputs a plot directly.
    """
    # Convert Pandas Series to NumPy array
    x_original = original_data['x'].to_numpy()
    x_interpolated = interpolated_data['x'].to_numpy()

    # Original signal FFT
    yf_original = fft(x_original)
    xf_original = fftfreq(len(x_original), 1 / sampling_rate)

    # Interpolated signal FFT
    yf_interpolated = fft(x_interpolated)
    xf_interpolated = fftfreq(len(x_interpolated), 1 / sampling_rate)

    plt.figure(figsize=(14, 5))
    plt.plot(xf_original, 2.0 / len(x_original) * abs(yf_original), label='Original FFT')
    plt.plot(xf_interpolated, 2.0 / len(x_interpolated) * abs(yf_interpolated), label='Interpolated FFT', alpha=0.7)
    plt.title('FFT of signals')
    plt.legend()
    plt.show()


def median_filters_grid_search(data, filter_orders):
    """
    Conduct a grid search to determine the optimal median filter window size for signal data.

    This function iterates over a range of filter window sizes, applies a median filter to the
    signal data, and calculates the Signal-to-Noise Ratio (SNR) improvement and correlation of the
    filtered signal to the original. The results are compiled for subsequent analysis.

    Args:
        data (pandas.DataFrame): DataFrame containing signal data with 'x', 'y', and 'z' columns.
        filter_orders (list of int): List of median filter window sizes to test.

    Returns:
        dict: A dictionary with keys 'order', 'filtered_data', 'snr', and 'correlation', containing
        the median filter window sizes, filtered data, SNR improvements, and correlations for each filter order.

    The function assumes the input data is pre-processed_micromovements and does not handle exceptions arising from invalid input.
    """
    # Dictionary to store the filtered data and metrics
    results = {
        'order': [],
        'filtered_data': [],
        'snr': [],
        'correlation': []
    }

    # Signal variance for SNR calculation
    original_variance = data.var()

    for order in filter_orders:
        # Apply median filter
        filtered_x = medfilt(data['x'], kernel_size=order)
        filtered_y = medfilt(data['y'], kernel_size=order)
        filtered_z = medfilt(data['z'], kernel_size=order)

        # Calculate SNR: ratio of variance reduction
        noise_variance = np.var(data['x'] - filtered_x)
        snr_improvement = 10 * np.log10(original_variance / noise_variance) if noise_variance > 0 else float('inf')

        # Calculate correlation with original data
        correlation_x = np.corrcoef(data['x'], filtered_x)[0, 1]

        # Store results
        results['order'].append(order)
        results['filtered_data'].append((filtered_x, filtered_y, filtered_z))
        results['snr'].append(snr_improvement)
        results['correlation'].append(correlation_x)

    return results


def evaluate_resampling(original_data, resampled_data):
    """
    Evaluate the quality of resampling by comparing statistical metrics and plotting the original and resampled data.

    This function interpolates the resampled data to the original timestamps, compares statistical
    metrics such as mean and standard deviation, computes error metrics like Mean Squared Error (MSE)
    and Mean Absolute Error (MAE), and visually inspects the overlap of the original and resampled signals
    through plots.

    Args:
        original_data (pandas.DataFrame): DataFrame containing the original sensor data with 'time' and 'x', 'y', 'z' columns.
        resampled_data (pandas.DataFrame): DataFrame containing the resampled sensor data with 'time' and 'x', 'y', 'z' columns.

    Returns:
        tuple: Contains a DataFrame with statistical comparison metrics, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

    The function assumes the input data is structured properly and does not handle exceptions for invalid data formats or interpolation errors.
    """
    # Interpolate resampled data back to original timestamps for comparison
    resampled_interpolated = resampled_data.set_index('time').reindex(
        original_data.set_index('time').index).interpolate(method='time').reset_index()

    # Statistical comparison
    stats_comparison = pd.DataFrame({
        'Original_Mean': original_data[['x', 'y', 'z']].mean(),
        'Resampled_Mean': resampled_interpolated[['x', 'y', 'z']].mean(),
        'Original_Std': original_data[['x', 'y', 'z']].std(),
        'Resampled_Std': resampled_interpolated[['x', 'y', 'z']].std()
    })

    # Error analysis
    mse = ((original_data[['x', 'y', 'z']] - resampled_interpolated[['x', 'y', 'z']]) ** 2).mean()
    mae = (original_data[['x', 'y', 'z']] - resampled_interpolated[['x', 'y', 'z']].abs()).mean()

    # Visual inspection
    plt.figure(figsize=(14, 4))
    for i, axis in enumerate(['x', 'y', 'z'], 1):
        plt.subplot(1, 3, i)
        original_data[axis].plot(label='Original', alpha=0.5)
        resampled_interpolated[axis].plot(label='Resampled', alpha=0.5)
        plt.title(f'Axis {axis.upper()}')
        plt.legend()
    plt.tight_layout()
    plt.show()

    return stats_comparison, mse, mae
