import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_eating_style(intervals, subject_id, window_size=3):
    """
    Plot bite intervals, their moving average, and a linear regression line.

    Parameters:
    - intervals: numpy.ndarray, array of intervals between bites.
    - window_size: int, size of the window for calculating the moving average.
    """
    # Calculate moving average
    moving_avg = np.convolve(intervals, np.ones(window_size) / window_size, mode='valid')

    # Calculate linear regression
    x = np.arange(len(intervals))
    slope, intercept, _, _, _ = linregress(x, intervals)
    lin_reg_line = intercept + slope * x

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(intervals, label='Bite Intervals', marker='o', linestyle='-', alpha=0.5)
    plt.plot(np.arange(window_size - 1, len(moving_avg) + window_size - 1), moving_avg, label='Moving Average', color='red', linewidth=2)
    plt.plot(x, lin_reg_line, 'g--', label='Linear Regression', linewidth=2)

    # Enhancing the plot
    plt.title(f'Analysis of Eating Behavior for Subject {subject_id}', fontsize=16)
    plt.xlabel('Bite Number', fontsize=14)
    plt.ylabel('Interval Between Bites (seconds)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'eating_style_{subject_id}.png')
    plt.show()
