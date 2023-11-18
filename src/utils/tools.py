def calculate_frequency(data):
    """
    Calculate the sampling frequency of sensor data based on the time differences between data points.

    Args:
        data (pandas.DataFrame): Dataframe containing sensor data with a 'time' column.

    Returns:
        float: The calculated average sampling frequency in Hertz (Hz).
    """
    try:
        time_diffs = data['time'].diff().dropna()
        time_diffs_ms = time_diffs.dt.total_seconds() * 1000
        avg_interval = time_diffs_ms.mean()
        return 1000 / avg_interval
    except Exception as e:
        print(f"An error occurred while calculating the frequency: {e}")
        raise
