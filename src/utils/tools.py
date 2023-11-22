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


def convert_time_to_seconds(time_str):
    """
    Convert a time string in the format "HH:MM:SS.SSSS" to seconds.

    Parameters:
    - time_str (str): Time string to convert.

    Returns:
    - float: Time in seconds.
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
