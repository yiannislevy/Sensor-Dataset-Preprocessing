import pandas as pd
import os


def load_raw_mando_data(path):
    """
        Load raw weight data from a specified directory.

        This function searches for a text file in the provided directory that starts with 'weights_' and ends with '.txt',
        which contains weight measurements. It reads the first such file it finds. If no such file is present, it raises an
        error. The function assumes that the weight data is stored in a single-column CSV format without a header.

        Args:
            path (str): The file path to the directory where the weight data files are stored.

        Returns:
            pandas.Series: A pandas Series containing the weights loaded from the file.

        Raises:
            ValueError: If no weight files are found in the specified directory.

        The function uses 'os.listdir' to find files in the given directory and 'pandas.read_csv' to load the data.
    """
    # Find the weight file in the directory
    files = [f for f in os.listdir(path) if f.startswith('weights_') and f.endswith('.txt')]
    if not files:
        raise ValueError(f"No weight files found in directory: {path}")

    # Load the data
    file_path = os.path.join(path, files[0])
    weights = pd.read_csv(file_path, header=None, names=['weight'])
    return weights['weight']
