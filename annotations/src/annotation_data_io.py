import os
import pickle


def load_bite_events(file_path):
    """
    Load bite events from a pickle file.

    Parameters:
    - file_path: str, path to the .pkl file containing the bite events array.

    Returns:
    - numpy.ndarray, array of bite events with start and end times.
    """
    with open(file_path, 'rb') as file:
        bite_events_array = pickle.load(file)
    return bite_events_array


def save_data(data, processed_data_directory, filename):
    """
    Save data to a pickle file.

    Args:
        data: Data to be saved, can be of any type.
        processed_data_directory (str): Directory to save the data.
        filename (str): Name of the file to save.
    """
    # Create the directory for the file if it doesn't exist
    file_dir = os.path.join(processed_data_directory)
    os.makedirs(file_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(file_dir, f"{filename}.pkl")

    # Save the data in pickle format
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
