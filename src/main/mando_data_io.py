import pandas as pd
import numpy as np
import os

# TODO cleanup

def load_raw_mando_data(path):
    # TODO docstring
    # Find the weight file in the directory
    files = [f for f in os.listdir(path) if f.startswith('weights_') and f.endswith('.txt')]
    if not files:
        raise ValueError(f"No weight files found in directory: {path}")

    # Load the data
    file_path = os.path.join(path, files[0])
    weights = pd.read_csv(file_path, header=None, names=['weight'])
    return weights['weight']
