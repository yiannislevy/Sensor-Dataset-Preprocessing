import json
import numpy as np
import pickle

# Path to the annotation file
file_path = '../processed_micromovements/bites/bite_events_19.txt'

subject_id = 19

data_structure_json = {
    "subject_id": subject_id,
    "bites": []
}

temp_data_structure = []

# Read from the file
with open(file_path, 'r') as file:
    for line in file:
        # Split the line into components
        parts = line.strip().split('\t')

        # Extract the relevant information
        _, _, _, start_seconds, _, end_seconds, _, duration_seconds, bite_id = parts
        bite_id = int(bite_id.split('_')[1])  # Extract and convert bite_id to integer

        # Create a structured dictionary for the current event
        bite_event = {
            "bite_id": bite_id,
            "start_seconds": float(start_seconds),
            "end_seconds": float(end_seconds),
            "duration_seconds": float(duration_seconds),
        }

        # Append the structured dictionary to the bites list
        data_structure_json["bites"].append(bite_event)

        temp_data_structure.append((start_seconds, end_seconds))

# Convert the main data structure to JSON
json_output = json.dumps(data_structure_json, indent=4)

# Optionally, save the JSON output to a file
output_file_path = file_path.replace('.txt', '.json')
with open(output_file_path, 'w') as json_file:
    json_file.write(json_output)

print(f"JSON output saved to: {output_file_path}")

data_structure_np = np.array(temp_data_structure, dtype=float)

pkl_file_path = file_path.replace('.txt', '.pkl')
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(data_structure_np, pkl_file)

print(f"NumPy array saved to: {pkl_file_path}")
