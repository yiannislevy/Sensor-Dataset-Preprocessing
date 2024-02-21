import plotly.graph_objects as go
import numpy as np


def plot_annotated_meal(meal_data):
    """
    Plot the meal weight over time with vertical lines at the start of each bite,
    indicating bite events. These lines are parallel to the y-axis and span the entire plot height.

    Parameters:
    - meal_data (dict): Dictionary with meal and bite data.

    Returns:
    - Plotly graph object figure.
    """
    bites = meal_data['meals'][0]['bites']
    initial_weight = sum(b['bite_weight'] for b in bites if b['bite_weight'])
    weight_timeline = [(0, initial_weight)]
    vertical_lines = []

    for b in bites:
        if b['bite_weight']:
            weight_timeline.append((b['bite_start_t'], weight_timeline[-1][1]))
            weight_timeline.append((b['bite_end_t'], weight_timeline[-1][1] - b['bite_weight']))
            # Define a vertical line at the start of the bite
            vertical_lines.append({'type': 'line',
                                   'x0': b['bite_start_t'],
                                   'y0': 0,
                                   'x1': b['bite_start_t'],
                                   'y1': 1,
                                   'xref': 'x',
                                   'yref': 'paper',
                                   'line': {'color': 'red', 'width': 1}})

    # Extend the last point beyond the last bite for better visualization
    weight_timeline.append((weight_timeline[-1][0] + 10, weight_timeline[-1][1]))

    # Extract the time and weights as separate lists for plotting
    time, weights = zip(*weight_timeline)

    fig = go.Figure(data=go.Scatter(x=time, y=weights, mode='lines', line=dict(color='green')))
    fig.update_layout(title='Meal Weight Over Time',
                      xaxis_title='Time (sec)',
                      yaxis_title='Weight (g)',
                      shapes=vertical_lines)  # Add the vertical lines here

    return fig


def plot_comparison(ground_truth_data, estimated_data):
    """
    Plot the comparison of ground truth meal data with estimated meal data.

    Parameters:
    - ground_truth_data (dict): Dictionary with true meal and bite data.
    - estimated_data (np.array): The processed meal weight data from the estimation algorithm.

    Returns:
    - Plotly graph object figure.
    """
    # Process ground truth data
    bites = ground_truth_data['meals'][0]['bites']
    initial_weight = sum(b['bite_weight'] for b in bites if b['bite_weight'])
    ground_truth_timeline = [(0, initial_weight)]

    for b in bites:
        if b['bite_weight']:
            ground_truth_timeline.append((b['bite_start_t'], ground_truth_timeline[-1][1]))
            ground_truth_timeline.append((b['bite_end_t'], ground_truth_timeline[-1][1] - b['bite_weight']))

    ground_truth_timeline.append((ground_truth_timeline[-1][0] + 10, ground_truth_timeline[-1][1]))

    # Extract time and weights for ground truth
    time_gt, weights_gt = zip(*ground_truth_timeline)

    # For the estimated data
    # Assuming the estimated data has the same time intervals as the ground truth
    time_est = [time for time, _ in ground_truth_timeline]
    weights_est = estimated_data[:len(time_est)]

    # Create the figure
    fig = go.Figure()

    # Add ground truth data
    fig.add_trace(go.Scatter(x=time_gt, y=weights_gt, mode='lines', name='Ground Truth', line=dict(color='green')))

    # Add estimated data
    fig.add_trace(go.Scatter(x=time_est, y=weights_est, mode='lines', name='Estimation', line=dict(color='red')))

    # Update the layout
    fig.update_layout(title='Comparison of Meal Weight Over Time',
                      xaxis_title='Time (sec)',
                      yaxis_title='Weight (g)')

    return fig

# Use the function to plot your data
# Assuming 'ground_truth_data' and 'estimated_data' are already defined and processed
# fig = plot_comparison(ground_truth_data, estimated_data)
# fig.show()
