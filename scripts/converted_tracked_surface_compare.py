import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

# Function to find the clotboundary field in the data structure
def find_clotboundary_field(data):
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].dtype == object:
            if all(isinstance(cell, np.ndarray) for cell in data[key].squeeze()):
                return key
    return None

# Function to plot and save scatter plots for cell data
def plot_scatter(coordinates_with_labels, title, output_folder, file_name, x_label='X (μm)', y_label='Y (μm)', x_ticks=None, y_lims=None, x_lims=None):
    plt.figure()
    for item in coordinates_with_labels:
        coords = item['data']
        label = item.get('label', '')
        plt.scatter(coords[:, 0], coords[:, 1], label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().invert_yaxis()
    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_lims is not None:
        plt.ylim(y_lims)
    if x_lims is not None:
        plt.xlim(x_lims)
    plt.legend()
    plt.box()
    plt.savefig(os.path.join(output_folder, file_name))
    plt.close()


# Function to plot and save line plots
def plot_line(x, y, title, xlabel, ylabel, legend, output_folder, file_name):
    plt.figure()
    for y_values, label in zip(y, legend):
        plt.plot(x, y_values, '-o', label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.box()
    plt.savefig(os.path.join(output_folder, file_name))
    plt.close()

# Specify the actual lengths in micrometers
actual_length_x = 2400
actual_length_y = 1800

# File paths and other parameters
file_paths = ['TrackedData.mat']
legend_labels = ['Bare Zn']
output_folder = 'Tracked_surface_compare_plots'

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Data processing and analysis
all_clotboundary = []
all_mean_absolute_deviations = []
all_mean_y_values = []
all_thresholds = []

# Load and process each .mat file
for file_path in file_paths:
    data = loadmat(file_path, squeeze_me=True)
    clotboundary_field = find_clotboundary_field(data)

    if clotboundary_field:
        clotboundary_data = data[clotboundary_field].squeeze()
        all_clotboundary.append(clotboundary_data)

        mean_absolute_deviations = []
        mean_y_values = []
        thresholds = []

        for i, cb in enumerate(clotboundary_data):
            if cb.size > 0 and cb.ndim == 2:
                coordinates = np.flip(cb, axis=1)  # Swap x and y coordinates
                mean_y = np.mean(coordinates[:, 1])
                threshold = 1.3 * mean_y

                valid_indices = (coordinates[:, 0] >= 1) & (coordinates[:, 0] <= 2400) & (coordinates[:, 1] <= 1.2 * threshold)
                filtered_coords = coordinates[valid_indices]

                if filtered_coords.size > 0:
                    mad = np.mean(np.abs(filtered_coords[:, 1] - np.min(filtered_coords[:, 1])))
                    mean_absolute_deviations.append(mad)
                    mean_y_values.append(mean_y)
                    thresholds.append(threshold)
                else:
                    mean_absolute_deviations.append(np.nan)
            else:
                mean_absolute_deviations.append(np.nan)

        all_mean_absolute_deviations.append(mean_absolute_deviations)
        all_mean_y_values.append(mean_y_values)
        all_thresholds.append(thresholds)
    else:
        print(f'Clotboundary variable not found in {file_path}')

# Display the results
for f, file_path in enumerate(file_paths):
    print(f'File {f+1} Mean Absolute Deviations:')
    print(all_mean_absolute_deviations[f])
    print(f'File {f+1} Mean Y Values:')
    print(all_mean_y_values[f])
    print(f'File {f+1} Thresholds:')
    print(all_thresholds[f])

# Plot all cell data for each file in a single plot
for i in range(len(all_clotboundary[0])):
    coordinates_with_labels = []
    for f, clotboundary in enumerate(all_clotboundary):
        if i < len(clotboundary) and clotboundary[i].size > 0:
            coordinates = np.flip(clotboundary[i], axis=1)  # Swap x and y coordinates
            mean_y = all_mean_y_values[f][i]
            threshold = all_thresholds[f][i]

            # Filter coordinates
            valid_indices = (coordinates[:, 0] >= 0) & (coordinates[:, 0] <= 2400) & (coordinates[:, 1] <= 0.74 * threshold)
            filtered_coords = coordinates[valid_indices]

            if filtered_coords.size > 0:
                # Scale both X and Y coordinates correctly
                scaled_x = (filtered_coords[:, 0] / np.max(filtered_coords[:, 0])) * actual_length_x
                scaled_y = (filtered_coords[:, 1] / np.max(filtered_coords[:, 1])) * actual_length_y

                # Add data to list for plotting
                coordinates_with_labels.append({
                    'data': np.column_stack((scaled_x, scaled_y)),
                    'label': legend_labels[f]
                })

    # Check if there's any data to plot, then plot
    if coordinates_with_labels:
        plot_scatter(
            coordinates_with_labels,
            title=f'{(i-1)*10} min',
            output_folder=output_folder,
            file_name=f'{(i-1)*10} min.jpg',
            x_ticks=np.arange(0, 2401, 600),
            y_lims=[1100, 1600],
            x_lims=[0, 2400]
        )


# Plot mean absolute deviations for each file with connecting lines
times = np.arange(len(all_mean_absolute_deviations[0])) * 10  # Assuming each index is a 10-minute interval
plot_line(
    times,
    all_mean_absolute_deviations,
    title='Mean Absolute Deviations for Each Cell and File',
    xlabel='Time (min)',
    ylabel='Mean Absolute Deviation',
    legend=legend_labels,
    output_folder=output_folder,
    file_name='Mean_Absolute_Deviation_plot.jpg'
)
