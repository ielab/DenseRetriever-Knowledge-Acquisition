import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from glob import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot neuron activation heatmap from attributions file."
    )
    parser.add_argument(
        "--save_folders",
        nargs="+",
        required=True,
        help="One or more folders with JSON attribution files to plot side by side."
    )
    parser.add_argument("--num_bins", type=int, default=10,
                        help="Number of bins to aggregate neuron activations")
    parser.add_argument("--max_number", type=int, default=54414,
                        help="Maximum number of activations")
    return parser.parse_args()

def bin_data(data, num_bins, max_number):
    """Aggregate data into a fixed number of bins."""
    bin_size = max(1, len(data) // num_bins)
    binned_data = np.zeros(num_bins)

    for i in range(num_bins):
        start = i * bin_size
        end = min(start + bin_size, len(data))
        # Take the max in this bin and scale by max_number
        binned_data[i] = max(data[start:end]) / max_number

    return binned_data

def load_attributions(file_paths):
    """
    Loads attributions from all JSON files in file_paths,
    summing them if they share the same key.
    Returns:
      attributions: dict of {layer_name: [activation values]}
      num_neuron_dict: dict of {layer_name: count of >0 entries}
    """
    attributions = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            current_attributions = data["attributions"]
            for key, value in current_attributions.items():
                if key not in attributions:
                    attributions[key] = value
                else:
                    # Add the lists element-wise
                    attributions[key] = [x + y for x, y in zip(attributions[key], value)]

    # Count how many items are above 0
    num_neuron_dict = {}
    for key in attributions:
        binarized = [1 if x > 0 else 0 for x in attributions[key]]
        num_neuron_dict[key] = sum(binarized)

    return attributions, num_neuron_dict

def prepare_data_for_plot(attributions, num_item_dict, num_bins, max_number):
    """
    Sort keys (layer names), bin them, and return
    the row keys, row keys with counts, and a 2D matrix.
    """
    def sort_key(key):
        # Example key: "layer_1_intermediate" or "layer_1_output"
        # We parse out the layer number and whether it's intermediate or output
        parts = key.split('_')
        layer_num = int(parts[1])
        # Put 'intermediate' before 'output'
        order = 0 if parts[2] == "intermediate" else 1
        return (layer_num, order)

    row_keys = sorted(attributions.keys(), key=sort_key)
    # Full label: "layer_0_intermediate (1309)"
    row_keys_representation = [f"{key} ({num_item_dict[key]})" for key in row_keys]
    # Count-only label: "1309"
    row_keys_counts = [f"({str(num_item_dict[k])})" for k in row_keys]

    binned_attributions = {
        k: bin_data(attributions[k], num_bins, max_number=max_number) for k in row_keys
    }

    max_width = num_bins
    num_rows = len(row_keys)
    mat = np.zeros((num_rows, max_width))

    for i, key in enumerate(row_keys):
        row = np.array(binned_attributions[key])
        mat[i, :] = row

    return row_keys, row_keys_representation, row_keys_counts, mat

def main():
    args = parse_args()

    num_folders = len(args.save_folders)
    if num_folders < 1:
        print("Please provide at least one folder.")
        return

    # Create side-by-side subplots: 1 row, N columns
    fig, axes = plt.subplots(1, num_folders, figsize=(10, 6))
    if num_folders == 1:
        # Make sure axes is iterable
        axes = [axes]

    # Updated colormap with more colors for better visualization
    cmap = LinearSegmentedColormap.from_list("paper",
                                             ["white", "lightblue", "blue", "green", "yellow", "red"])
    im = None  # We'll store the last image for the shared colorbar

    for idx, folder in enumerate(args.save_folders):
        folder_files = glob(f"{folder}/*.json")
        attributions, num_item_dict = load_attributions(folder_files)
        if not attributions:
            print(f"No attributions found in folder: {folder}")
            continue

        row_keys, row_keys_representation, row_keys_counts, mat = prepare_data_for_plot(
            attributions, num_item_dict, args.num_bins, args.max_number
        )

        ax = axes[idx]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Set title using folder name, adjust font size if needed
        ax.set_title(folder.split("/")[-2], fontsize=18)
        # Set the Y ticks
        ax.set_yticks(np.arange(len(row_keys)))

        # For the second subplot (idx == 1), only show the count:
        if idx == 1:
            ax.set_yticklabels(row_keys_counts, fontsize=12)
        else:
            ax.set_yticklabels(row_keys_representation, fontsize=12)

        ax.set_xticks([])

    # Adjust layout so there's space at the bottom for the colorbar
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Add a new axis for the color bar at the bottom.
    cax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label("Percentage of Examples above Threshold", fontsize=14)
    # Increase the size of the tick labels on the color bar
    cbar.ax.tick_params(labelsize=14)

    # Save the figure
    output_path = f"{args.save_folders[0].replace('output', 'graphs')}/activation_heatmap_side_by_side.pdf"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main()
