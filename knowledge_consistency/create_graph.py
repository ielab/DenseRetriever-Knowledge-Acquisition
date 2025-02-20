import argparse
import os
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def parse_boolean_file(filepath):
    """
    Reads a file where each line is 'true' or 'false'.
    Returns a list of 1/0 values.
    """
    bool_values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if line == "true":
                bool_values.append(1)
            elif line == "false":
                bool_values.append(0)
            else:
                continue
    return bool_values

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results with Bonferroni-corrected t-tests (PNG output).")
    args = parser.parse_args()

    # Increase default font sizes
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14
    })

    # Define how many layers we use per encoder
    # cls/mean => 0..12, eos => 0..32
    layers_per_encoder = {
        "cls":  range(0, 13),
        "mean": range(0, 13),
        "eos":  range(0, 33)
    }

    # We'll reduce x-axis ticks so they're not too cluttered
    xticks_for_encoder = {
        "cls":  [0, 2, 4, 6, 8, 10, 12],
        "mean": [0, 2, 4, 6, 8, 10, 12],
        "eos":  [0, 4, 8, 12, 16, 20, 24, 28, 32]
    }

    # For each group index, we have a set of models and a corresponding encoder type
    model_namess = [
        ["bert-base-uncased", "dpr-question", "dpr-context", "dpr-paired"],  # group0 -> cls
        ["bert-base-uncased", "contriever"],                                # group1 -> mean
        ["llama", "repllama"]                                               # group2 -> eos
    ]
    encoder_types = ["cls", "mean", "eos"]

    # Base model for each group (for significance testing)
    base_model_for_group = {
        0: "bert-base-uncased",
        1: "bert-base-uncased",
        2: "llama"
    }

    # We'll produce a figure for each dataset
    datasets = ["nq", "msmarco"]
    # We'll produce one row of 4 plots for each file, one per 'num_passages'
    num_passages = (2, 3, 4, 5)

    base_folder = "linear_multi"

    # Store raw boolean lists:
    # results[(model_name, dataset, n_pass, layer, encoder_type)] = [0/1, ...]
    results = {}

    # 1) Load all data
    for group_idx, model_names in enumerate(model_namess):
        encoder_type = encoder_types[group_idx]
        layer_range = layers_per_encoder[encoder_type]
        for model_name in model_names:
            for dataset in datasets:
                for n_pass in num_passages:
                    for layer in layer_range:
                        file_path = (
                            f"{base_folder}/{dataset}/{encoder_type}/{model_name}/"
                            f"num_passages_{n_pass}/layer_{layer}/evaluation_results.txt"
                        )
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"File not found: {file_path}")
                        bool_list = parse_boolean_file(file_path)
                        results[(model_name, dataset, n_pass, layer, encoder_type)] = bool_list

    # 2) Create plots: one PNG per (dataset, encoder_type).
    for dataset in datasets:
        for group_idx, model_names in enumerate(model_namess):
            encoder_type = encoder_types[group_idx]
            base_model_name = base_model_for_group[group_idx]
            layer_range = layers_per_encoder[encoder_type]

            # Determine how many tests we do for Bonferroni: one per layer
            num_tests = len(layer_range)
            alpha = 0.05

            # Create a single-row figure with 4 columns (1 row x 4 columns)
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharey=True)

            for ax_idx, n_pass in enumerate(num_passages):
                ax = axes[ax_idx]
                ax.set_title(f"Passage = {n_pass}", fontsize=16)

                # X-axis ticks
                ax.set_xticks(xticks_for_encoder[encoder_type])
                # No grid (as requested)
                # ax.grid(False)  # (default is no grid)

                # Y-label only on the leftmost subplot
                if ax_idx == 0:
                    ax.set_ylabel("Accuracy")

                # Plot each model line
                for model_name in model_names:
                    layer_accuracies = []
                    sig_positions = []

                    for layer in layer_range:
                        bool_list = results[(model_name, dataset, n_pass, layer, encoder_type)]
                        accuracy = np.mean(bool_list)

                        # T-test vs base model (unless it's the base model itself)
                        if model_name != base_model_name:
                            base_bool_list = results[(base_model_name, dataset, n_pass, layer, encoder_type)]
                            tstat, pval = ttest_ind(bool_list, base_bool_list, alternative='two-sided')
                            # Bonferroni correction: multiply by number of tests or compare to alpha/num_tests
                            pval_corrected = pval * num_tests
                            if pval_corrected < alpha:
                                sig_positions.append((layer, accuracy))

                        layer_accuracies.append(accuracy)

                    # Plot line with marker size=2
                    line, = ax.plot(
                        layer_range,
                        layer_accuracies,
                        marker='o',
                        markersize=2,
                        label=model_name
                    )

                    # Plot red star marker (size=4) for significant points
                    if model_name != base_model_name:
                        for (x_star, y_star) in sig_positions:
                            ax.plot(
                                [x_star], [y_star],
                                marker='*',
                                markersize=4,
                                color='red',
                                linestyle='None'
                            )

            # Put a single x-label under the entire row of subplots
            fig.text(0.5, 0.02, 'Layer', ha='center', fontsize=16)

            # Create one combined legend at the top center, in one row
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
                       ncol=len(model_names), frameon=False)

            fig.tight_layout(rect=[0, 0.04, 1, 0.95])

            # Save the figure as PNG: e.g. "nq-cls.png"
            out_filename = f"{dataset}-{encoder_type}.pdf"
            plt.savefig(out_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved figure to {out_filename}")

if __name__ == "__main__":
    main()
