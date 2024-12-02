import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAGE_SIZE = 4096  # Page size in bytes (commonly 4096)

def annotate_statistics(ax, normal_mean, stress_mean, overhead_diff, xlabel, ylabel):
    """Annotate overhead and percentage differences."""
    ax.axhline(normal_mean, color='blue', linestyle='--', label=f"Normal Mean Overhead: {normal_mean:.2f}%")
    ax.axhline(stress_mean, color='orange', linestyle='--', label=f"Stress Mean Overhead: {stress_mean:.2f}%")
    ax.text(
        0.5, 0.95,
        f"Difference in Overhead: {overhead_diff:.2f}%",
        fontsize=10,
        color='red',
        ha='center',
        va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    ax.grid(True)


# # Overlay histograms
# def plot_overlay_histograms(normal_data, stress_data, column, xlabel, ylabel, title, output_path):
#     """Overlay histograms for normal and stress periods."""
#     plt.figure(figsize=(10, 6))

#     # Normal data
#     plt.hist(normal_data[column], bins=30, alpha=0.6, label="Normal", color="blue")
#     # Stress data
#     plt.hist(stress_data[column], bins=30, alpha=0.6, label="Stress", color="orange")

#     # Annotate statistics
#     ax = plt.gca()
#     annotate_statistics(ax, normal_data[column], "Normal")
#     annotate_statistics(ax, stress_data[column], "Stress")

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend(loc="upper right")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()


# Overlay scatter plots
def plot_overlay_scatter(normal_data, stress_data, x_col, y_col, xlabel, ylabel, title, output_path):
    """Overlay scatter plots for normal and stress periods."""
    plt.figure(figsize=(10, 6))

    # Normal data
    plt.scatter(normal_data[x_col], normal_data[y_col], alpha=0.6, label="Normal", color="blue")
    # Stress data
    plt.scatter(stress_data[x_col], stress_data[y_col], alpha=0.6, label="Stress", color="orange")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_stratified_overlay(normal_data, stress_data, group_column, value_column, xlabel, ylabel, output_path):
    """Stratify data by a column and overlay stress vs normal."""
    stratified_groups = normal_data[group_column].unique()

    plt.figure(figsize=(12, 8))
    for group in stratified_groups:
        normal_group = normal_data[normal_data[group_column] == group]
        stress_group = stress_data[stress_data[group_column] == group]

        if normal_group.empty or stress_group.empty:
            continue

        # Calculate mean overhead
        normal_mean = normal_group[value_column].mean()
        stress_mean = stress_group[value_column].mean()
        overhead_diff = stress_mean - normal_mean

        # Plot stress and normal
        plt.scatter(normal_group["Memory Usage"], normal_group[value_column], label=f"{group} Normal", alpha=0.6, color="blue")
        plt.scatter(stress_group["Memory Usage"], stress_group[value_column], label=f"{group} Stress", alpha=0.6, color="orange")

        # Annotate key statistics
        annotate_statistics(
            plt.gca(), normal_mean, stress_mean, overhead_diff,
            xlabel="Memory Usage (Bytes)", ylabel=ylabel
        )

    plt.title(f"Stratified {value_column} Overlays: Normal vs Stress")
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Function to identify and highlight slabs with the greatest changes
def highlight_greatest_changes(normal_data, stress_data, output_dir):
    """Identify slabs with the greatest overhead changes."""
    slabs_diff = []
    for slab in normal_data["Cache Name"].unique():
        if slab not in stress_data["Cache Name"].values:
            continue

        normal_overhead = normal_data[normal_data["Cache Name"] == slab]["Overhead Percentage"].mean()
        stress_overhead = stress_data[stress_data["Cache Name"] == slab]["Overhead Percentage"].mean()

        if not np.isnan(normal_overhead) and not np.isnan(stress_overhead):
            slabs_diff.append((slab, normal_overhead, stress_overhead, stress_overhead - normal_overhead))

    # Sort by absolute change
    slabs_diff = sorted(slabs_diff, key=lambda x: abs(x[3]), reverse=True)

    # Top slabs
    top_slabs = slabs_diff[:5]
    print("Top Slabs with Greatest Overhead Change:")
    for slab, normal, stress, diff in top_slabs:
        print(f"Slab: {slab}, Normal: {normal:.2f}%, Stress: {stress:.2f}%, Diff: {diff:.2f}%")

    # Highlight in a plot
    plt.figure(figsize=(10, 6))
    labels, normal_values, stress_values, diffs = zip(*top_slabs)
    x = np.arange(len(labels))

    plt.bar(x - 0.2, normal_values, width=0.4, label="Normal Overhead", color="blue")
    plt.bar(x + 0.2, stress_values, width=0.4, label="Stress Overhead", color="orange")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Cache Name")
    plt.ylabel("Overhead Percentage (%)")
    plt.title("Top 5 Slabs with Greatest Overhead Change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "greatest_overhead_changes.png"))
    plt.close()


def parse_slabinfo(filepath):
    """Parse a single slabinfo file and return a DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

        # Ensure slabinfo header is present
        if not lines or not lines[0].startswith("slabinfo - version:"):
            raise ValueError(f"File {filepath} does not appear to be a valid slabinfo file.")

        for line in lines[2:]:  # Skip the first two header lines
            fields = re.split(r'\s+', line.strip())
            if len(fields) >= 6:  # Ensure enough fields are present
                try:
                    name = fields[0]
                    active_objs = int(fields[1])
                    total_objs = int(fields[2])
                    obj_size = int(fields[3])
                    obj_per_slab = int(fields[4])
                    pages_per_slab = int(fields[5])

                    if total_objs == 0:
                        continue  # Skip entries with no objects

                    memory_usage = total_objs * obj_size  # Compute total memory usage
                    overhead_percentage = max(0, ((pages_per_slab * PAGE_SIZE) / obj_per_slab - obj_size) / obj_size * 100)

                    data.append({
                        "Cache Name": name,
                        "Active Objects": active_objs,
                        "Total Objects": total_objs,
                        "Object Size": obj_size,
                        "Memory Usage": memory_usage,
                        "Overhead Percentage": overhead_percentage,
                    })
                except ValueError:
                    continue  # Skip malformed lines

    return pd.DataFrame(data)


# Main function to plot
def main():
    input_dir_normal = "./data/overhead/normal/"
    input_dir_stress = "./data/overhead/stress/"
    output_dir = "./memory_overhead_graphs/overlay/"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    def aggregate_slabinfo(input_dir):
        """Aggregate data from all slabinfo files in the directory."""
        all_data = []
        for file in sorted(os.listdir(input_dir)):
            filepath = os.path.join(input_dir, file)
            if file.endswith(".txt"):
                print(f"Parsing file: {filepath}")  # Debug
                parsed_data = parse_slabinfo(filepath)
                if not parsed_data.empty:
                    all_data.append(parsed_data)
                else:
                    print(f"No valid data in file: {filepath}")  # Debug

        if not all_data:
            raise ValueError(f"No valid slabinfo data found in directory: {input_dir}")

        aggregated = pd.concat(all_data, ignore_index=True)
        print("Aggregated Data Columns:", aggregated.columns)  # Debug
        return aggregated



    print("Aggregating normal workload data...")
    normal_data = aggregate_slabinfo(input_dir_normal)

    print("Aggregating stress workload data...")
    stress_data = aggregate_slabinfo(input_dir_stress)

    # # Overlay histograms
    # plot_overlay_histograms(
    #     normal_data, stress_data,
    #     column="Memory Usage",
    #     xlabel="Memory Usage (Bytes)",
    #     ylabel="Frequency",
    #     title="Memory Usage Comparison: Normal vs Stress",
    #     output_path=os.path.join(output_dir, "memory_usage_overlay.png")
    # )

    # plot_overlay_histograms(
    #     normal_data, stress_data,
    #     column="Overhead Percentage",
    #     xlabel="Overhead Percentage (%)",
    #     ylabel="Frequency",
    #     title="Overhead Percentage Comparison: Normal vs Stress",
    #     output_path=os.path.join(output_dir, "overhead_percentage_overlay.png")
    # )

    # Overlay scatter plots
    plot_overlay_scatter(
        normal_data, stress_data,
        x_col="Memory Usage",
        y_col="Overhead Percentage",
        xlabel="Memory Usage (Bytes)",
        ylabel="Overhead Percentage (%)",
        title="Memory Usage vs Overhead: Normal vs Stress",
        output_path=os.path.join(output_dir, "scatter_memory_vs_overhead_overlay.png")
    )

    # Plot stratified overlays
    plot_stratified_overlay(
        normal_data, stress_data,
        group_column="Cache Name",
        value_column="Overhead Percentage",
        xlabel="Memory Usage (Bytes)",
        ylabel="Overhead Percentage (%)",
        output_path=os.path.join(output_dir, "stratified_overlay.png")
    )

    # Highlight slabs with greatest changes
    highlight_greatest_changes(normal_data, stress_data, output_dir)
    
    print(f"Overlay plots saved in {output_dir}")


if __name__ == "__main__":
    main()
