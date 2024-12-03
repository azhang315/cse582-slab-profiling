import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAGE_SIZE = 4096  # Page size in bytes (commonly 4096)

# Utility to parse slabinfo files
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

# Aggregate data from all files in a directory
def aggregate_slabinfo(input_dir):
    """Aggregate data from all slabinfo files in the directory."""
    all_data = []
    for file in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        if file.endswith(".txt"):
            parsed_data = parse_slabinfo(filepath)
            if not parsed_data.empty:
                all_data.append(parsed_data)

    if not all_data:
        raise ValueError(f"No valid slabinfo data found in directory: {input_dir}")

    aggregated = pd.concat(all_data, ignore_index=True)
    return aggregated

# Overlay scatter plots for multiple kernels
def overlay_scatter_by_kernel(normal_data, stress_data, kernel_types, output_dir):
    """Overlay scatter plots stratified by kernel types."""
    plt.figure(figsize=(12, 8))

    for kernel_type, normal_df, stress_df in zip(kernel_types, normal_data, stress_data):
        plt.scatter(
            normal_df["Memory Usage"], normal_df["Overhead Percentage"],
            label=f"{kernel_type} Normal", alpha=0.6
        )
        plt.scatter(
            stress_df["Memory Usage"], stress_df["Overhead Percentage"],
            label=f"{kernel_type} Stress", alpha=0.6
        )

    plt.xlabel("Memory Usage (Bytes)")
    plt.ylabel("Overhead Percentage (%)")
    plt.title("Memory Usage vs Overhead: Kernel Comparison")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "scatter_overlay_kernels.png"))
    plt.close()

def plot_double_bar_graph_with_errors(kernel_data, categories, output_path):
    """
    Generate a double bar graph comparing average overhead % with error bars and annotated differences.
    """

    # Prepare data
    results = {}
    errors = {}
    for kernel, data in kernel_data.items():
        filtered_data = data[data["Cache Name"].isin(categories)]
        category_means = filtered_data.groupby("Cache Name")["Overhead Percentage"].mean()
        category_stds = filtered_data.groupby("Cache Name")["Overhead Percentage"].std()

        # Debugging output: print the means and standard deviations
        print(f"\nKernel: {kernel}")
        print("Means:")
        print(category_means)
        print("Standard Deviations:")
        print(category_stds)

        results[kernel] = category_means
        errors[kernel] = category_stds

    # Create a unified index of categories
    all_categories = sorted(set().union(*[res.index for res in results.values()]))

    # Align results with all categories
    aligned_results = {
        kernel: [results[kernel].get(cat, 0) for cat in all_categories]
        for kernel in results
    }
    aligned_errors = {
        kernel: [errors[kernel].get(cat, 0) for cat in all_categories]
        for kernel in results
    }

    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(all_categories))

    # Plot the double bar graph
    plt.figure(figsize=(12, 8))
    bars = []
    for i, (kernel, values) in enumerate(aligned_results.items()):
        error_values = aligned_errors[kernel]
        bar = plt.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=kernel,
            yerr=error_values,  # Add error bars here
            capsize=5,
            alpha=0.8
        )
        bars.append(bar)

    # Annotate percentage differences
    if len(aligned_results) >= 2:
        kernels = list(aligned_results.keys())
        for idx, category in enumerate(all_categories):
            val1 = aligned_results[kernels[0]][idx]
            val2 = aligned_results[kernels[1]][idx]

            if val1 > 0:  # Avoid division by zero
                percent_diff = (val2 - val1) / val1 * 100
                annotation_x = idx + bar_width / 2
                annotation_y = max(val1, val2) + 2  # Position slightly above the taller bar
                plt.text(
                    annotation_x,
                    annotation_y,
                    f"{percent_diff:.1f}%",
                    ha="center",
                    fontsize=10,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
                )

    # Add labels, legend, and grid
    plt.xlabel("Cache Categories")
    plt.ylabel("Average Overhead Percentage (%)")
    plt.title("Comparison of Average Overhead Percentage Across Kernels (Stress)")
    plt.xticks(x + (bar_width / 2), all_categories, rotation=45, ha="right")
    plt.legend(title="Kernel Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Double bar graph with error bars saved at {output_path}")



def main():
    input_dir_base = "./data/stress-overhead/"
    output_dir = "./memory_overhead_graphs/overlay/"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data for each kernel
    kernel_dirs = [d for d in os.listdir(input_dir_base) if os.path.isdir(os.path.join(input_dir_base, d))]
    kernel_data = {}
    for kernel in kernel_dirs:
        stress_dir = os.path.join(input_dir_base, kernel, "stress")
        kernel_data[kernel] = aggregate_slabinfo(stress_dir)

    # Categories to include
    slab_related_categories = [
        "mqueue_inode_cache", "request_queue", "v9fs_inode_cache", "net_namespace", "proc_inode_cache"
    ]

    # Double bar graph
    plot_double_bar_graph_with_errors(
        kernel_data=kernel_data,
        categories=slab_related_categories,
        output_path=os.path.join(output_dir, "double_bar_graph_overhead_comparison.png")
    )

    print(f"Overlay plots saved in {output_dir}")


if __name__ == "__main__":
    main()
