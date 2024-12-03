import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAGE_SIZE = 4096  # Page size in bytes (commonly 4096)

import re
import pandas as pd

def parse_slabtop(filepath):
    """Parse a single slabtop file and return a DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip the summary section
    start_parsing = False
    for line in lines:
        if line.startswith("  OBJS"):  # Start parsing after this header
            start_parsing = True
            continue
        if not start_parsing:
            continue

        # Parse slabtop cache data
        fields = re.split(r'\s+', line.strip())
        if len(fields) >= 8:  # Ensure sufficient fields are present
            try:
                name = fields[7]  # Cache name
                # total_objs = int(fields[0])  # Total objects
                # obj_size_kb = float(fields[3][:-1])  # Object size (strip 'K')
                # cache_size_kb = float(fields[5][:-1])  # Cache size (strip 'K')

                # if total_objs == 0 or obj_size_kb == 0 or cache_size_kb == 0:
                #     continue

                # # Theoretical memory usage
                # theoretical_usage_kb = total_objs * obj_size_kb

                # # Overhead percentage
                # overhead_percentage = max(0, (cache_size_kb - theoretical_usage_kb) / cache_size_kb * 100)
                # Example: Debug parsing of slabtop fields
                total_objects = int(fields[0])  # Total objects
                object_size_kb = float(fields[3][:-1])  # Remove "K" and convert to float
                cache_size_kb = float(fields[6][:-1])  # Remove "K" and convert to float

                # Overhead calculation
                theoretical_usage = total_objects * object_size_kb  # In KB
                overhead_percentage = max(0, (cache_size_kb - theoretical_usage) / cache_size_kb * 100 if cache_size_kb > 0 else 0)

                # Debug output
                print(f"DEBUG: {name}: Total={total_objects}, Size={object_size_kb}, Cache={cache_size_kb}, Overhead={overhead_percentage:.2f}%")

                data.append({
                    "Cache Name": name,
                    # "Total Objects": total_objs,
                    # "Object Size (KB)": obj_size_kb,
                    # "Cache Size (KB)": cache_size_kb,
                    "Overhead Percentage": overhead_percentage,
                })

                # Debugging output
                # print(f"DEBUG: {name}: Total={total_objs}, Size={obj_size_kb}, Cache={cache_size_kb}, Overhead={overhead_percentage:.2f}%")

            except (ValueError, IndexError, ZeroDivisionError):
                continue  # Skip malformed lines

    return pd.DataFrame(data)





def aggregate_slabtop(input_dir):
    """Aggregate data from all slabtop files in the directory."""
    all_data = []
    for file in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        if file.endswith(".txt"):
            parsed_data = parse_slabtop(filepath)
            if not parsed_data.empty:
                all_data.append(parsed_data)

    if not all_data:
        raise ValueError(f"No valid slabtop data found in directory: {input_dir}")

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
        kernel_data[kernel] = aggregate_slabtop(stress_dir)

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

