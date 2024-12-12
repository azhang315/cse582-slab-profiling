import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAGE_SIZE = 4096  # Page size in bytes (commonly 4096)

def parse_slabtop(filepath):
    """Parse a single slabtop file and return a DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_parsing = False
    for line in lines:
        if line.startswith("  OBJS"):  # Start parsing after this header
            start_parsing = True
            continue
        if not start_parsing:
            continue

        fields = re.split(r'\s+', line.strip())
        if len(fields) >= 8:  # Ensure sufficient fields are present
            try:
                name = fields[7]  # Cache name
                total_objects = int(fields[0])  # Total objects
                object_size_kb = float(fields[3][:-1])  # Remove "K" and convert to float
                cache_size_kb = float(fields[6][:-1])  # Remove "K" and convert to float

                # Overhead calculation
                theoretical_usage = total_objects * object_size_kb  # In KB
                overhead_percentage = max(0, (cache_size_kb - theoretical_usage) / cache_size_kb * 100 if cache_size_kb > 0 else 0)

                data.append({
                    "Cache Name": name,
                    "Overhead Percentage": overhead_percentage,
                })

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

def plot_side_by_side_bars(kernel_data, relevant_categories, output_path):
    """Generate side-by-side bar graphs comparing overhead percentages for relevant caches."""
    # Prepare data
    results = {}
    for kernel, data in kernel_data.items():
        filtered_data = data[data["Cache Name"].isin(relevant_categories)]
        category_means = filtered_data.groupby("Cache Name")["Overhead Percentage"].mean()

        # Debugging output: print the means
        print(f"\nKernel: {kernel}")
       
        print("Means:")
        print(category_means)

        # Store results
        results[kernel] = category_means

    # Create a unified index of relevant categories
    all_categories = sorted(relevant_categories)

    # Align results with all categories
    aligned_results = {
        kernel: [results[kernel].get(cat, 0) for cat in all_categories]
        for kernel in results
    }

    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(all_categories))

    # Plot the side-by-side bar graph
    plt.figure(figsize=(12, 8))
    for i, (kernel, values) in enumerate(aligned_results.items()):
        bar = plt.bar(
            x + i * bar_width,
            values,
            bar_width,
            label=kernel,
            alpha=0.8
        )

    # Add labels and title
    plt.xlabel("Cache Categories")
    plt.ylabel("Average Overhead Percentage (%)")
    plt.title("Comparison of Overhead Percentages for Relevant Caches Across Kernels")
    plt.xticks(x + (bar_width / 2), all_categories, rotation=45, ha="right")
    plt.legend(title="Kernel Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Side-by-side bar graph saved at {output_path}")

def main():
    input_dir_base = "./data/stress-overhead/"
    output_dir = "./memory_overhead_graphs/relevant_caches/"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Relevant cache categories to include
    relevant_categories = [
        "mqueue_inode_cache", "request_queue", "v9fs_inode_cache",
        "net_namespace", "proc_inode_cache", "sock_inode_cache",
        "radix_tree_node", "task_struct", "inode_cache", "dentry_cache"
    ]

    # Load data for each kernel
    kernel_dirs = [d for d in os.listdir(input_dir_base) if os.path.isdir(os.path.join(input_dir_base, d))]
    kernel_data = {}
    for kernel in kernel_dirs:
        stress_dir = os.path.join(input_dir_base, kernel, "stress")
        kernel_data[kernel] = aggregate_slabtop(stress_dir)

    # Plot side-by-side bar graph
    plot_side_by_side_bars(
        kernel_data=kernel_data,
        relevant_categories=relevant_categories,
        output_path=os.path.join(output_dir, "side_by_side_relevant_caches.png")
    )

if __name__ == "__main__":
    main()
