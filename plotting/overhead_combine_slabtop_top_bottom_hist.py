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

        # Parse slabtop cache data
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

def find_top_and_bottom_caches(data, top_n=5):
    """Find the top and bottom caches based on Overhead Percentage."""
    top_caches = data.nlargest(top_n, "Overhead Percentage")
    bottom_caches = data[data["Overhead Percentage"] > 0].nsmallest(top_n, "Overhead Percentage")
    return top_caches, bottom_caches

def plot_top_bottom_caches(top_caches, bottom_caches, output_path):
    """Plot bar graphs for the top and bottom caches."""
    plt.figure(figsize=(12, 6))

    # Top caches
    plt.bar(top_caches["Cache Name"], top_caches["Overhead Percentage"], color="red", alpha=0.7, label="Top Overheads")

    # Bottom caches
    plt.bar(bottom_caches["Cache Name"], bottom_caches["Overhead Percentage"], color="blue", alpha=0.7, label="Bottom Overheads (Non-Zero)")

    # Annotate bars
    for idx, row in top_caches.iterrows():
        plt.text(idx, row["Overhead Percentage"], f"{row['Overhead Percentage']:.1f}%", ha="center", va="bottom", fontsize=8)

    for idx, row in bottom_caches.iterrows():
        plt.text(idx + len(top_caches), row["Overhead Percentage"], f"{row['Overhead Percentage']:.1f}%", ha="center", va="bottom", fontsize=8)

    # Plot configurations
    plt.title("Top and Bottom Cache Overhead Percentages")
    plt.xlabel("Cache Categories")
    plt.ylabel("Overhead Percentage (%)")
    plt.xticks(
        ticks=range(len(top_caches) + len(bottom_caches)),
        labels=list(top_caches["Cache Name"]) + list(bottom_caches["Cache Name"]),
        rotation=45,
        ha="right"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Top and bottom cache plots saved at {output_path}")

def main():
    input_dir_base = "./data/stress-overhead/"
    output_dir = "./memory_overhead_graphs/top_bottom/"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data for each kernel
    kernel_dirs = [d for d in os.listdir(input_dir_base) if os.path.isdir(os.path.join(input_dir_base, d))]
    kernel_data = {}
    for kernel in kernel_dirs:
        stress_dir = os.path.join(input_dir_base, kernel, "stress")
        kernel_data[kernel] = aggregate_slabtop(stress_dir)

    # Analyze and plot top/bottom caches for each kernel
    for kernel, data in kernel_data.items():
        top_caches, bottom_caches = find_top_and_bottom_caches(data, top_n=5)

        # Plot top and bottom caches
        plot_top_bottom_caches(
            top_caches=top_caches,
            bottom_caches=bottom_caches,
            output_path=os.path.join(output_dir, f"top_bottom_caches_{kernel}.png")
        )

if __name__ == "__main__":
    main()
