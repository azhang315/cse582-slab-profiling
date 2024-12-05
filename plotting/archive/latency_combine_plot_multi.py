import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools

if len(sys.argv) < 2:
    print("Usage: python3 latency_combine_plot.py <combined_file_1> <combined_file_2>")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

# Load data for each kernel
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

data1["Kernel"] = "Kernel 1"
data2["Kernel"] = "Kernel 2"

combined_data = pd.concat([data1, data2], ignore_index=True)

# Process latencies and group by kernel and size
specific_sizes = [32, 4096]
for size in specific_sizes:
    size_data = combined_data[combined_data["Bytes"] == size]
    
    plt.figure()
    ax = plt.gca()
    colors = itertools.cycle(plt.cm.tab10.colors)  # Color cycle

    for kernel, group in size_data.groupby("Kernel"):
        # Calculate sorted latencies and CDF
        latency_sorted = group["Latency (µs)"].sort_values()
        cdf = latency_sorted.rank(method="average", pct=True)

        # Calculate statistics for error bands
        mean_latency = latency_sorted.mean()
        std_latency = latency_sorted.std()
        lower_bound = latency_sorted - std_latency
        upper_bound = latency_sorted + std_latency

        # Plot main CDF line
        color = next(colors)
        plt.plot(latency_sorted, cdf, label=f"{kernel} (Mean: {mean_latency:.2f} µs, Samples: {len(group)})", color=color)

        # Add error bands
        plt.fill_between(latency_sorted, cdf, cdf, where=(lower_bound >= 0), color=color, alpha=0.2, label=f"{kernel} ±1 SD")

    # Add labels and titles
    plt.xlabel("Latency (µs)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Allocation Latency (Overlayed, {size}B)")
    plt.legend(title="Kernel", loc="upper left")
    plt.grid(True)
    plt.savefig(f"./combined_graphs/overlay_latency_cdf_{size}B_with_error.png")
    plt.close()
