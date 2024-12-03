import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools

# Shared directory for preprocessed data
shared_dir = "./shared_data"
output_dir = "./combined_graphs"
os.makedirs(output_dir, exist_ok=True)

# Load all dataframes from the shared directory
def load_dataframes(shared_dir):
    df_list = []
    for file in os.listdir(shared_dir):
        if file.endswith(".csv"):
            kernel_name = file.split("_")[-1].replace(".csv", "")  # Extract kernel version
            df = pd.read_csv(os.path.join(shared_dir, file))
            df["Kernel"] = kernel_name  # Add a column to track the kernel version
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Function to filter outliers using IQR
def filter_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Load combined dataframes
combined_df = load_dataframes(shared_dir)
print("Loaded combined dataframe:")
print(combined_df.head())

# Define the specific sizes to inspect
specific_sizes = [32, 4096]  # Only process 32B and 4096B allocations

# Generate plots for specific sizes
colors = itertools.cycle(plt.cm.tab10.colors)  # Use consistent cycling colors

for size in specific_sizes:
    size_data = combined_df[combined_df["Bytes"] == size]

    if size_data.empty:
        print(f"No data available for {size}B allocations.")
        continue

    # Filter outliers
    filtered_data = filter_outliers(size_data, "Latency (µs)")

    # Overlayed CDF Plot with Error Bands
    plt.figure()
    ax = plt.gca()

    # Track metrics
    kernel_means = {}
    sample_counts = {}
    colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle

    for kernel, group in filtered_data.groupby("Kernel"):
        # Calculate metrics
        mean_latency = group["Latency (µs)"].mean()  # Already in µs
        std_latency = group["Latency (µs)"].std()    # Standard deviation
        kernel_means[kernel] = mean_latency
        sample_counts[kernel] = len(group)

        # Calculate sorted latencies and CDF
        latency_sorted = group["Latency (µs)"].sort_values()
        cdf = latency_sorted.rank(method="average", pct=True)

        # Plot main CDF line
        color = next(colors)
        plt.plot(
            latency_sorted, cdf,
            label=f"{kernel} (Mean: {mean_latency:.2f} µs, Samples: {len(group)})",
            linestyle="-", alpha=0.8, color=color
        )

        # Add error band using standard deviation
        lower_bound = latency_sorted - std_latency
        upper_bound = latency_sorted + std_latency
        plt.fill_between(
            latency_sorted, cdf, cdf,
            where=(lower_bound >= 0),
            color=color,
            alpha=0.2,
            label=f"{kernel} ±1 SD"
        )

    # Calculate differences
    kernel_list = list(kernel_means.keys())
    if len(kernel_list) == 2:
        k1, k2 = kernel_list
        avg_diff = abs(kernel_means[k1] - kernel_means[k2])
        pct_diff = (avg_diff / ((kernel_means[k1] + kernel_means[k2]) / 2)) * 100
        higher_latency_kernel = k1 if kernel_means[k1] > kernel_means[k2] else k2

        # Annotate percentage difference
        ax.annotate(
            f"Avg Diff: {avg_diff:.2f} µs\n% Diff: {pct_diff:.2f}%\n{higher_latency_kernel} higher latency",
            xy=(0.7, 0.2),  # Adjusted location for visibility
            xycoords="axes fraction",
            fontsize=10,
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
        )

    plt.xlabel("Latency (µs)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Allocation Latency (Overlayed, {size}B)")
    plt.grid(True)

    # Adjust legend font size and location
    plt.legend(title="Kernel", fontsize=10, loc="upper left", title_fontsize=11)
    plt.savefig(f"{output_dir}/overlay_latency_cdf_{size}B_with_error.png")
    plt.close()


    # Overlayed Latency Distribution (Histogram)
    plt.figure()
    ax = plt.gca()
    colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
    for kernel, group in filtered_data.groupby("Kernel"):
        plt.hist(
            group["Latency (µs)"],
            bins=30,
            alpha=0.5,
            label=f"{kernel} (Samples: {len(group)})",
            color=next(colors)
        )
    plt.xlabel("Latency (µs)")
    plt.ylabel("Frequency")
    plt.title(f"Latency Distribution (Overlayed, {size}B)")
    plt.grid(True)
    plt.legend(title="Kernel")
    plt.savefig(f"{output_dir}/overlay_latency_distribution_{size}B.png")
    plt.close()

print(f"Individual and overlayed plots saved in: {output_dir}")
