import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Check for input file
if len(sys.argv) < 2:
    print("Usage: python3 latency_ptr_match.py <perf_script_output>")
    sys.exit(1)

# Get kernel version using uname -r
kernel_version = subprocess.check_output("uname -r", shell=True).decode("utf-8").strip()
root = f"./graphs/{kernel_version}"

# Input file and output directory
input_file = sys.argv[1]
output_dir = f"{root}/latency_graphs"
os.makedirs(output_dir, exist_ok=True)

# Parse perf script output
def parse_perf_output(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if "kmem:kmalloc" in line or "kmem:kfree" in line:
                try:
                    parts = line.split()
                    timestamp = float(parts[3].strip(":"))  # Extract timestamp
                    event = parts[4]  # Extract event
                    ptr = None
                    bytes_req = None
                    if "ptr=" in line:
                        ptr = line.split("ptr=")[1].split()[0]
                    if "bytes_req=" in line:
                        bytes_req = int(line.split("bytes_req=")[1].split()[0])
                    data.append({"Timestamp": timestamp, "Event": event, "Ptr": ptr, "Bytes_Req": bytes_req})
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
    return pd.DataFrame(data)

# Load and parse data
data = parse_perf_output(input_file)

# Debug: Print first few rows
if data.empty:
    print("No data was parsed from the input file. Please check the input format.")
    sys.exit(1)

print("Parsed Data (first 5 rows):")
print(data.head())

# Separate kmalloc and kfree events
kmalloc_data = data[data["Event"].str.contains("kmem:kmalloc")].copy()
kfree_data = data[data["Event"].str.contains("kmem:kfree")].copy()

if kmalloc_data.empty or kfree_data.empty:
    print("No kmalloc or kfree events found in the input data.")
    sys.exit(1)

# Match kmalloc and kfree in parallel
def match_kfree_to_kmalloc(kfree_chunk):
    results = []
    for i, kfree in kfree_chunk.iterrows():
        potential_kmallocs = kmalloc_data[kmalloc_data["Ptr"] == kfree["Ptr"]]
        if not potential_kmallocs.empty:
            earliest_kmalloc = potential_kmallocs.iloc[0]
            latency = (kfree["Timestamp"] - earliest_kmalloc["Timestamp"]) * 1e6  # Convert to microseconds
            results.append({"Kmalloc Time": earliest_kmalloc["Timestamp"], "Latency": latency, "Bytes": earliest_kmalloc["Bytes_Req"]})
    return results

# Split kfree_data into chunks for parallel processing
num_chunks = os.cpu_count()
kfree_chunks = np.array_split(kfree_data, num_chunks)

# Process chunks in parallel
with ProcessPoolExecutor(max_workers=num_chunks) as executor:
    results = executor.map(match_kfree_to_kmalloc, kfree_chunks)

# Combine results
latencies = [item for sublist in results for item in sublist]
latency_df = pd.DataFrame(latencies)

if latency_df.empty:
    print("No latency data could be calculated. Check the event matching.")
    sys.exit(1)

# Add Size Ranges for Stratification
latency_df["Size Group"] = pd.cut(
    latency_df["Bytes"],
    bins=[0, 64, 256, 1024, 4096, float("inf")],
    labels=["0-64B", "64-256B", "256-1K", "1K-4K", "4K+"],
    right=False
)

# Debug: Print size group stats
print("Size Group Distribution:\n", latency_df["Size Group"].value_counts())

# Function to filter outliers using IQR
def filter_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Function to add number of samples to the plot
def add_sample_count(ax, sample_count):
    ax.annotate(
        f"Samples: {sample_count}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )

# Generate separate plots for each size group
size_groups = latency_df["Size Group"].unique()
colors = plt.cm.tab10.colors  # Use a color map for consistent, distinguishable colors

for i, size_group in enumerate(size_groups):
    subset = latency_df[latency_df["Size Group"] == size_group]
    
    if subset.empty:
        continue

    # Filter outliers
    filtered_subset = filter_outliers(subset, "Latency")

    # Summary Statistics
    latency_summary = filtered_subset["Latency"].describe(percentiles=[0.99])
    print(f"Latency Summary Statistics for {size_group}:\n", latency_summary)

    sample_count = len(filtered_subset)

    # Allocation Latency Over Time
    plt.figure()
    ax = plt.gca()
    plt.scatter(filtered_subset["Kmalloc Time"], filtered_subset["Latency"], alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel("Kmalloc Timestamp")
    plt.ylabel("Latency (µs)")
    plt.title(f"Allocation Latency Over Time ({size_group})")
    plt.ylim([0, filtered_subset["Latency"].max() * 1.1])  # Consistent scaling
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/allocation_latency_{size_group}.png")
    plt.close()

    # Latency Distribution
    plt.figure()
    ax = plt.gca()
    plt.hist(filtered_subset["Latency"], bins=30, alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel("Latency (µs)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Allocation Latency ({size_group})")
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/latency_distribution_{size_group}.png")
    plt.close()

    # CDF Plot
    plt.figure()
    ax = plt.gca()
    latency_sorted = filtered_subset["Latency"].sort_values()
    cdf = latency_sorted.rank(method="average", pct=True)
    plt.plot(latency_sorted, cdf, marker=".", linestyle="-", alpha=0.8, color=colors[i % len(colors)])
    plt.xlabel("Latency (µs)")
    plt.ylabel("CDF")
    plt.title(f"Cumulative Distribution of Allocation Latency ({size_group})")
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/allocation_latency_cdf_{size_group}.png")
    plt.close()

    # Annotated Latency Distribution
    plt.figure()
    ax = plt.gca()
    plt.hist(filtered_subset["Latency"], bins=30, alpha=0.7, color=colors[i % len(colors)])
    plt.axvline(latency_summary["mean"], color="r", linestyle="--", label=f"Mean: {latency_summary['mean']:.2f} µs")
    plt.axvline(latency_summary["50%"], color="g", linestyle="--", label=f"Median: {latency_summary['50%']:.2f} µs")
    plt.axvline(latency_summary["99%"], color="b", linestyle="--", label=f"99th Percentile: {latency_summary['99%']:.2f} µs")
    plt.axvline(latency_summary["max"], color="k", linestyle="--", label=f"Max: {latency_summary['max']:.2f} µs")
    plt.xlabel("Latency (µs)")
    plt.ylabel("Frequency")
    plt.title(f"Annotated Latency Distribution ({size_group})")
    plt.legend()
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/annotated_latency_distribution_{size_group}.png") 
    plt.close()

print("Latency Graphs saved in:", output_dir)