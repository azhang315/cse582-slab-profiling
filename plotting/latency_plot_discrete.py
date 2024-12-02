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

# Function to filter outliers using IQR
def filter_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Handle old output files by moving them to an "old" directory
def manage_old_outputs(output_dir):
    old_output_dir = os.path.join(output_dir, "old")
    if os.path.exists(output_dir):
        # Create "old" directory if it doesn't exist
        os.makedirs(old_output_dir, exist_ok=True)
        # Move all existing files to the "old" directory
        for file in os.listdir(output_dir):
            if file != "old":  # Skip the "old" directory itself
                old_path = os.path.join(output_dir, file)
                new_path = os.path.join(old_output_dir, file)
                os.rename(old_path, new_path)
    else:
        # If output directory doesn't exist, create it
        os.makedirs(output_dir, exist_ok=True)


# Input file and output directory
input_file = sys.argv[1]
output_dir = f"{root}/latency_graphs"

# Clear previous outputs
manage_old_outputs(output_dir)


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


# Parse perf script output
def parse_perf_output(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if "kmem:kmalloc" in line or "kmem:kfree" in line:
                try:
                    parts = line.split()
                    timestamp = float(parts[3].strip(":"))  # Extract timestamp (in seconds)
                    event = parts[4]  # Extract event
                    ptr = None
                    bytes_req = None
                    if "ptr=" in line:
                        ptr = line.split("ptr=")[1].split()[0]
                    if "bytes_req=" in line:
                        bytes_req = int(line.split("bytes_req=")[1].split()[0])
                    data.append({"Timestamp (s)": timestamp, "Event": event, "Ptr": ptr, "Bytes_Req": bytes_req})
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
            # Calculate latency in microseconds
            latency_us = (kfree["Timestamp (s)"] - earliest_kmalloc["Timestamp (s)"]) * 1e6
            results.append({
                "Kmalloc Time (s)": earliest_kmalloc["Timestamp (s)"],
                "Latency (µs)": latency_us,
                "Bytes": earliest_kmalloc["Bytes_Req"],
            })
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

# Debug: Print first few rows of the latency dataframe
print("Latency Data (first 5 rows):")
print(latency_df.head())

# Define the specific sizes to inspect
specific_sizes = [32, 4096]  # Only process 32B and 4096B allocations

# Filter dataset for specific allocation sizes
filtered_data = latency_df[latency_df["Bytes"].isin(specific_sizes)]

# Debug: Check size distribution
print("Filtered Data for Specific Sizes:\n", filtered_data["Bytes"].value_counts())

# Generate plots for specific sizes
colors = plt.cm.tab10.colors  # Use consistent colors for plots
for i, size in enumerate(specific_sizes):
    size_data = filtered_data[filtered_data["Bytes"] == size]

    if size_data.empty:
        print(f"No data available for {size}B allocations.")
        continue

    # Filter outliers
    filtered_subset = filter_outliers(size_data, "Latency (µs)")

    # Summary Statistics
    latency_summary = filtered_subset["Latency (µs)"].describe(percentiles=[0.99])
    sample_count = len(filtered_subset)
    print(f"Latency Summary for {size}B Allocations:\n", latency_summary)

    # Scatter Plot (Allocation Latency Over Time)
    plt.figure()
    ax = plt.gca()
    plt.scatter(filtered_subset["Kmalloc Time (s)"], filtered_subset["Latency (µs)"], alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel("Kmalloc Timestamp (s)")
    plt.ylabel("Latency (µs)")
    plt.title(f"Allocation Latency Over Time ({size}B Allocations)")
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/allocation_latency_{size}B.png")
    plt.close()

    # Latency Distribution (Histogram)
    plt.figure()
    ax = plt.gca()
    plt.hist(filtered_subset["Latency (µs)"], bins=30, alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel("Latency (µs)")
    plt.ylabel("Frequency")
    plt.title(f"Latency Distribution ({size}B Allocations)")
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/latency_distribution_{size}B.png")
    plt.close()

    # CDF Plot
    plt.figure()
    ax = plt.gca()
    latency_sorted = filtered_subset["Latency (µs)"].sort_values()
    cdf = latency_sorted.rank(method="average", pct=True)
    plt.plot(latency_sorted, cdf, marker=".", linestyle="-", alpha=0.8, color=colors[i % len(colors)])
    plt.xlabel("Latency (µs)")
    plt.ylabel("CDF")
    plt.title(f"CDF of Allocation Latency ({size}B Allocations)")
    plt.grid(True)
    add_sample_count(ax, sample_count)  # Add sample count annotation
    plt.savefig(f"{output_dir}/allocation_latency_cdf_{size}B.png")
    plt.close()

print("Plots for 32B and 4096B allocations saved in:", output_dir)
