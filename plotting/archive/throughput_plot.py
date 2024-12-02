import pandas as pd
import sys
import os
import subprocess

# Check for input arguments
if len(sys.argv) < 3:
    print("Usage: python3 throughput_measurement.py <perf_script_output> <benchmark_runtime>")
    sys.exit(1)

# Input file and runtime
input_file = sys.argv[1]
runtime_seconds = float(sys.argv[2])

# Get kernel version using uname -r
kernel_version = subprocess.check_output("uname -r", shell=True).decode("utf-8").strip()
root = f"./graphs/{kernel_version}"

# Create output directory for the kernel version
output_dir = f"{root}/throughput_graphs"
os.makedirs(output_dir, exist_ok=True)

# Parse perf script output
def parse_perf_output(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if "kmem:kmalloc" in line or "kmem:kfree" in line:
                try:
                    parts = line.split()
                    event = parts[4]  # kmem:kmalloc or kmem:kfree
                    ptr = None
                    bytes_req = None
                    if "ptr=" in line:
                        ptr = line.split("ptr=")[1].split()[0]
                    if "bytes_req=" in line:
                        bytes_req = int(line.split("bytes_req=")[1].split()[0])
                    data.append({"Event": event, "Ptr": ptr, "Bytes_Req": bytes_req})
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
    return pd.DataFrame(data)

# Load and parse data
data = parse_perf_output(input_file)

# Check if data is empty
if data.empty:
    print("No data was parsed from the input file. Please check the input format.")
    sys.exit(1)

# Debug: Print unique event types
print("Unique Event Values:\n", data["Event"].unique())

# Define size groups
data["Size Group"] = pd.cut(
    data["Bytes_Req"],
    bins=[0, 64, 256, 1024, 4096, 16384],
    labels=["0-64B", "64-256B", "256-1K", "1K-4K", "4K+"],
    right=False
)

# Debug: Print size group distribution
print("Size Group Distribution:\n", data["Size Group"].value_counts())

# Group events by size and type
size_group_counts = data.groupby(["Size Group", "Event"]).size().unstack(fill_value=0)

# Debug: Print grouped data
print("Grouped Data by Size and Event:\n", size_group_counts)

# Calculate throughput
if "kmem:kmalloc" in size_group_counts.columns:
    size_group_counts["Allocation Throughput (ops/sec)"] = size_group_counts["kmem:kmalloc"] / runtime_seconds
else:
    size_group_counts["Allocation Throughput (ops/sec)"] = 0

if "kmem:kfree" in size_group_counts.columns:
    size_group_counts["Deallocation Throughput (ops/sec)"] = size_group_counts["kmem:kfree"] / runtime_seconds
else:
    size_group_counts["Deallocation Throughput (ops/sec)"] = 0

size_group_counts["Total Throughput (ops/sec)"] = size_group_counts.sum(axis=1) / runtime_seconds

# Print throughput results
print("Throughput by Size Group:")
print(size_group_counts)

# Save to CSV in the output directory
output_file = os.path.join(output_dir, "throughput_by_size_group.csv")
size_group_counts.to_csv(output_file)
print(f"Throughput results saved to: {output_file}")
