import subprocess
import pandas as pd
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Check for input file
if len(sys.argv) < 2:
    print("Usage: python3 latency_combine_preprocess.py <perf_script_output>")
    sys.exit(1)

# Get kernel version using uname -r
kernel_version = subprocess.check_output("uname -r", shell=True).decode("utf-8").strip()

# Input file and shared directory
input_file = sys.argv[1]
shared_dir = "./shared_data"
os.makedirs(shared_dir, exist_ok=True)

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

# Debug: Check parsed data
if data.empty:
    print("No data was parsed from the input file. Please check the input format.")
    sys.exit(1)

# Separate kmalloc and kfree events
kmalloc_data = data[data["Event"].str.contains("kmem:kmalloc")].copy()
kfree_data = data[data["Event"].str.contains("kmem:kfree")].copy()

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

# Save latency dataframe to the shared directory
output_file = os.path.join(shared_dir, f"latency_data_{kernel_version}.csv")
latency_df.to_csv(output_file, index=False)
print(f"Latency dataframe saved to: {output_file}")
