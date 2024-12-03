import itertools
import os
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

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
                    data.append({"Timestamp (s)": timestamp, "Event": event, "Ptr": ptr, "Bytes_Req": bytes_req})
                except (ValueError, IndexError) as e:
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")
    return pd.DataFrame(data)

# Combine data from all runs
def combine_runs(data_dir):
    combined_data = pd.DataFrame()

    # Iterate through all perf_script_output files
    for file in sorted(os.listdir(data_dir)):
        if file.startswith("perf_script_output") and file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            print(f"Processing file: {file_path}")
            data = parse_perf_output(file_path)  # Parse each run
            combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data

# Match kmalloc and kfree events to calculate latency
def match_kfree_to_kmalloc(kfree_chunk, kmalloc_data):
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

if len(sys.argv) < 2:
    print("Usage: python3 latency_combine_preprocess.py <directory>")
    sys.exit(1)

data_dir = sys.argv[1]
output_dir = "./data/latency/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "combined_data.csv")

print(data_dir)
# Combine all data
combined_data = combine_runs(data_dir)

if combined_data.empty:
    print("No data was parsed. Exiting.")
    sys.exit(1)

# Separate kmalloc and kfree events
kmalloc_data = combined_data[combined_data["Event"].str.contains("kmem:kmalloc")].copy()
kfree_data = combined_data[combined_data["Event"].str.contains("kmem:kfree")].copy()

if kmalloc_data.empty or kfree_data.empty:
    print("No kmalloc or kfree events found in the input data.")
    sys.exit(1)

# Match kmalloc and kfree events in parallel
num_chunks = os.cpu_count()
kfree_chunks = np.array_split(kfree_data, num_chunks)

print("Matching kmalloc and kfree events...")
with ProcessPoolExecutor(max_workers=num_chunks) as executor:
    results = executor.map(match_kfree_to_kmalloc, kfree_chunks, itertools.repeat(kmalloc_data))

# Combine results into a latency DataFrame
latencies = [item for sublist in results for item in sublist]
latency_df = pd.DataFrame(latencies)

if latency_df.empty:
    print("No latency data could be calculated. Check the event matching.")
    sys.exit(1)

# Save the latency DataFrame
latency_df.to_csv(output_file, index=False)
print(f"Combined latency data saved to: {output_file}")
