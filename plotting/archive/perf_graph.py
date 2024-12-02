import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Check for input file
if len(sys.argv) < 2:
    print("Usage: python3 perf_plot.py <perf_script_output>")
    sys.exit(1)

# Input file and output directory
input_file = sys.argv[1]
output_dir = "perf_graphs"
os.makedirs(output_dir, exist_ok=True)

# Function to parse perf script output
def parse_perf_output(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    timestamp = float(parts[0])
                    event = parts[1]
                    bytes_req = 0
                    if len(parts) == 3 and "bytes_req=" in parts[2]:
                        bytes_req = int(parts[2].split("=")[-1])
                    data.append({"Timestamp": timestamp, "Event": event, "Bytes": bytes_req})
                except ValueError:
                    continue
    return pd.DataFrame(data)

# Load and parse data
data = parse_perf_output(input_file)
data["Relative Time"] = data["Timestamp"] - data["Timestamp"].min()

# Separate kmalloc and kfree data
kmalloc_data = data[data["Event"] == "kmalloc"].copy()
kfree_data = data[data["Event"] == "kfree"].copy()

# Add "Matched" column explicitly using .loc
kmalloc_data.loc[:, "Matched"] = False
kfree_data.loc[:, "Matched"] = False

# Match kmalloc and kfree by relative time
matched_pairs = []

for i, kmalloc in kmalloc_data.iterrows():
    # Filter unmatched kfree rows
    potential_frees = kfree_data[~kfree_data["Matched"]]
    if not potential_frees.empty:
        closest_idx = (potential_frees["Timestamp"] - kmalloc["Timestamp"]).abs().idxmin()
        closest_free = potential_frees.loc[closest_idx]
        matched_pairs.append((kmalloc["Relative Time"], closest_free["Relative Time"], kmalloc["Bytes"]))
        kfree_data.loc[closest_idx, "Matched"] = True
        kmalloc_data.loc[i, "Matched"] = True

# Convert matches to DataFrame
matches = pd.DataFrame(matched_pairs, columns=["Kmalloc Time", "Kfree Time", "Bytes"])

# Plot 1: Kmalloc and Kfree with matching
plt.figure()
plt.scatter(matches["Kmalloc Time"], matches["Bytes"], label="kmalloc (bytes)", alpha=0.6)
plt.scatter(matches["Kfree Time"], matches["Bytes"], label="kfree (bytes)", alpha=0.6)
plt.xlabel("Time (relative)")
plt.ylabel("Bytes")
plt.title("Kmalloc and Kfree Over Time (Matched)")
plt.legend()
plt.savefig(f"{output_dir}/kmalloc_kfree_matched.png")
plt.close()

# Plot 2: Histogram of kmalloc sizes
plt.figure()
plt.hist(kmalloc_data["Bytes"], bins=30, alpha=0.7, label="kmalloc (bytes)")
plt.xlabel("Bytes Allocated")
plt.ylabel("Frequency")
plt.title("Distribution of Kmalloc Allocation Sizes")
plt.legend()
plt.savefig(f"{output_dir}/kmalloc_histogram.png")
plt.close()

# Plot 3: Cumulative kmalloc bytes
kmalloc_data["Cumulative Bytes"] = kmalloc_data["Bytes"].cumsum()
plt.figure()
plt.plot(kmalloc_data["Relative Time"], kmalloc_data["Cumulative Bytes"], label="Cumulative kmalloc (bytes)")
plt.xlabel("Time (relative)")
plt.ylabel("Cumulative Bytes Allocated")
plt.title("Cumulative Kmalloc Allocations Over Time")
plt.legend()
plt.savefig(f"{output_dir}/kmalloc_cumulative.png")
plt.close()

print(f"Graphs saved in {output_dir}")
