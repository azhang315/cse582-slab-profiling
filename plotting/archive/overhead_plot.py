import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Constants
PAGE_SIZE = 4096  # Page size in bytes (typically 4096)

# Relevant slab caches for overhead analysis
RELEVANT_CACHES = ["kmalloc", "kmem_cache", "dma-kmalloc"]

# Parse /proc/slabinfo
def parse_slabinfo(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[2:]:  # Skip the first two header lines
            fields = re.split(r'\s+', line.strip())
            if len(fields) >= 5:
                try:
                    name = fields[0]
                    active_objs = int(fields[1])
                    total_objs = int(fields[2])
                    obj_size = int(fields[3])
                    pages_per_slab = int(fields[4])  # Slab size in pages
                    slab_size = pages_per_slab * PAGE_SIZE  # Convert to bytes
                    if total_objs > 0:  # Skip slabs with zero objects
                        data.append({
                            "Timestamp": int(filepath.split('_')[-1].split('.')[0]),  # Extract timestamp
                            "Cache Name": name,
                            "Active Objects": active_objs,
                            "Total Objects": total_objs,
                            "Object Size": obj_size,
                            "Slab Size": slab_size
                        })
                except ValueError:
                    continue  # Skip malformed lines
    return pd.DataFrame(data)

def calculate_overhead(df):
    df = df[df['Total Objects'] > 0]  # Ensure valid total objects
    df['Slab Size (Bytes)'] = df['Slab Size']  # Already converted to bytes during parsing

    # Calculate the memory allocated per object
    df['Allocated Per Object'] = df['Slab Size (Bytes)'] / df['Total Objects']

    # Calculate the overhead per object
    df['Overhead Per Object'] = df['Allocated Per Object'] - df['Object Size']

    # Normalize the overhead percentage
    df['Overhead Percentage'] = (df['Overhead Per Object'] / df['Object Size']) * 100

    # Clamp values to [0, 100]
    df['Overhead Percentage'] = df['Overhead Percentage'].clip(lower=0, upper=100)

    return df

# Filter relevant caches
def filter_relevant_caches(df):
    return df[df["Cache Name"].str.contains("|".join(RELEVANT_CACHES), na=False)]

# Aggregate data over time
def aggregate_slabinfo(directory):
    all_data = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".txt"):
            filepath = os.path.join(directory, file)
            slabinfo_df = parse_slabinfo(filepath)
            all_data.append(slabinfo_df)
    return pd.concat(all_data, ignore_index=True)

# Plot overhead over time for a single cache
def plot_cache_overhead(cache_name, group, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(group['Time'], group['Overhead Percentage'], marker='o', label="Overhead (%)", color='blue')
    plt.xlabel("Time")
    plt.ylabel("Overhead (%)")
    plt.title(f"Overhead Over Time - {cache_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{cache_name.replace('/', '_')}_overhead.png")
    plt.savefig(output_path)
    plt.close()

# Main
def main(normal_dir, stress_dir):
    print("Aggregating normal execution data...")
    normal_df = aggregate_slabinfo(normal_dir)
    normal_df = filter_relevant_caches(normal_df)
    normal_df = calculate_overhead(normal_df)
    normal_df['Time'] = pd.to_datetime(normal_df['Timestamp'], unit='s')

    print("Aggregating stress execution data...")
    stress_df = aggregate_slabinfo(stress_dir)
    stress_df = filter_relevant_caches(stress_df)
    stress_df = calculate_overhead(stress_df)
    stress_df['Time'] = pd.to_datetime(stress_df['Timestamp'], unit='s')

    # Create output directories
    normal_output_dir = "slab_overhead_graphs/normal"
    stress_output_dir = "slab_overhead_graphs/stress"
    os.makedirs(normal_output_dir, exist_ok=True)
    os.makedirs(stress_output_dir, exist_ok=True)

    print("Plotting normal execution overhead...")
    for cache_name, group in normal_df.groupby("Cache Name"):
        plot_cache_overhead(cache_name, group, normal_output_dir)

    print("Plotting stress execution overhead...")
    for cache_name, group in stress_df.groupby("Cache Name"):
        plot_cache_overhead(cache_name, group, stress_output_dir)

    # Combined Summary
    combined = pd.concat([normal_df, stress_df])
    summary = combined.groupby("Cache Name").agg({
        "Overhead Percentage": ["mean", "min", "max"]
    }).reset_index()
    summary.columns = ["Cache Name", "Mean Overhead (%)", "Min Overhead (%)", "Max Overhead (%)"]

    print("Summary of overhead:")
    print(summary)

    # Save the summary
    summary.to_csv("slab_overhead_graphs/summary.csv", index=False)

# Run script
if __name__ == "__main__":
    normal_dir = "slabinfo_data/normal"
    stress_dir = "slabinfo_data/stress"
    main(normal_dir, stress_dir)
