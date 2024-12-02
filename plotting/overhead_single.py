import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Constants
PAGE_SIZE = 4096  # Page size in bytes (commonly 4096)

def parse_slabinfo(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[2:]:  # Skip the first two header lines
            fields = re.split(r'\s+', line.strip())
            if len(fields) >= 6:  # Ensure enough fields exist
                try:
                    name = fields[0]
                    active_objs = int(fields[1])
                    total_objs = int(fields[2])
                    obj_size = int(fields[3])
                    obj_per_slab = int(fields[4])
                    pages_per_slab = int(fields[5])

                    # Skip slabs with invalid data
                    if total_objs == 0 or pages_per_slab == 0:
                        continue

                    # Calculate overhead
                    slab_size = pages_per_slab * PAGE_SIZE
                    allocated_per_object = slab_size / obj_per_slab
                    overhead_per_object = allocated_per_object - obj_size
                    overhead_percentage = max(0, (overhead_per_object / obj_size) * 100)

                    data.append({
                        "Cache Name": name,
                        "Active Objects": active_objs,
                        "Total Objects": total_objs,
                        "Object Size": obj_size,
                        "Slab Size": slab_size,
                        "Overhead Per Object": overhead_per_object,
                        "Overhead Percentage": overhead_percentage,
                    })
                except (ValueError, ZeroDivisionError):
                    continue  # Skip malformed or invalid lines
    return pd.DataFrame(data)


# Plot Memory Overhead
def plot_memory_overhead(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cache_name, group in df.groupby("Cache Name"):
        plt.figure()
        plt.bar(group['Cache Name'], group['Overhead Percentage'], color='blue', alpha=0.7)
        plt.xlabel("Cache Name")
        plt.ylabel("Overhead (%)")
        plt.title(f"Memory Overhead for {cache_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"memory_overhead_{cache_name.replace('/', '_')}.png")
        plt.savefig(output_path)
        plt.close()

# Main
def main(normal_file, stress_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing normal execution data...")
    normal_df = parse_slabinfo(normal_file)
    # normal_df = calculate_overhead(normal_df)

    print("Parsing stress execution data...")
    stress_df = parse_slabinfo(stress_file)
    # stress_df = calculate_overhead(stress_df)

    # Plot results
    print("Plotting memory overhead for normal execution...")
    plot_memory_overhead(normal_df, os.path.join(output_dir, "normal"))

    print("Plotting memory overhead for stress execution...")
    plot_memory_overhead(stress_df, os.path.join(output_dir, "stress"))

    # Combined Summary
    combined = pd.concat([normal_df, stress_df])
    summary = combined.groupby("Cache Name").agg({
        "Overhead Percentage": ["mean", "min", "max"]
    }).reset_index()
    summary.columns = ["Cache Name", "Mean Overhead (%)", "Min Overhead (%)", "Max Overhead (%)"]
    summary.to_csv(os.path.join(output_dir, "memory_overhead_summary.csv"), index=False)
    print("Summary saved at:", os.path.join(output_dir, "memory_overhead_summary.csv"))

# Run the script
if __name__ == "__main__":
    normal_file = "./data/overhead/normal_slabinfo.txt"
    stress_file = "./data/overhead/stress_slabinfo.txt"
    output_dir = "./memory_overhead_graphs"
    main(normal_file, stress_file, output_dir)
