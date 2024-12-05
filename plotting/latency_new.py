import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises_2samp

def parse_perf_output(file_path, target_processes=None):
    """Parse perf script output into a structured DataFrame, filtering by process name."""
    data = []
    total_lines = 0
    skipped_lines = 0
    with open(file_path, "r") as f:
        for line in f:
            total_lines += 1
            if "kmem:kmalloc" in line or "kmem:kfree" in line:
                try:
                    # Extract process name (first word in the line)
                    process_name = line.split()[0]

                    # Filter by target processes
                    if target_processes and not any(proc in process_name for proc in target_processes):
                        continue

                    # Parse relevant data
                    parts = line.split()
                    timestamp = float(parts[3].strip(":"))  # Extract timestamp
                    event = parts[4]  # Extract event
                    ptr = None
                    bytes_req = None
                    if "ptr=" in line:
                        ptr = line.split("ptr=")[1].split()[0]
                    if "bytes_req=" in line:
                        bytes_req = int(line.split("bytes_req=")[1].split()[0])
                    data.append({
                        "Process": process_name,
                        "Timestamp (s)": timestamp,
                        "Event": event,
                        "Ptr": ptr,
                        "Bytes_Req": bytes_req,
                    })
                except (ValueError, IndexError) as e:
                    skipped_lines += 1
                    print(f"Skipping malformed line: {line.strip()} (Error: {e})")

    print(f"Parsed {len(data)} events from {file_path} ({total_lines} total lines, {skipped_lines} skipped).")
    return pd.DataFrame(data)





def combine_runs(data_dir, target_processes=None):
    """Combine all runs from a directory into a single DataFrame."""
    combined_data = pd.DataFrame()
    total_runs = 0
    for i, file in enumerate(sorted(os.listdir(data_dir)), start=1):  # Assign Run ID
        if file.startswith("perf_script_output") and file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            print(f"Processing file: {file_path}")
            data = parse_perf_output(file_path, target_processes=target_processes)
            if data.empty:
                print(f"No relevant data in {file_path}. Skipping.")
                continue
            total_runs += 1
            data["Run ID"] = i  # Add Run ID to the DataFrame
            combined_data = pd.concat([combined_data, data], ignore_index=True)

    print(f"Total runs processed: {total_runs}")
    print(f"Combined data size: {len(combined_data)} rows.")
    if combined_data.empty:
        print("No valid data found in any run.")
    return combined_data





def match_kfree_to_kmalloc(kfree_chunk, kmalloc_data):
    """Match kmalloc and kfree events to calculate latencies."""
    results = []
    unmatched_count = 0
    for _, kfree in kfree_chunk.iterrows():
        potential_kmallocs = kmalloc_data[kmalloc_data["Ptr"] == kfree["Ptr"]]
        if not potential_kmallocs.empty:
            earliest_kmalloc = potential_kmallocs.iloc[0]
            # Calculate latency in microseconds
            latency_us = (kfree["Timestamp (s)"] - earliest_kmalloc["Timestamp (s)"]) * 1e6
            results.append({
                "Ptr": kfree["Ptr"],  # Include Ptr for matching
                "Kmalloc Time (s)": earliest_kmalloc["Timestamp (s)"],
                "Latency (µs)": latency_us,
                "Bytes": earliest_kmalloc["Bytes_Req"],
                "Run ID": kfree["Run ID"]
            })
        else:
            unmatched_count += 1

    print(f"Matched {len(results)} kfree events. Unmatched: {unmatched_count}.")
    return results




def filter_outliers(df):
    """Filter out outliers and latencies below 0."""


    print(f"Max latency before filtering: {df['Latency (µs)'].max()}")
    print(f"Avg latency before filtering: {df['Latency (µs)'].mean()}")

    # Remove negative latencies
    df = df[df["Latency (µs)"] >= 0]

    # Remove outliers using IQR
    q1 = df["Latency (µs)"].quantile(0.25)
    q3 = df["Latency (µs)"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    

    filtered_df = df[(df["Latency (µs)"] >= lower_bound) & (df["Latency (µs)"] <= upper_bound)]

    print(f"Max latency after filtering: {filtered_df['Latency (µs)'].max()}")
    print(f"Avg latency after filtering: {filtered_df['Latency (µs)'].mean()}")


    return filtered_df


def process_kernel_data(data_dir, kernel_label, checkpoint_dir="checkpoints"):
    """Process data for a single kernel with checkpointing."""
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    # Checkpoint filenames
    combined_data_file = os.path.join(checkpoint_dir, f"{kernel_label}_combined_data.pkl")
    matched_data_file = os.path.join(checkpoint_dir, f"{kernel_label}_matched_data.pkl")
    filtered_data_file = os.path.join(checkpoint_dir, f"{kernel_label}_filtered_data.pkl")

    # Step 1: Combine Runs (Checkpoint 1)
    if os.path.exists(combined_data_file):
        print(f"Loading combined data checkpoint for {kernel_label}...")
        combined_data = pd.read_pickle(combined_data_file)
    else:
        target_processes = ["stress-ng", "stress-ng-sock"]
        print(f"Combining data from directory: {data_dir}...")
        combined_data = combine_runs(data_dir, target_processes=target_processes)
        if combined_data.empty:
            print(f"No relevant data found in {data_dir}. Skipping.")
            return pd.DataFrame()
        print(f"Saving combined data checkpoint for {kernel_label}...")
        combined_data.to_pickle(combined_data_file)

    # Step 2: Separate and Match Events (Checkpoint 2)
    if os.path.exists(matched_data_file):
        print(f"Loading matched data checkpoint for {kernel_label}...")
        latency_df = pd.read_pickle(matched_data_file)
    else:
        kmalloc_data = combined_data.loc[
            combined_data["Event"].str.contains("kmem:kmalloc"), ["Timestamp (s)", "Ptr", "Bytes_Req", "Run ID"]
        ].copy()
        kfree_data = combined_data.loc[
            combined_data["Event"].str.contains("kmem:kfree"), ["Timestamp (s)", "Ptr", "Run ID"]
        ].copy()

        print(f"Total kmalloc events: {len(kmalloc_data)}")
        print(f"Total kfree events: {len(kfree_data)}")

        if kmalloc_data.empty or kfree_data.empty:
            print(f"No kmalloc or kfree events found in {data_dir}.")
            return pd.DataFrame()

        # Match kmalloc and kfree events in parallel
        num_chunks = min(os.cpu_count(), len(kfree_data))
        kfree_chunks = np.array_split(kfree_data, num_chunks)

        print(f"Matching kmalloc and kfree events for {data_dir} across {num_chunks} chunks...")
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            results = executor.map(match_kfree_to_kmalloc, kfree_chunks, itertools.repeat(kmalloc_data))

        # Combine results into a latency DataFrame
        latencies = [item for sublist in results for item in sublist]
        print(f"Matched latencies: {len(latencies)}")

        if not latencies:
            print("No latencies could be calculated. Returning empty DataFrame.")
            return pd.DataFrame()

        latency_df = pd.DataFrame(latencies)
        latency_df["Kernel"] = kernel_label  # Add kernel label
        

        print(f"Saving matched data checkpoint for {kernel_label}...")
        latency_df.to_pickle(matched_data_file)

    if "Run ID" not in latency_df.columns:
        print("Error: 'Run ID' column missing after matching events.")
    else:
        print(f"'Run ID' column retained. Unique Run IDs: {latency_df['Run ID'].nunique()}")

    # Step 3: Filter Outliers (Checkpoint 3)
    if os.path.exists(filtered_data_file):
        print(f"Loading filtered data checkpoint for {kernel_label}...")
        latency_df = pd.read_pickle(filtered_data_file)
    else:
        print("Filtering outliers and negative latencies...")
        latency_df = filter_outliers(latency_df)
        print(f"Saving filtered data checkpoint for {kernel_label}...")
        latency_df.to_pickle(filtered_data_file)

    # Debug: Final output stats
    print(f"Final latencies for {kernel_label}: {len(latency_df)}")
    print(f"Sample Filtered Latency DataFrame:\n{latency_df.head()}")

    return latency_df





def compute_cvm_test(cdf1, cdf2):
    """
    Computes the Cramér–von Mises test for two distributions.

    Args:
        cdf1: Array-like, first set of data points.
        cdf2: Array-like, second set of data points.

    Returns:
        stat: The CvM test statistic.
        p_value: The p-value for the null hypothesis that the distributions are identical.
    """
    result = cramervonmises_2samp(cdf1, cdf2)
    return result.statistic, result.pvalue


def plot_latency_cdf_with_stats(dataframes, output_dir, specific_sizes):
    """Generate overlayed CDF plots with error bars and add statistical annotations."""
    os.makedirs(output_dir, exist_ok=True)

    # Combine data from all kernels
    combined_data = pd.concat(dataframes, ignore_index=True)

    if combined_data.empty:
        print("No data available for plotting. Skipping.")
        return

    for size in specific_sizes:
        size_data = combined_data[combined_data["Bytes"] == size]
        if size_data.empty:
            print(f"No data available for size {size}B. Skipping.")
            continue

        print(f"\n--- Processing size: {size}B ---")
        plt.figure(figsize=(10, 6))
        colors = itertools.cycle(plt.cm.tab10.colors)  # Color cycle

        stats = []  # Collect statistics for annotations

        for kernel, group in size_data.groupby("Kernel"):
            if "Run ID" not in group.columns:
                print(f"Warning: No 'Run ID' column found for {kernel}. Skipping.")
                continue

            print(f"\nKernel: {kernel}")
            print(f"Total samples (runs): {len(group['Run ID'].unique())}")
            print(f"Total data points: {len(group)}")

            latencies = []

            # Collect CDFs for each run
            for run_id in group["Run ID"].unique():
                run_data = group[group["Run ID"] == run_id]["Latency (µs)"].sort_values()
                cdf = np.linspace(0, 1, len(run_data))
                latencies.append((run_data.values, cdf))

            # Calculate statistics at fixed latencies
            max_latency = max(group["Latency (µs)"])
            fixed_latencies = np.linspace(0, max_latency, 100)  # 100 fixed points
            interval_size = max_latency / (len(fixed_latencies) - 1)
            print(f"Fixed latency intervals: {len(fixed_latencies)} points")
            print(f"Interval size: {interval_size:.2f} µs")

            cdf_stats = []
            for latency in fixed_latencies:
                cdf_values = [
                    np.interp(latency, run_latencies, run_cdf)
                    for run_latencies, run_cdf in latencies
                ]
                mean_cdf = np.mean(cdf_values)
                std_cdf = np.std(cdf_values)
                cdf_stats.append((latency, mean_cdf, std_cdf))

            # Extract stats for plotting
            cdf_mean = [stat[1] for stat in cdf_stats]
            cdf_std = [stat[2] for stat in cdf_stats]
            latency_values = [stat[0] for stat in cdf_stats]

            # Plot CDF with error bars
            color = next(colors)
            plt.plot(latency_values, cdf_mean, label=f"{kernel} (Samples: {len(group['Run ID'].unique())})", color=color)
            plt.fill_between(
                latency_values,
                np.clip(np.array(cdf_mean) - np.array(cdf_std), 0, 1),
                np.clip(np.array(cdf_mean) + np.array(cdf_std), 0, 1),
                color=color,
                alpha=0.1,  # Adjust transparency
                label=f"{kernel} ± 1 SD",
            )

            print(f"{kernel}: Fixed Latency Analysis Complete")

            # Collect for statistical comparison
            stats.append((kernel, np.array(cdf_mean)))

        # Statistical comparison between kernels
        if len(stats) == 2:
            # # Extract the full CDF for each kernel
            # cdf1 = np.concatenate([run[1] for run in latencies if run[1].size > 0])
            # cdf2 = np.concatenate([run[1] for run in latencies if run[1].size > 0])
            # # Compute CvM test
            # stat, p_value = compute_cvm_test(cdf1, cdf2)
            # print(f"{kernel}: CvM Statistic = {stat}, p-value = {p_value}")
            # # Add CvM statistic and p-value to the plot annotation
            # plt.annotate(f"CvM Statistic: {stat:.4f}\nCvM p-value: {p_value:.4f}",
            #             xy=(0.7, 0.2), xycoords='axes fraction', fontsize=10)
            kernel1, cdf1 = stats[0]
            kernel2, cdf2 = stats[1]
            # p_value = ks_2samp(cdf1, cdf2).pvalue
            stat, p_value = compute_cvm_test(cdf1, cdf2)


            avg_diff = np.mean(np.abs(cdf1 - cdf2)) * 100  # % difference

            # latency_values_kernel1 = np.concatenate([run[0] for run in latencies if run[1].size > 0])
            # latency_values_kernel2 = np.concatenate([run[0] for run in latencies if run[1].size > 0])
            # # Horizontal differences at fixed CDF points
            # fixed_cdf_points = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
            # latencies_kernel1 = np.interp(fixed_cdf_points, cdf1, latency_values_kernel1)
            # latencies_kernel2 = np.interp(fixed_cdf_points, cdf2, latency_values_kernel2)
            # horizontal_differences = np.abs(latencies_kernel1 - latencies_kernel2)
            # latency_diff_us = np.mean(horizontal_differences)  # Average horizontal latency difference


            avg_points_per_sample = (
                group.groupby("Run ID")["Latency (µs)"].count().mean()
            )

            # Add annotations
            stats_text = (
                f"Statistical Comparison ({size}B):\n"
                f"CvM Statistic: {stat:.4f}\n"
                f"CvM p-value: {p_value:.4f}\n"
                f"Avg % Diff: {avg_diff:.2f}%\n"
                f"Avg. points per sample: {avg_points_per_sample:.1f}"
            )
            plt.gca().text(
                0.95, 0.05, stats_text,
                transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
            )

        # Configure plot
        plt.xlabel("Latency (µs)")
        plt.ylabel("CDF")
        plt.title(f"CDF of Allocation Latency (Overlayed, {size}B)")
        plt.legend(title="Kernel", loc="upper left")
        plt.grid(True)

        # Save plot
        plot_filename = os.path.join(output_dir, f"overlay_latency_cdf_{size}B_with_error.png")
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()







def main(data_dirs, output_dir):
    """Main function for processing and plotting data."""
    specific_sizes = [32, 4096]  # Adjust these as needed
    latency_dfs = []

    for i, data_dir in enumerate(data_dirs, start=1):
        kernel_label = f"Kernel {i}"
        print(f"Processing data for {kernel_label} from {data_dir}...")


        # if i == 1:
        #     kernel_label = "Native-C"
        # elif i == 2:
        #     kernel_label = "Port-R"
        # else:
        #     print(f"More than two kernels not supported")
        #     return

        kernel_data = process_kernel_data(data_dir, kernel_label)
        if not kernel_data.empty:
            latency_dfs.append(kernel_data)
    

    if not latency_dfs:
        print("No valid data found for any kernel. Exiting.")
        sys.exit(1)

    # Plot CDFs for the collected data
    # plot_latency_cdf(latency_dfs, output_dir, specific_sizes)
    plot_latency_cdf_with_stats(latency_dfs, output_dir, specific_sizes)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 latency_new.py <raw_dir_kernel_1> <raw_dir_kernel_2> <output_dir>")
        sys.exit(1)

    data_dirs = sys.argv[1:-1]
    output_dir = sys.argv[-1]
    main(data_dirs, output_dir)
