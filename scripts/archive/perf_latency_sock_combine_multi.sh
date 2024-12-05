#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Not running as root"
    exit 1
fi

# Set up directories
KERNEL_VERSION=$(uname -r)  # Get the current kernel version
DATA_DIR="/data/latency/${KERNEL_VERSION}"  # Directory specific to the kernel
mkdir -p "$DATA_DIR"

# Drop caches for a cold start
echo "Dropping caches for a cold start..."
echo 3 > /proc/sys/vm/drop_caches

# Run multiple perf recordings
NUM_RUNS=10
for i in $(seq 1 $NUM_RUNS); do
    echo "Starting perf run $i/$NUM_RUNS..."

    # Generate unique filenames for each run
    PERF_DATA_FILE="$DATA_DIR/perf_${i}.data"
    PERF_SCRIPT_FILE="$DATA_DIR/perf_script_output_${i}.txt"

    # Record perf data
    sudo perf record -e kmem:kmalloc,kmem:kfree -a -o "$PERF_DATA_FILE" -- stress-ng --sock 10 --timeout 1s

    # Fix ownership of the perf.data file
    sudo chown $USER:$USER "$PERF_DATA_FILE"

    # Generate perf script output
    perf script -i "$PERF_DATA_FILE" > "$PERF_SCRIPT_FILE"

    echo "Perf run $i/$NUM_RUNS completed: Output saved to $PERF_SCRIPT_FILE"
done

echo "All $NUM_RUNS perf runs completed."

# Optionally, call the preprocessing script for the kernel data
echo "Preprocessing combined data for kernel: $KERNEL_VERSION..."
python3 ./plotting/latency_combine_preprocess_multi.py "$DATA_DIR"
