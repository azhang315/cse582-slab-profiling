#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Not running as root"
    exit 1
fi

# Set up directories
TYPE="aio"
KERNEL_VERSION=$(uname -r)
DATA_DIR="/data/latency/${KERNEL_VERSION}"
RAW_DATA_DIR="$DATA_DIR/$TYPE/raw"
LOCAL_RAW_DATA_DIR="/data/tmp_latency/${KERNEL_VERSION}/$TYPE/raw"  # Local VM temp dir
MOUNT_DATA_DIR="./data/latency/${KERNEL_VERSION}/$TYPE/raw"        # Mount directory for final data storage

mkdir -p "$RAW_DATA_DIR"
mkdir -p "$LOCAL_RAW_DATA_DIR"
mkdir -p "$MOUNT_DATA_DIR"

echo "Dropping caches for a cold start..."
echo 3 > /proc/sys/vm/drop_caches

NUM_RUNS=100

for i in $(seq 1 $NUM_RUNS); do
    LOCAL_PERF_DATA_FILE="$LOCAL_RAW_DATA_DIR/perf_${i}.data"
    LOCAL_PERF_SCRIPT_FILE="$LOCAL_RAW_DATA_DIR/perf_script_output_${i}.txt"
    MOUNT_PERF_SCRIPT_FILE="$MOUNT_DATA_DIR/perf_script_output_${i}.txt"

    echo "Starting perf run $i/$NUM_RUNS..."
    sudo perf record -e kmem:kmalloc,kmem:kfree -a -o "$LOCAL_PERF_DATA_FILE" -- stress-ng --$TYPE 3 --timeout 1s

    if [[ ! -s "$LOCAL_PERF_DATA_FILE" ]]; then
        echo "perf.data is empty for run $i. Skipping script output generation."
        continue
    fi

    sudo chown $USER:$USER "$LOCAL_PERF_DATA_FILE"
    perf script -i "$LOCAL_PERF_DATA_FILE" | head -n 1000 > "$LOCAL_PERF_SCRIPT_FILE" # FILTER


    if [[ ! -s "$LOCAL_PERF_SCRIPT_FILE" ]]; then
        echo "perf_script_output_${i}.txt is empty. Check your workload or kernel events."
    else
        echo "Perf run $i/$NUM_RUNS completed: Output saved locally to $LOCAL_PERF_SCRIPT_FILE"
    fi

    # Copy files to the mount directory
    echo "Copying results to mount directory: $MOUNT_DATA_DIR"
    cp "$LOCAL_PERF_SCRIPT_FILE" "$MOUNT_PERF_SCRIPT_FILE"
done

echo "All runs completed. Data has been saved to $MOUNT_DATA_DIR"
