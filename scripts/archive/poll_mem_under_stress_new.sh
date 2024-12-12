#!/bin/bash

# Ensure the script is run as root
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "This script must be run as root."
    exit 1
fi

# Infer kernel type using uname
KERNEL_TYPE=$(uname -r)

# Define directory structure
BASE_DIR="./data/stress-overhead"
KERNEL_DIR="$BASE_DIR/$KERNEL_TYPE"
NORMAL_DIR="$KERNEL_DIR/normal"
STRESS_DIR="$KERNEL_DIR/stress"

# Parameters for data collection
NORMAL_DURATION=1  # Duration for normal monitoring (seconds)
NORMAL_INTERVAL=0.5  # Interval between each `slabtop` call (seconds)
STRESS_DURATION=60  # Duration for stress monitoring (seconds)
STRESS_INTERVAL=0.5  # Interval between each `slabtop` call (seconds)

# Function to dispatch `slabtop` jobs
dispatch_slabtop_job() {
    local output_dir=$1
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    echo "Dispatching slabtop job at $timestamp to $output_dir"  # Logging
    nice -n -10 slabtop -o > "$output_dir/slabtop_$timestamp.txt" &
}

# Clean the directories before starting
echo "Cleaning up old data..."
rm -rf "$NORMAL_DIR" "$STRESS_DIR"
mkdir -p "$NORMAL_DIR"
mkdir -p "$STRESS_DIR"
echo "Old data cleaned. Ready for fresh data collection."

# Drop caches for a cold start
echo 3 > /proc/sys/vm/drop_caches
echo "Caches dropped."

# Collect slabtop during normal execution
echo "Capturing slabtop during normal execution..."
start_time=$(date +%s)  # Record the start time
while (( $(date +%s) - start_time < NORMAL_DURATION )); do
    dispatch_slabtop_job "$NORMAL_DIR"
    sleep $NORMAL_INTERVAL
done

# Collect slabtop during stress execution
echo "Starting stress-ng workload and capturing slabtop..."
stress-ng --sock 1 --timeout ${STRESS_DURATION}s &

STRESS_PID=$!  # Capture the PID of stress-ng

# Collect slabtop during the stress workload
start_time=$(date +%s)  # Reset the timer
while ps -p $STRESS_PID > /dev/null; do
    if (( $(date +%s) - start_time >= STRESS_DURATION )); then
        break
    fi
    dispatch_slabtop_job "$STRESS_DIR"
    sleep $STRESS_INTERVAL
done

# Wait for stress-ng to complete
wait $STRESS_PID

# Wait for any remaining `slabtop` background jobs to finish
wait

# Completion message
echo "Slabtop monitoring completed."
echo "Data saved in:"
echo "  - Normal: $NORMAL_DIR"
echo "  - Stress: $STRESS_DIR"
