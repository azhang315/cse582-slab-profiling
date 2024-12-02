#!/bin/bash

# Ensure the script is run as root
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "This script must be run as root."
    exit 1
fi

# Define directory structure
BASE_DIR="./data/overhead"
NORMAL_DIR="$BASE_DIR/normal"
STRESS_DIR="$BASE_DIR/stress"

# Create necessary directories
mkdir -p "$NORMAL_DIR"
mkdir -p "$STRESS_DIR"

# Collect slabinfo during normal execution
echo "Capturing slabinfo during normal execution..."
NORMAL_DURATION=30  # 60 seconds
NORMAL_INTERVAL=0.5  # Every 0.5 seconds
END_TIME=$((SECONDS + NORMAL_DURATION))
while [ $SECONDS -lt $END_TIME ]; do
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    cat /proc/slabinfo > "$NORMAL_DIR/slabinfo_$TIMESTAMP.txt"
    sleep $NORMAL_INTERVAL
done

# Collect slabinfo during stress execution
echo "Starting stress-ng workload and capturing slabinfo..."
STRESS_DURATION=30  # Match stress-ng timeout
STRESS_INTERVAL=0.5  # Every 0.5 seconds
# stress-ng --vm 10 --vm-bytes 50% --timeout ${STRESS_DURATION}s &  # Adjust workload as needed
stress-ng --sock 50 --timeout ${STRESS_DURATION}s &

STRESS_PID=$!  # Capture the PID of stress-ng

# Collect slabinfo during the stress workload
while ps -p $STRESS_PID > /dev/null; do
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    cat /proc/slabinfo > "$STRESS_DIR/slabinfo_$TIMESTAMP.txt"
    sleep $STRESS_INTERVAL
done

# Wait for stress-ng to complete
wait $STRESS_PID

echo "Slabinfo monitoring completed."
echo "Data saved in:"
echo "  - Normal: $NORMAL_DIR"
echo "  - Stress: $STRESS_DIR"
