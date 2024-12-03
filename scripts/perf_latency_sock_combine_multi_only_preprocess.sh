#!/bin/bash

if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Not running as root"
    exit 1
fi

# Set up directories
KERNEL_VERSION=$(uname -r)  # Get the current kernel version
DATA_DIR="/data/latency/${KERNEL_VERSION}"  # Directory specific to the kernel
mkdir -p "$DATA_DIR"

# Optionally, call the preprocessing script for the kernel data
echo "Preprocessing combined data for kernel: $KERNEL_VERSION..."
python3 ./plotting/latency_combine_preprocess_multi.py "$DATA_DIR"
