#!/bin/bash
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Not running as root"
    exit
fi

DATA_DIR="/data/latency" # local to VM
mkdir -p "$DATA_DIR"


echo 3 > /proc/sys/vm/drop_caches # Drop Caches - Cold Start

sudo perf record -e kmem:kmalloc,kmem:kfree -a -o "$DATA_DIR/perf.data" -- stress-ng --sock 10 --timeout 60s

sudo chown $USER:$USER "$DATA_DIR/perf.data" # Fix ownership of the perf.data file

perf script -i "$DATA_DIR/perf.data" > "$DATA_DIR/perf_script_output.txt"

python3 ./plotting/latency_combine_preprocess.py "$DATA_DIR/perf_script_output.txt"
