#!/bin/bash

# Configurable variables
OUTPUT_DIR="perf_data"
STRESS_CMD="stress-ng --malloc 100 --malloc-bytes 4K --timeout 10s"
EVENTS="kmem:kmalloc,kmem:kfree"  # Perf events to monitor
RECORD_TIME=30  # Duration to record perf events
GRAPH_SCRIPT="perf_graph.py"  # Python script for graphing

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Run the workload and record perf events
echo "Starting stress workload: $STRESS_CMD"
echo "Recording perf events: $EVENTS for $RECORD_TIME seconds"

sudo perf record -e $EVENTS -a -- sleep $RECORD_TIME
sudo perf script > "$OUTPUT_DIR/perf_script.txt"

# Step 2: Parse and prepare the data for visualization
echo "Parsing perf data into CSV format"
awk '
    /kmem:kmalloc/ { print $2 "," "kmalloc" "," $8 }
    /kmem:kfree/ { print $2 "," "kfree" "," $8 }
' "$OUTPUT_DIR/perf_script.txt" > "$OUTPUT_DIR/perf_data.csv"

# Step 3: Run Python script to graph the results
echo "Graphing results with Python..."
python3 "$GRAPH_SCRIPT" "$OUTPUT_DIR/perf_data.csv"

echo "All done! Graphs saved in the output directory: $OUTPUT_DIR"

