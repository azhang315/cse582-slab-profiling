#!/bin/bash
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "Not running as root"
    exit
fi

sudo perf record -e kmem:kmalloc,kmem:kfree -a -o ./data/tpt/perf.data -- stress-ng --sock 10 --timeout 3s
sudo perf script -i ./data/tpt/perf.data > ./data/perf_script_output.txt

python3 ./plotting/throughput_plot.py ./data/perf_script_output.txt 10
