sudo perf record -e kmem:kmalloc,kmem:kfree -a -- sleep 30
sudo perf script > perf_script_output.txt

python3 latency_plot.py perf_script_output.txt
