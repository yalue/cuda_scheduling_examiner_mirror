# This script produces data for a scatterplot, consisting of a separate run of
# 1000 samples for 1 through 1024 threads. This means that 1,024 output JSON
# files will be produced, so be warned.
import json
import subprocess

def generate_config(thread_count):
    """Returns a JSON string containing a benchmark config. The config will use
    the counter_spin benchmark with a single block and the given number of
    threads. The thread_count therefore can be at most 1024."""
    if (thread_count <= 0) or (thread_count > 1024):
        raise ValueError("The thread count must be at least 1 and under 1024")
    benchmark_config = {
        "label": str(thread_count),
        "log_name": "thread_interference_%d_threads.json" % (thread_count),
        "filename": "./bin/counter_spin.so",
        "thread_count": thread_count,
        "block_count": 1,
        "data_size": 0,
        "additional_info": 1000
    }
    overall_config = {
        "name": "Thread count vs. block duration (single block per kernel)",
        "max_iterations": 1000,
        "max_time": 0,
        "cuda_device": 0,
        "pin_cpus": True,
        "benchmarks": [benchmark_config]
    }
    return json.dumps(overall_config)

def run_process(thread_count):
    """This function will start a benchmark process that will carry out the
    test with the given number of threads."""
    config = generate_config(thread_count)
    print "Starting test with %d threads." % (thread_count)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

for thread_count in range(1, 1025):
    run_process(thread_count)

