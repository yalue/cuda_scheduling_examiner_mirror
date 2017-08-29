# This script produces configs that attempt to discern the "scheduling epochs",
# if they exist. Run this, then use the view_blocksbysm.py script to see if
# kernels form clusters at certain times.
import json
import subprocess

def generate_config(kernel_interval, kernel_count, kernel_threads,
    kernel_time):
    """Returns a JSON string containing a benchmark config. The config will
    contain kernel_count kernel releases, separated by kernel_interval in time.
    The kernel_threads argument specifies how many threads each kernel will
    use."""
    benchmark_configs = []
    for i in range(kernel_count):
        benchmark = {
            "label": "Kernel " + str(i),
            "log_name": "epoch_test_%d.json" % (i),
            "filename": "./bin/timer_spin.so",
            "thread_count": kernel_threads,
            "block_count": 1,
            "data_size": 0,
            "additional_info": int(1.0e9 * kernel_time),
            "release_time": float(i) * kernel_interval
        }
        benchmark_configs.append(benchmark)
    overall_config = {
        "name": "Timeline epoch viewer attempt",
        "max_iterations": 1,
        "max_time": 0,
        "cuda_device": 0,
        "pin_cpus": True,
        "benchmarks": benchmark_configs
    }
    return json.dumps(overall_config)

def run_process(kernel_interval, kernel_count, kernel_threads, kernel_time):
    config = generate_config(kernel_interval, kernel_count, kernel_threads,
        kernel_time)
    print "Starting test with an interval of %f seconds." % (kernel_interval)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

run_process(0.1, 20, 128, 0.5)

