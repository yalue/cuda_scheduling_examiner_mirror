# This script is an example for how automatic scenario generation could work.
# It prints a configuration to stdout, which can be piped to ./bin/runner.
#
# For example:
#
#    python multikernel_example.py | ./bin/runner -
#
# The above command will generate result files named
# random_kernels_<number>.json in the results directory.
import json
import random

def generate_multikernel_benchmark(number, min_threads, max_threads,
    min_blocks, max_blocks, min_spin_ns, max_spin_ns, kernel_count):
    """ Returns a dict that, when converted to JSON, will work as a benchmark
    configuration for the multikernel.so benchmark. The multikernel.so
    benchmark will submit a number of kernels given by kernel_count, with
    random thread and block counts in the given ranges (thread counts will
    always be rounded up to the nearest multiple of 32)."""

    kernel_config_string = ""
    kernel_config_list = []
    for i in range(kernel_count):
        to_append = {}
        # Create the kernel name. For example, the kernel named "1_K2" will be
        # kernel 2 of benchmark number 1
        to_append["kernel_label"] = "%d_K%d" % (number, i + 1)
        # Generate a random time for this kernel to spin, in ns
        to_append["duration"] = random.randint(min_spin_ns, max_spin_ns)
        # Generate a random block count for this kernel
        to_append["block_count"] = random.randint(min_blocks, max_blocks)
        # Generate a random thread count, rounded up to a multiple of 32
        thread_count = random.randint(min_threads, max_threads)
        if (thread_count % 32) != 0:
            thread_count += 32 - (thread_count % 32)
        to_append["thread_count"] = thread_count
        kernel_config_list.append(to_append)

    to_return = {}
    to_return["filename"] = "./bin/multikernel.so"
    to_return["log_name"] = "random_kernels_%d.json" % (number)
    to_return["label"] = "Random kernel stream %d" % (number)
    # Only additional_info is used for multikernel.so
    to_return["thread_count"] = 0
    to_return["block_count"] = 0
    to_return["data_size"] = 0
    to_return["additional_info"] = kernel_config_list
    return to_return

def generate_overall_config():
    """ Returns a JSON string which can serve as a benchmark config."""

    stream_count = 4
    kernels_per_stream = 3
    min_blocks = 1
    max_blocks = 3
    min_threads = 128
    max_threads = 1024
    # All kernels will spin for 0.5 seconds
    min_spin_ns = 500000000
    max_spin_ns = 500000000

    # Generate individual benchmark configs
    benchmarks = []
    for i in range(stream_count):
        benchmark = generate_multikernel_benchmark(i + 1, min_threads,
            max_threads, min_blocks, max_blocks, min_spin_ns, max_spin_ns,
            kernels_per_stream)
        benchmarks.append(benchmark)

    # Generate the top-level global config
    config = {}
    config["name"] = "Randomly-generated benchmarks"
    config["max_iterations"] = 1
    config["max_time"] = 0
    config["cuda_device"] = 0
    config["pin_cpus"] = True
    config["benchmarks"] = benchmarks

    # Return the config as a JSON string.
    return json.dumps(config, indent=2)

if __name__ == "__main__":
    print generate_overall_config()
