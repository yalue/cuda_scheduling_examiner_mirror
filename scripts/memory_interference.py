import json
from subprocess import Popen, PIPE

def generate_config(thread_count, block_count, data_size, access_count):
    """Returns a JSON string containing a benchmark config for the given number
    of threads, data size, and access count."""
    benchmark_config = {
        "label": "%dx%d threads" % (thread_count, block_count),
        "log_name": "random_walk_%d_%d_%d_%d.json" % (thread_count,
            block_count, data_size, access_count),
        "filename": "./bin/random_walk.so",
        "thread_count": thread_count,
        "block_count": block_count,
        "data_size": data_size,
        "additional_info": access_count
    }
    overall_config = {
        "name": "Random walk thread comparison, %d byte buffer" % (data_size),
        "max_iterations": 2000,
        "max_time": 0,
        "cuda_device": 0,
        "pin_cpus": True,
        "benchmarks": [benchmark_config]
    }
    return json.dumps(overall_config)

def start_process(thread_count = 32, block_count = 1, data_size = 8192,
        access_count = 1000):
    config = generate_config(thread_count, block_count, data_size,
        access_count)
    print "Starting proccess with %dx%d threads, %d byte buffers..." % (
        thread_count, block_count, data_size)
    process = Popen(["./bin/runner", "-"], stdin=PIPE)
    process.communicate(input=config)

def run_thread_tests(buffer_size):
    for i in range(4):
        # Have a very small number of threads and a bigger number of threads.
        thread_count = (i + 1)
        start_process(thread_count, data_size = buffer_size)
        thread_count = (i + 1) * 8
        start_process(thread_count, data_size = buffer_size)

def run_buffer_tests():
    page_size = 4096
    buffer_sizes = [page_size, 4 * page_size, 1024 * 1024 * 4]
    for i in range(4):
        buffer_size = (i + 1) * 2048
        print "Running tests with %dk buffers..." % (buffer_size / 1024)
        run_thread_tests(buffer_size)

run_buffer_tests()
