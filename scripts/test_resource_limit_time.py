import json
from subprocess import Popen, PIPE

def generate_config(mps_ratio):
    """ Creates a config with two benchmarks: worker and thrasher. Worker will
    be limited to the given GPU utilization value (between 0 and 1). Thrasher
    will be a random-walk benchmark allowed to occupy the rest of the GPU. """
    if mps_ratio <= 0.0:
        print "The worker must be allowed to use over 0% of the GPU!"
        exit(1)
    thrasher_ratio = 1.0 - mps_ratio
    thrasher_config = {
        "label": "Competitor",
        "mps_thread_percentage": thrasher_ratio * 100.0,
        "log_name": "thrasher_%f.json" % (mps_ratio),
        "thread_count": 1024,
        "block_count": 160,
        "data_size": 2 * 1024 * 1024,
        "filename": "./bin/random_walk.so",
        "additional_info": 15000
    }
    # The Mandelbrot-set benchmark calculates its number of threads based on
    # data size. We'll need at least 160 * 1024 threads to occupy the entire
    # GPU.
    mandelbrot_data_size = 1024 * 1024
    worker_config = {
        "label": "%f" % (mps_ratio * 100.0),
        "mps_thread_percentage": mps_ratio * 100.0,
        "log_name": "worker_%f.json" % (mps_ratio),
        "thread_count": 512,
        "block_count": 160,
        "data_size": mandelbrot_data_size,
        "filename": "./bin/mandelbrot.so",
        "additional_info": 2000
    }
    benchmarks = [worker_config]
    # Don't include a thrasher at all at 100%
    if mps_ratio < 1.0:
        benchmarks.append(thrasher_config)
    overall_config = {
            "name": "Volta MPS resource limit test",
            "max_iterations": 100,
            "max_time": 0,
            "cuda_device": 0,
            "pin_cpus": True,
            "use_processes": True,
            "sync_every_iteration": True,
            "benchmarks": benchmarks
    }
    return json.dumps(overall_config, indent=2)

def start_process(mps_ratio):
    config = generate_config(mps_ratio)
    print "Starting process with a ratio of %.2f." % (mps_ratio)
    process = Popen(["./bin/runner", "-"], stdin=PIPE)
    process.communicate(input=config)

def run_tests():
    ratios = [1.0, 0.9, 0.75, 0.8, 0.6, 0.5, 0.4, 0.25, 0.1]
    for r in ratios:
        start_process(r)

run_tests()

