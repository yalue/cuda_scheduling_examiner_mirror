# This script generate VisionWorks demos configurations according to the parameters.
#
# For example:
#
#    python visionworks_generator.py --ft 1 -p | ./bin/runner -
#
# The above command will generate configuration input for ./bin/runner to run
# one feature tracker instance as a process.
#

import json
import random
import argparse

def generate_multikernel_benchmark(filename, log_name):
    """ Returns a dict that, when converted to JSON, will work as a benchmark
    configuration for the multikernel.so benchmark. The multikernel.so
    benchmark will submit a number of kernels given by kernel_count, with
    random thread and block counts in the given ranges (thread counts will
    always be rounded up to the nearest multiple of 32)."""


    to_return = {}
    to_return["filename"] = filename
    to_return["log_name"] = log_name
    to_return["label"] = log_name
    # Only additional_info is used for multikernel.so
    to_return["thread_count"] = 0
    to_return["block_count"] = 0
    to_return["data_size"] = 0
    # to_return["additional_info"] = kernel_config_list
    return to_return

def generate_overall_config(benchmarks, filename, log_name, count):
    """ Returns a JSON string which can serve as a benchmark config."""
    if count==0:
        return;

    # Generate individual benchmark configs
    for i in range(count):
        benchmark = generate_multikernel_benchmark(filename, log_name)
        benchmarks.append(benchmark)


    # Return the config as a JSON string.
    return json.dumps(config, indent=2)

if __name__ == "__main__":
    config = {}
    config["name"] = "VisionWorks demos"
    config["max_iterations"] = 1000
    config["max_time"] = 0
    config["cuda_device"] = 0
    config["pin_cpus"] = False
    benchmarks = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", help="feature tracker", type=int, default=0)
    parser.add_argument("--ht", help="hough transform", type=int, default=0)
    parser.add_argument("--me", help="motion estimation", type=int, default=0)
    parser.add_argument("--sm", help="stereo matching", type=int, default=0)
    parser.add_argument("--vs", help="video stabilizer", type=int, default=0)
    parser.add_argument("-p", "--process", action="store_true")
    args = parser.parse_args()
    config["use_processes"] = args.process
    generate_overall_config(benchmarks,
            "./bin/nvx_demo_feature_tracker.so",
            "feature_tracker", args.ft)
    generate_overall_config(benchmarks,
            "./bin/nvx_demo_hough_transform.so",
            "hough_transform", args.ht)
    generate_overall_config(benchmarks,
            "./bin/nvx_demo_motion_estimation.so",
            "motion_estimation", args.me)
    generate_overall_config(benchmarks,
            "./bin/nvx_demo_stereo_matching.so",
            "stereo_matching", args.sm)
    generate_overall_config(benchmarks,
            "./bin/nvx_demo_video_stabilizer.so",
            "video_stabilizer", args.vs)
    # Generate the top-level global config
    config["benchmarks"] = benchmarks
    print json.dumps(config, indent=2)
