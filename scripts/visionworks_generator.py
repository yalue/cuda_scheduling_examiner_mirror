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
import sys

withMPS = False

def generate_benchmark(filename, log_name, label_name):
    """ Returns a dict that, when converted to JSON, will work as a benchmark
    configuration for the visionworks benchmark. """

    sys.stderr.write(log_name)
    sys.stderr.write(label_name)
    to_return = {}
    to_return["filename"] = filename
    to_return["log_name"] = log_name
    to_return["label"] = label_name 
    to_return["thread_count"] = 0
    to_return["block_count"] = 0
    to_return["data_size"] = 0

    # Running multiple instances of one or different demos in a common address
    # space is not supported yet. Global status shared for multiple instances
    # does not support multiple entry
    #
    # WARNING: comment this line when generating config file for multiple
    # instances while -p is not turned on.
    #
    # to_return["additional_info"] = {"shouldRender" : True};
    return to_return

def generate_app_config(benchmarks, filename, log_name, count):
    """ Returns a JSON string which can serve as a benchmark config."""
    if count==0:
        return;

    # Generate individual benchmark configs
    for i in range(count):
        benchmark = generate_benchmark(filename, log_name + ("_MPS" if withMPS else "_MT") + "_x" + str(count) + "_" + str(i) + ".json",
                "x" + str(count) + (" (MPS)" if withMPS else ""))
        benchmarks.append(benchmark)

    # Return the config as a JSON string.
    return json.dumps(config, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", help="feature tracker", type=int, default=0)
    parser.add_argument("--ht", help="hough transform", type=int, default=0)
    parser.add_argument("--me", help="motion estimation", type=int, default=0)
    parser.add_argument("--sm", help="stereo matching", type=int, default=0)
    parser.add_argument("--vs", help="video stabilizer", type=int, default=0)
    parser.add_argument("-n", "--name", help="scenario name", default="VisionWorks demo")
    parser.add_argument("-i", "--iter", help="max iterations number", type=int, default=1000)
    parser.add_argument("-p", "--process", action="store_true")
    args = parser.parse_args()
    config = {}
    # Generate the top-level global config
    config["use_processes"] = args.process
    #config["use_processes"] = True
    # For now, if each instance is a process, by default we have MPS set up.
    withMPS = args.process
    if withMPS and args.process:
        sys.stderr.write("MPS is used.\n")

    config["name"] = args.name
    config["max_iterations"] = args.iter
    config["max_time"] = 0
    config["cuda_device"] = 0
    config["pin_cpus"] = False
    #config["sync_every_iteration"] = True
    benchmarks = []
    generate_app_config(benchmarks,
            "./bin/nvx_demo_feature_tracker.so",
            "feature_tracker", args.ft)
    generate_app_config(benchmarks,
            "./bin/nvx_demo_hough_transform.so",
            "hough_transform", args.ht)
    generate_app_config(benchmarks,
            "./bin/nvx_demo_motion_estimation.so",
            "motion_estimation", args.me)
    generate_app_config(benchmarks,
            "./bin/nvx_demo_stereo_matching.so",
            "stereo_matching", args.sm)
    generate_app_config(benchmarks,
            "./bin/nvx_demo_video_stabilizer.so",
            "video_stabilizer", args.vs)
    config["benchmarks"] = benchmarks
    print json.dumps(config, indent=2)
