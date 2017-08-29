# This script reads all output JSON files and produces inputs for Don's tool.
# By default, input files are read from the './results' directory, and outputs
# are written into the results directory, with two files created per stream.
# This script will fail if more than one experiment currently has output files
# in the './results' directory, so make sure to clear the results, run a single
# experiment, and then run this script. Optionally, a different input and
# output directory can be specified as command-line arguments. Usage:
#
#    python generate_simulator_input_files.py [input_directory [output_dir]]
import glob
import json
import sys

def format_single_stream(data, kernel_filename, block_filename):
    """Takes a parsed result JSON file and the filenames to write the kernel
    and block configuration for. The written configurations will be in the
    format needed by the scheduling simulator."""
    # First, get a list of all kernels using the kernel_name field. Generate a
    # map of kernel name to warp count, block count, and start time.
    kernels = {}
    for t in data["times"]:
        if "kernel_name" not in t:
            continue
        if t["kernel_name"] in kernels:
                # We've already recorded this kernel.
                continue
        kernel = {}
        warp_count = t["thread_count"] / 32
        if (t["thread_count"] % 32) != 0:
            warp_count += 1
        kernel["warp_count"] = warp_count
        kernel["block_count"] = len(t["block_times"]) / 2
        kernel["launch_time"] = int(t["cuda_launch_times"][0] * 1e9)
        kernels[t["kernel_name"]] = kernel

    # The kernel array file must have kernels sorted by launch time.
    kernel_array = []
    for name in kernels:
        k = kernels[name]
        launch_time = k["launch_time"]
        block_count = k["block_count"]
        warp_count = k["warp_count"]
        kernel_array.append([launch_time, block_count, warp_count])
    kernel_array = sorted(kernel_array, key=lambda v: v[0])

    # This file consists of plaintext columns: 4 integers per line. The line
    # contains a kernel number, start time, block count, and warp count.
    f = open(kernel_filename, "wb")
    for i in range(len(kernel_array)):
        k = kernel_array[i]
        line = "%d %d %d %d\n" % (i, k[0], k[1], k[2])
        f.write(line)
    f.close()

    # Next, write the block times to the blocks file.
    f = open(block_filename, "wb")
    for t in data["times"]:
        if "block_times" not in t:
            continue
        i = 0
        # This file consists of one line per block, with 3 integers per line.
        # In order, the numbers are SM ID, start time, and end time.
        for i in range(len(t["block_smids"])):
            start_nanoseconds = int(t["block_times"][i * 2] * 1e9)
            end_nanoseconds = int(t["block_times"][(i * 2) + 1] * 1e9)
            sm = t["block_smids"][i]
            line = "%d %d %d\n" % (sm, start_nanoseconds, end_nanoseconds)
            f.write(line)
    f.close()

def reformat_data(input_files, output_directory):
    """Takes a list of input JSON files and a directory in which output files
    should be written. Parses the input (experiment result) JSON files and
    generates a number of output files, two for each input JSON file. Output
    files will always be named along the lines of s<X>_kernels and s<X>_blocks,
    where <X> is a number in the range [0, N), where N is the number of
    streams/experiment result files."""
    i = 0
    # Use this to detect if different experiments have run.
    scenario_name = None
    for filename in input_files:
        with open(filename) as f:
            parsed = json.loads(f.read())
            if scenario_name is None:
                scenario_name = parsed["scenario_name"]
            if scenario_name != parsed["scenario_name"]:
                message = "Multiple scenarios found in output files. "
                message += "Try clearing your output files, then re-running "
                message += "the experiment."
                raise Exception(message)
            kernel_output = "%s/s%d_kernels" % (output_directory, i)
            block_output = "%s/s%d_blocks" % (output_directory, i)
            format_single_stream(parsed, kernel_output, block_output)
        i += 1

if __name__ == "__main__":
    base_directory = "./results"
    output_directory = "./results"
    if len(sys.argv) > 3:
        print "Usage: python %s [input_directory [output_directory]]" % (
            sys.argv[0])
        exit(1)
    if len(sys.argv) >= 2:
        base_directory = sys.argv[1]
    if len(sys.argv) == 3:
        output_directory = sys.argv[2]
    input_files = glob.glob(base_directory + "/*.json")
    reformat_data(input_files, output_directory)
