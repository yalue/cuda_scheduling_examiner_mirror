# This script takes an GPU scheduling simulator output and generates a JSON-
# format output file that is compatible with our current schedule-viewing
# scripts. By default, it looks for a file in './results/sim_log', and
# generates files with names like 'simulation_generated_*.json" in the
# './results' directory. These directories can be overridden using command-line
# arguments, though. Usage:
#
#    python convert_simulator_output_files.py [path_to_sim_log [output_dir]]
import json
import sys

def generate_output_file(data, stream_id, output_directory, time_offset):
    """Takes block time data for a single stream only, and the stream's ID,
    then creates a JSON-format output file in the output directory. The time
    offset is the first release time of any kernel in any stream."""
    output_filename = "converted_sim_stream%d.json" % (stream_id)
    output_filename = output_directory + "/" + output_filename
    # Blocks need to be split by kernel
    kernel_blocks = {}
    for b in data:
        kernel_id = b["kernel_number"]
        if kernel_id not in kernel_blocks:
            kernel_blocks[kernel_id] = []
        kernel_blocks[kernel_id].append(b)

    # Sort blocks by which one started first
    def sort_by_start_key(a):
        return a["start_time"]
    for kernel_id in kernel_blocks:
        resorted = sorted(kernel_blocks[kernel_id], key = sort_by_start_key)
        kernel_blocks[kernel_id] = resorted

    times_array = []
    # This first empty element is required by the output file format.
    times_array.append({})

    # Append kernel data to the times array
    for kernel_id in kernel_blocks:
        k = kernel_blocks[kernel_id]
        if len(k) == 0:
            continue
        to_append = {}
        to_append["kernel_name"] = "Kernel %d" % (kernel_id)
        to_append["block_count"] = len(k)
        to_append["thread_count"] = k[0]["thread_count"]
        to_append["shared_memory"] = 0
        to_append["cpu_core"] = 0
        # The launch times are just taken from the start and end times of the
        # blocks.
        launch_times = [0.0, 0.0, 0.0]
        launch_times[0] = k[0]["start_time"] - time_offset
        launch_times[1] = k[0]["start_time"] - time_offset
        launch_times[2] = k[-1]["end_time"] - time_offset
        to_append["cuda_launch_times"] = launch_times
        block_times = []
        sm_ids = []
        for block in k:
            block_times.append(block["start_time"] - time_offset)
            block_times.append(block["end_time"] - time_offset)
            sm_ids.append(block["sm_id"])
        to_append["block_times"] = block_times
        to_append["block_smids"] = sm_ids
        times_array.append(to_append)

    # Place the times array in the top-level output file format.
    output_data = {
        "scenario_name": "Converted simulation output",
        "benchmark_name": "Simulated kernels",
        "label": "Stream ID %d" % (stream_id),
        "max_resident_threads": 4096,
        "data_size": 0,
        "release_time": 0,
        "PID": 1337,
        "TID": 1337,
        "times": times_array
    }

    # Convert the output data to JSON and write it to a file
    f = open(output_filename, "wb")
    f.write(json.dumps(output_data))
    f.close()

def split_into_output_files(data, output_directory):
    """Takes an array of parsed input file lines and splits the data by stream
    into output files."""
    # Group the data by stream, and figure out the earliest start time at the
    # same time.
    stream_blocks = {}
    overall_start_time = 1.0e20
    for b in data:
        stream_id = b["stream_id"]
        if stream_id not in stream_blocks:
            stream_blocks[stream_id] = []
        stream_blocks[stream_id].append(b)
        if b["start_time"] < overall_start_time:
            overall_start_time = b["start_time"]
    for stream_id in stream_blocks:
        generate_output_file(stream_blocks[stream_id], stream_id,
            output_directory, overall_start_time)

def generate_output_files(sim_log, output_directory):
    """Generates JSON-formatted result files in the given output directory,
    from the contents of the simulation file at the path given by sim_log."""
    data = []
    lines = []
    with open(sim_log) as f:
        lines = f.readlines()
    for l in lines:
        parsed_line = {}
        tokens = l.strip().split(" ")
        if (len(tokens) % 2) != 0:
            raise Exception("Files must have an even number of columns")
        for i in range(len(tokens) / 2):
            label = tokens[i * 2]
            value = int(tokens[(i * 2) + 1])
            if label == "SQ=":
                parsed_line["stream_id"] = value
            elif label == "K=":
                parsed_line["kernel_number"] = value
            elif label == "B=":
                parsed_line["block_index"] = value
            elif label == "W=":
                parsed_line["thread_count"] = value * 32
            elif label == "SM=":
                parsed_line["sm_id"] = value
            elif label == "S=":
                parsed_line["start_time"] = float(value) / 1.0e9
            elif label == "E=":
                parsed_line["end_time"] = float(value) / 1.0e9
            else:
                raise Exception("Unknown column label: " + label)
        data.append(parsed_line)
    split_into_output_files(data, output_directory)

if __name__ == "__main__":
    output_directory = "./results"
    input_file = "./results/sim_log"
    if len(sys.argv) > 3:
        print "Usage: python %s [path_to_sim_log [output_directory]]" % (
            sys.argv[0])
        exit(1)
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_directory = sys.argv[2]
    generate_output_files(input_file, output_directory)
