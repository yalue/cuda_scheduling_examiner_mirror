# This script reads all JSON result files and uses matplotlib to display a
# timeline indicating when blocks and threads from multiple jobs were run on
# GPU. For this to work, all result filenames must end in .json.
#
# Usage: python view_timeline.py [results directory (default: ./results)]
import glob
import json
import matplotlib.pyplot as plot
import sys

def get_block_timeline(benchmark):
    """"Takes a parsed benchmark dict and returns timeline data consisting of
    a list of two lists. The first list will contain times, and the second list
    will contain the corresponding number of threads running at each time."""
    # TODO: Make the results files contain the *actual* number of threads per
    # block in kernels?
    start_times = []
    end_times = []
    # Remember, the first entry in the times array is an empty object.
    all_times = benchmark["times"][1:]
    # Get only the block times.
    for i in range(len(all_times)):
        all_times[i] = all_times[i]["block times"]
    # Separate each list of times into start and stop times.
    for invocation in all_times:
        for i in range(len(invocation)):
            if (i % 2) == 0:
                start_times.append(invocation[i])
            else:
                end_times.append(invocation[i])
    # Sort in reverse order, so the earliest times will be popped first.
    start_times.sort(reverse = True)
    end_times.sort(reverse = True)
    # Now, iterate over start and end times, keeping a count of how many blocks
    # are running.
    timeline_times = []
    timeline_values = []
    # All benchmarks must start with nothing running.
    timeline_times.append(0.0)
    timeline_values.append(0)
    current_block_count = 0
    while True:
        if len(start_times) == 0:
            if len(end_times) == 0:
                break
        if len(end_times) == 0:
            print "Error! The last block end time was before a start time."
            exit(1)
        current_time = 0.0
        is_start_time = False
        if len(start_times) != 0:
            # Get the next closest time, be it a start or an end time.
            if start_times[-1] < end_times[-1]:
                current_time = start_times.pop()
                is_start_time = True
            else:
                current_time = end_times.pop()
                is_start_time = False
        else:
            # Only end times are left, so keep decrementing the block count.
            current_time = end_times.pop()
            is_start_time = False
        if is_start_time:
            current_block_count += 1
        else:
            current_block_count -= 1
        timeline_times.append(current_time)
        timeline_values.append(current_block_count)
    return [timeline_times, timeline_values]

def get_largest_value(lists):
    """Takes a list of lists, and returns the largest value at the tail of one
    or more of the lists."""
    current_max = -1e99
    for v in lists:
        if (len(v) > 0) and (v[-1] > current_max):
            current_max = v[-1]
    return current_max

def all_lists_empty(lists):
    """Returns true if all of the given lists contain no entries."""
    for v in lists:
        if len(v) != 0:
            return False
    return True

def get_stackplot_values(benchmarks):
    """Takes a list of benchmark results and returns a list of lists of data
    that can be passed as arguments to stackplot (with a single list of
    x-values followed by multiple lists of y-values)."""
    # TODO (next): Re-do this completely! Just keep track of current indices
    # into each timeline, and build the resulting timeline from start to end
    # rather than in reverse order.
    timelines = []
    for b in benchmarks:
        timelines.append(get_block_timeline(b))
    return to_return

def plot_scenario(benchmarks, name):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a plot showing the timeline of benchmark behaviors for the
    specific scenario. Returns a matplotlib Figure object."""
    figure = plot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.set_title(name)
    values = get_stackplot_values(benchmarks)
    print values
    axes.stackplot(*values)
    return figure

def show_plots(filenames):
    """Takes a list of filenames, and generates one plot per scenario found in
    the files."""
    parsed_files = []
    for name in filenames:
        with open(name) as f:
            parsed_files.append(json.loads(f.read()))
    # Group the files by scenario
    scenarios = {}
    for benchmark in parsed_files:
        scenario = benchmark["scenario_name"]
        if not scenario in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(benchmark)
    figures = []
    for scenario in scenarios:
        figures.append(plot_scenario(scenarios[scenario], scenario))
    plot.show()

if __name__ == "__main__":
    base_directory = "./results"
    if len(sys.argv) > 2:
        print "Usage: python %s [directory containing results (./results)]" % (
            sys.argv[0])
        exit(1)
    if len(sys.argv) == 2:
        base_directory = sys.argv[1]
    filenames = glob.glob(base_directory + "/*.json")
    show_plots(filenames)
