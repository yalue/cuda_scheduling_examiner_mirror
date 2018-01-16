# This script reads all JSON result files and uses matplotlib to display a
# timeline indicating when blocks and threads from multiple jobs were run on
# GPU. For this to work, all result filenames must end in .json.
#
# Usage: python view_timeline.py [results directory (default: ./results)]
import glob
import json
import matplotlib.pyplot as plot
import numpy
import sys

def get_kernel_timeline(kernel_times):
    """Takes a single kernel invocation's information from the benchmark struct
    and returns two lists. The first list contains times, and the second list
    contains the number of threads running at each time."""
    block_times = kernel_times["block_times"]
    start_times = []
    end_times = []
    # Split block times into start and end times.
    for i in range(len(block_times)):
        if (i % 2) == 0:
            start_times.append(block_times[i])
        else:
            end_times.append(block_times[i])
    # Sort times in reverse order, so the earliest times will be popped first.
    start_times.sort(reverse = True)
    end_times.sort(reverse = True)
    # Initialize the timeline to return so that 0 threads are running at time 0
    timeline_times = []
    timeline_values = []
    timeline_times.append(0.0)
    timeline_values.append(0)
    # Next, iterate over start and end times and keep a running count of how
    # many threads are running.
    current_thread_count = 0
    while True:
        if (len(start_times) == 0) and (len(end_times) == 0):
            break
        if len(end_times) == 0:
            print "Error! The last block end time was before a start time."
            exit(1)
        current_time = 0.0
        previous_thread_count = current_thread_count
        if len(start_times) != 0:
            # Get the next closest time, be it a start or an end time. The <=
            # is important, since we always want increment the block count
            # before decrementing in the case when a block starts at the same
            # time another ends.
            if start_times[-1] == end_times[-1]:
                # A block started and ended at the same time, don't change the
                # current thread count.
                current_time = start_times.pop()
                end_times.pop()
            elif start_times[-1] <= end_times[-1]:
                # A block started, so increase the thread count
                current_time = start_times.pop()
                current_thread_count += kernel_times["thread_count"]
            else:
                # A block ended, so decrease the thread count
                current_time = end_times.pop()
                current_thread_count -= kernel_times["thread_count"]
        else:
            # Only end times are left, so keep decrementing the block count.
            current_time = end_times.pop()
            current_thread_count -= kernel_times["thread_count"]
        # Make sure that changes between numbers of running threads are abrupt.
        # Do this by only changing the number of blocks at the instant they
        # actually change rather than interpolating between two values.
        timeline_times.append(current_time)
        timeline_values.append(previous_thread_count)
        # Finally, append the new thread count.
        timeline_times.append(current_time)
        timeline_values.append(current_thread_count)
    return [timeline_times, timeline_values]

def merge_timelines(timeline_a, timeline_b):
    """Takes two timelines, both of which must contain a list of times and a
    list of values, and combines them into a single timeline containing a list
    of times and added values, so that if an interval in both timelines
    overlap, then the returned timeline will show the sum of the two values
    during that interval."""
    combined_times = []
    combined_values = []
    # Reverse the timelines so that we pop the earliest values first.
    times_a = timeline_a[0]
    values_a = timeline_a[1]
    times_b = timeline_b[0]
    values_b = timeline_b[1]
    times_a.reverse()
    values_a.reverse()
    times_b.reverse()
    values_b.reverse()
    # Track the a and b values independently
    current_a_value = 0
    current_b_value = 0

    # Returns the next time from either timeline to appended to the output.
    def get_next_smallest_time():
        if (len(times_a) == 0) and (len(times_b) == 0):
            raise Exception("Internal error combining timelines.")
        if (len(times_a) == 0):
            return times_b[-1]
        if (len(times_b) == 0):
            return times_a[-1]
        if times_a[-1] <= times_b[-1]:
            return times_a[-1]
        return times_b[-1]

    # Run the update_output() closure until all of the original timeline values
    # have been combined into the output.
    while (len(times_a) > 0) or (len(times_b) > 0):
        t = get_next_smallest_time()
        # It's possible that both timelines have a change in value at the same
        # time.
        if (len(times_a) != 0) and (times_a[-1] == t):
            times_a.pop()
            current_a_value = values_a.pop()
        if (len(times_b) != 0) and (times_b[-1] == t):
            times_b.pop()
            current_b_value = values_b.pop()
        combined_times.append(t)
        combined_values.append(current_a_value + current_b_value)

    return [combined_times, combined_values]

def get_thread_timeline(benchmark):
    """"Takes a parsed benchmark dict and returns timeline data consisting of
    a list of two lists. The first list will contain times, and the second list
    will contain the corresponding number of threads running at each time."""
    # Remember, the first entry in the times array is an empty object.
    all_kernels = benchmark["times"][1:]
    to_return = [[], []]
    # Combine all kernels' timelines into a single timeline.
    for k in all_kernels:
        if "cpu_times" in k:
            continue
        kernel_timeline = get_kernel_timeline(k)
        to_return = merge_timelines(to_return, kernel_timeline)
    return to_return

def get_stackplot_values(benchmarks):
    """Takes a list of benchmark results and returns a list of lists of data
    that can be passed as arguments to stackplot (with a single list of
    x-values followed by multiple lists of y-values)."""
    timelines = []
    for b in benchmarks:
        timelines.append(get_thread_timeline(b))
    # Track indices into the list of times and values from each benchmark as
    # we build an aggregate list.
    times_lists = []
    values_lists = []
    indices = []
    new_times = []
    new_values = []

    for t in timelines:
        times_lists.append(t[0])
        values_lists.append(t[1])
        indices.append(0)
        new_values.append([])

    # Selects the next smallest time we need to add to our output list.
    def current_min_time():
        current_min = 1e99
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(times_lists[i]):
                continue
            if times_lists[i][n] < current_min:
                current_min = times_lists[i][n]
        return current_min

    # Returns true if we've seen all the times in every list.
    def all_times_done():
        for i in range(len(indices)):
            if indices[i] < len(times_lists[i]):
                return False
        return True

    # Moves to the next time for each input list if the current time is at the
    # head of the list.
    def update_indices(current_time):
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(times_lists[i]):
                continue
            if times_lists[i][n] == current_time:
                indices[i] += 1

    def update_values():
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(values_lists[i]):
                new_values[i].append(0)
                continue
            new_values[i].append(values_lists[i][n])

    while not all_times_done():
        current_time = current_min_time()
        new_times.append(current_time)
        update_values()
        update_indices(current_time)

    to_return = []
    to_return.append(new_times)
    for v in new_values:
        to_return.append(v)
    return to_return

def get_total_timeline(benchmarks):
    """Similar to get_stackplot_values, but only returns a single list of
    values, containing the total number of threads from all benchmarks."""
    data = get_stackplot_values(benchmarks)
    total_counts = []
    for i in range(len(data[0])):
        total_counts.append(0)
    for i in range(len(data) - 1):
        for j in range(len(data[i + 1])):
            total_counts[j] += data[i + 1][j]
    return [data[0], total_counts]

def set_axes_dimensions(axes, min_x, max_x, min_y, max_y):
    """Sets the ticks and size for the given axes. Includes padding space so
    that plotted lines don't sit on top of the axis."""
    x_range = max_x - min_x
    x_pad = x_range * 0.05
    y_range = max_y - min_y
    y_pad = y_range * 0.05
    axes.set_xticks(numpy.arange(min_x, max_x + x_pad, x_range / 5.0))
    axes.set_xlim(min_x - x_pad, max_x + x_pad)
    axes.set_ylim(min_y - y_pad, max_y + y_pad)

def draw_release_arrow(axes, release_time):
    """Draws a vertical arrow indicating the release time on the given set of
    axes."""
    right, left = axes.get_xlim()
    width = (right - left) * 0.004
    bottom, top = axes.get_ylim()
    height = (top - bottom) * 0.4
    color = 'lightgray'
    axes.arrow(release_time, bottom, 0, height, color=color, lw=2, width=width,
        length_includes_head=True, head_length=height * 0.15,
        head_width=width * 5)
    return

def benchmark_sort_key(benchmark):
    """Returns the key that may be used to sort benchmarks by label."""
    if not "label" in benchmark:
        return ""
    return benchmark["label"]

def benchmark_has_block_times(benchmark):
    """Returns true only if the benchmark includes some block times."""
    if len(benchmark["times"]) < 2:
        return False
    for t in benchmark["times"][1:]:
        # Return true only if the benchmark contains at least 1 block time.
        if "block_times" not in t:
            continue
        if len(t["block_times"]) >= 1:
            return True
    return False

def plot_scenario(benchmarks, name):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a plot showing the timeline of benchmark behaviors for the
    specific scenario. Returns a matplotlib Figure object."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    # Remove any benchmarks that don't have any block times.
    tmp = []
    for b in benchmarks:
        if benchmark_has_block_times(b):
            tmp.append(b)
    benchmarks = tmp
    figure = plot.figure()
    figure.suptitle(name)
    total_timeline = get_total_timeline(benchmarks)
    min_time = min(total_timeline[0])
    max_time = max(total_timeline[0])
    max_resident_threads = benchmarks[0]["max_resident_threads"]
    # Plot each timeline in a separate subplot
    for i in range(len(benchmarks)):
        benchmark = benchmarks[i]
        axes = figure.add_subplot(len(benchmarks), 1, i + 1)
        timeline = get_thread_timeline(benchmark)
        # Make sure all timelines extend to the right end of the plot
        timeline[0].append(max_time)
        timeline[1].append(0)
        max_threads = max(timeline[1])
        set_axes_dimensions(axes, min_time, max_time, 0, max_threads)
        # Draw the release arrow before (below) the plotted line
        if "release_time" in benchmark:
            draw_release_arrow(axes, benchmark["release_time"])
        axes.plot(timeline[0], timeline[1], color="k", lw=2)
        label = "%d: %s" % (i + 1, benchmark["benchmark_name"])
        if "label" in benchmark:
            label = benchmark["label"]
        axes.set_ylabel("# threads,\n" + label)
    # Add the X label below the bottommost subplot
    axes.set_xlabel("Time (seconds)")
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
