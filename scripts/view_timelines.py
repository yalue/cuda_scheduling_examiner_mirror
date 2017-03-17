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

def get_thread_timeline(benchmark):
    """"Takes a parsed benchmark dict and returns timeline data consisting of
    a list of two lists. The first list will contain times, and the second list
    will contain the corresponding number of threads running at each time."""
    start_times = []
    end_times = []
    # Remember, the first entry in the times array is an empty object.
    all_times = benchmark["times"][1:]
    # Get only the block times.
    for i in range(len(all_times)):
        all_times[i] = all_times[i]["block_times"]
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
            # Get the next closest time, be it a start or an end time. The <=
            # is important, since we always want increment the block count
            # before decrementing in the case when a block starts at the same
            # time another ends.
            if start_times[-1] <= end_times[-1]:
                current_time = start_times.pop()
                is_start_time = True
            else:
                current_time = end_times.pop()
                is_start_time = False
        else:
            # Only end times are left, so keep decrementing the block count.
            current_time = end_times.pop()
            is_start_time = False
        # Make sure that changes between numbers of running blocks look abrupt.
        # Do this by only changing the number of blocks at the instant they
        # actually change rather than interpolating between two values.
        timeline_times.append(current_time)
        timeline_values.append(current_block_count)
        # Update the current running number of blocks
        if is_start_time:
            current_block_count += 1
        else:
            current_block_count -= 1
        timeline_times.append(current_time)
        timeline_values.append(current_block_count)
    # Convert the block count to a thread count
    # TODO: When multiple kernels are supported, this will probably need to be
    # updated.
    for i in range(len(timeline_values)):
        timeline_values[i] *= benchmark["thread_count"]
    return [timeline_times, timeline_values]

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
    height = (top - bottom) * 0.8
    color = 'pink'
    axes.arrow(release_time, bottom, 0, height, color=color, lw=2, width=width,
        length_includes_head=True, head_length=height * 0.15,
        head_width=width * 5)

def plot_scenario(benchmarks, name):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a plot showing the timeline of benchmark behaviors for the
    specific scenario. Returns a matplotlib Figure object."""
    figure = plot.figure()
    figure.suptitle(name)
    total_timeline = get_total_timeline(benchmarks)
    min_time = min(total_timeline[0])
    max_time = max(total_timeline[0])
    max_resident_threads = benchmarks[0]["max_resident_threads"]
    # Plot each timeline in a separate subplot
    for i in range(len(benchmarks)):
        benchmark = benchmarks[i]
        axes = figure.add_subplot(len(benchmarks) + 1, 1, i + 1)
        timeline = get_thread_timeline(benchmark)
        # Make sure all timelines extend to the right end of the plot
        timeline[0].append(max_time)
        timeline[1].append(0)
        max_threads = max(timeline[1])
        set_axes_dimensions(axes, min_time, max_time, 0, max_threads)
        # Draw the release arrow before (below) the plotted line
        if "release_time" in benchmark:
            draw_release_arrow(axes, benchmark["release_time"])
        axes.plot(timeline[0], timeline[1], c='k', lw=3)
        label = "# threads,\n%d: %s" % (i + 1, benchmark["benchmark_name"])
        if "label" in benchmark:
            label = benchmark["label"]
        axes.set_ylabel("# threads,\n" + label)
    # Add the total timeline in the bottommost subplot
    axes = figure.add_subplot(len(benchmarks) + 1, 1, len(benchmarks) + 1)
    max_threads = max(total_timeline[1])
    set_axes_dimensions(axes, min_time, max_time, 0, max_threads)
    axes.set_ylabel("# threads, total\n(max supported: %d): " %
        (max_resident_threads))
    axes.set_xlabel("Time (seconds)")
    axes.plot(total_timeline[0], total_timeline[1], c='k', lw=3)
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
