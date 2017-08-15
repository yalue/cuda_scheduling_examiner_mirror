# This scripts looks through JSON result files and uses matplotlib to display
# scatterplots containing the min, max and arithmetic mean for distributions of
# samples. In order for a distribution to be included in a plot, its "label"
# field must consist of a single number (may be floating-point). As with the
# other scripts, one plot will be created for each "name" in the output files.
import glob
import json
import matplotlib.pyplot as plot
import numpy
import sys

def convert_to_float(s):
    """Takes a string s and parses it as a floating-point number. If s can not
    be converted to a float, this returns None instead."""
    to_return = None
    try:
        to_return = float(s)
    except:
        to_return = None
    return to_return

def benchmark_summary_values(benchmark):
    """Takes a single benchmark results (one parsed output file) and returns
    a list containing 3 elements: [min block duration, max block duration,
    mean block duration]."""
    block_durations = []
    for t in benchmark["times"]:
        if "block_times" not in t:
            continue
        block_times = t["block_times"]
        i = 0
        while i < len(block_times):
            block_duration = block_times[i + 1] - block_times[i]
            block_durations.append(block_duration)
            i += 2
    return [min(block_durations), max(block_durations), numpy.mean(
        block_durations)]


def benchmark_sort_key(benchmark):
    """Returns the key that should be used to sort the benchmarks by label
    (treating each label as a floating-point value)."""
    return float(benchmark["label"])

def get_distribution(benchmarks):
    """Takes a list of benchmark results and returns a distribution of the
    form: [[x values], [y_min values], [y_max values], [y_mean values]].
    The X values are taken from each benchmark's label and the y values are
    from the corresponding block durations."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    x_values = []
    y_min_values = []
    y_max_values = []
    y_mean_values = []
    for b in benchmarks:
        x_value = float(b["label"])
        y_values = benchmark_summary_values(b)
        x_values.append(x_value)
        # Convert times to milliseconds here
        y_min_values.append(y_values[0] * 1000.0)
        y_max_values.append(y_values[1] * 1000.0)
        y_mean_values.append(y_values[2] * 1000.0)
    return [x_values, y_min_values, y_max_values, y_mean_values]

def plot_scenario(benchmarks, name):
    data = get_distribution(benchmarks)
    min_x = min(data[0])
    max_x = max(data[0])
    # Get the min and max y from the lists of minimum and maximum values.
    min_y = min(data[1])
    max_y = max(data[2])
    x_range = max_x - min_x
    x_pad = 0.025 * x_range
    y_range = max_y - min_y
    y_pad = 0.025 * y_range
    figure = plot.figure()
    figure.suptitle(name)
    plot.axis([min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad])
    plot.xticks(numpy.arange(min_x - 1, max_x + x_pad, (x_range + 1) / 8.0))
    plot.yticks(numpy.arange(min_y, max_y + y_pad, y_range / 8.0))
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(data[0], data[2], label="Max", linestyle="None", marker="^",
        fillstyle="full", markeredgewidth=0.0, ms=7)
    axes.plot(data[0], data[3], label="Average", linestyle="None", marker="*",
        fillstyle="full", markeredgewidth=0.0, ms=8)
    axes.plot(data[0], data[1], label="Min", linestyle="None", marker="v",
        fillstyle="full", markeredgewidth=0.0, ms=7)
    axes.set_ylabel("Block duration (ms)")
    axes.set_xlabel("Thread count")
    legend = plot.legend()
    legend.draggable()
    return figure

def show_plots(filenames):
    """Takes a list of filenames and generates one plot per named scenario
    across all of the files."""
    parsed_files = []
    counter = 1
    for name in filenames:
        print "Parsing file %d / %d: %s" % (counter, len(filenames), name)
        counter += 1
        with open(name) as f:
            parsed = json.loads(f.read())
            if len(parsed["times"]) < 2:
                print "Skipping %s: no recorded times in file." % (name)
                continue
            float_value = convert_to_float(parsed["label"])
            if float_value is None:
                print "Skipping %s: label isn't a number." % (name)
                continue
            parsed_files.append(parsed)
    # Group the files by scenario name.
    scenarios = {}
    for benchmark in parsed_files:
        scenario = benchmark["scenario_name"]
        if not scenario in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(benchmark)
    # Draw each scenario's plot.
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
