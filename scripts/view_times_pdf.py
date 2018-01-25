import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys
import argparse
from scipy import stats

def get_benchmark_raw_values(benchmark, times_key):
    """Takes a parsed benchmark result JSON file and returns raw values of the
    selected times for the benchmark.  The times_key is used to specify which
    specific time measurement (from objects in the times array) to use."""
    raw_values = []
    for t in benchmark["times"]:
        if not times_key in t:
            continue
        times = t[times_key]
        for i in range(len(times) / 2):
            start_index = i * 2
            end_index = i * 2 + 1
            milliseconds = (times[end_index] - times[start_index]) * 1000.0
            raw_values.append(milliseconds)
    return raw_values

def nice_sort_key(label):
    """If a label contains numbers, this will prevent sorting them
    lexicographically."""
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split(r'([0-9]+)', label)]

def benchmark_sort_key(benchmark):
    """Returns the key that may be used to sort benchmarks by label."""
    if not "label" in benchmark:
        return ""
    return nice_sort_key(benchmark["label"])

all_styles = None
def get_line_styles():
    """Returns a list of line style possibilities, that includes more options
    than matplotlib's default set that includes only a few solid colors."""
    global all_styles
    if all_styles is not None:
        return all_styles
    color_options = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "y",
        "black"
    ]
    style_options = [
        "-",
        "--",
        "-.",
        ":"
    ]
    marker_options = [
        None,
        "o",
        "v",
        "s",
        "*",
        "+",
        "D"
    ]
    # Build a combined list containing every style combination.
    all_styles = []
    for m in marker_options:
        for s in style_options:
            to_add = {}
            if m is not None:
                to_add["marker"] = m
                to_add["markevery"] = 0.1
            to_add["ls"] = s
            all_styles.append(to_add)
    return all_styles

def plot_scenario(benchmarks, name, times_key):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a PDF plot of CPU times for the scenario. See
    get_benchmark_raw_values for an explanation of the times_key argument."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    style_cycler = itertools.cycle(get_line_styles())
    raw_data_array = []
    labels = []
    min_x = 1.0e99
    max_x = -1.0
    c = 0
    for b in benchmarks:
        c += 1
        label = "%d: %s" % (c, b["benchmark_name"])
        if "label" in b:
            label = b["label"]
        labels.append(label)
        raw_data = get_benchmark_raw_values(b, times_key)
        min_time = min(raw_data)
        max_time = max(raw_data)
        if min_time < min_x:
            min_x = min_time
        if max_time > max_x:
            max_x = max_time
        raw_data_array.append(raw_data)
    x_range = max_x - min_x
    x_pad = 0.05 * x_range
    figure = plot.figure()
    figure.suptitle(name)
    plot.xticks(numpy.arange(int(min_x), max_x + x_pad, int(x_range / 5.0)))
    axes = figure.add_subplot(1, 1, 1)
    for i in range(len(raw_data_array)):
        density = stats.kde.gaussian_kde(raw_data_array[i])
        x = numpy.arange(0, max(raw_data_array[i]), .1)
        axes.plot(x, density(x), label=labels[i], **(style_cycler.next()))
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("Density")
    legend = plot.legend()
    legend.draggable()
    return figure

def show_plots(filenames, times_key="block_times"):
    """Takes a list of filenames, and generates one plot per scenario found in
    the files. See get_benchmark_cdf for an explanation of the times_key
    argument."""
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
        print scenario
        figures.append(plot_scenario(scenarios[scenario], scenario, times_key))
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="block_times")
    parser.add_argument("-r", "--regex",
        help="Regex for JSON files to be processed",
        default="*.json")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/" + args.regex)
    show_plots(filenames, args.times_key)
