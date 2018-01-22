import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys
import argparse


def get_benchmark_cdf(benchmark, times_key):
    """Takes a parsed benchmark result JSON file and returns a CDF (in seconds
    and percentages) of the CPU (total) times for the benchmark. The times_key
    argument can be used to specify which range of times (in the times array)
    should be used to calculate the durations to include in the CDF."""
    raw_values = []
    for t in benchmark["times"]:
        if not times_key in t:
            continue
        times = t[times_key]
        for i in range(len(times) / 2):
            start_index = i * 2
            end_index = i * 2 + 1
            # 1000 for converting to milliseconds from seconds
            raw_values.append(1000 * (times[end_index] - times[start_index]))
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
    generates a PDF plot of CPU times for the scenario. See get_benchmark_cdf
    for an explanation of the times_key argument."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    style_cycler = itertools.cycle(get_line_styles())
    cdfs = []
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
        print label
        cdf_data = get_benchmark_cdf(b, times_key)
        min_value = min(cdf_data)
        max_value = max(cdf_data)
        if min_value < min_x:
            min_x = min_value
        if max_value > max_x:
            max_x = max_value
        cdfs.append(cdf_data)
    x_range = max_x - min_x
    x_pad = 0.05 * x_range
    figure = plot.figure()
    figure.suptitle(name)
    plot.xlim(0, max_x + x_pad)
    plot.xticks(numpy.arange(0, max_x + x_pad, int(x_range / 10.0)))
    axes = figure.add_subplot(1, 1, 1)
    for i in range(len(cdfs)):
        axes.hist(cdfs[i], 80, normed=1, label=labels[i], histtype=u'step')
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
        print scenario + "\n"
        figures.append(plot_scenario(scenarios[scenario], scenario, times_key))
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory containing results.", default='./results')
    parser.add_argument("-k", "--times_key", help="key name for the time property to be plotted.",
            default="execute_times") #"block_times")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/*x4_0.json")
    show_plots(filenames, args.times_key)
