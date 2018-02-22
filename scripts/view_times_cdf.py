import argparse
import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys

def convert_values_to_cdf(values):
    """Takes a 1-D list of values and converts it to a CDF representation. The
    CDF consists of a vector of times and a vector of percentages of 100."""
    if len(values) == 0:
        return [[], []]
    values.sort()
    total_size = float(len(values))
    current_min = values[0]
    count = 0.0
    data_list = [values[0]]
    ratio_list = [0.0]
    for v in values:
        count += 1.0
        if v > current_min:
            data_list.append(v)
            ratio_list.append((count / total_size) * 100.0)
            current_min = v
    data_list.append(values[-1])
    ratio_list.append(100)
    # Convert seconds to milliseconds
    for i in range(len(data_list)):
        data_list[i] *= 1000.0
    return [data_list, ratio_list]

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
            raw_values.append(times[end_index] - times[start_index])
    return convert_values_to_cdf(raw_values)

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
            for c in color_options:
                to_add = {}
                if m is not None:
                    to_add["marker"] = m
                    to_add["markevery"] = 0.1
                to_add["ls"] = s
                to_add["c"] = c
                all_styles.append(to_add)
    return all_styles

def add_plot_padding(axes):
    """Takes matplotlib axes, and adds some padding so that lines close to
    edges aren't obscured by tickmarks or the plot border."""
    y_limits = axes.get_ybound()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xbound()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0] - y_pad, y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
        x_range / 5.0))
    axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
        y_range / 5.0))

def plot_scenario(benchmarks, name, times_key):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a CDF plot of CPU times for the scenario. See get_benchmark_cdf
    for an explanation of the times_key argument."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    style_cycler = itertools.cycle(get_line_styles())
    cdfs = []
    labels = []
    c = 0
    for b in benchmarks:
        c += 1
        label = "%d: %s" % (c, b["benchmark_name"])
        if "label" in b:
            label = b["label"]
        labels.append(label)
        cdf_data = get_benchmark_cdf(b, times_key)
        cdfs.append(cdf_data)
    figure = plot.figure()
    figure.suptitle(name)
    axes = figure.add_subplot(1, 1, 1)
    # Make the axes track data exactly, we'll manually add padding later.
    axes.autoscale(enable=True, axis='both', tight=True)
    for i in range(len(cdfs)):
        axes.plot(cdfs[i][0], cdfs[i][1], label=labels[i], lw=3,
            **(style_cycler.next()))
    add_plot_padding(axes)
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("% <= X")
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
        figures.append(plot_scenario(scenarios[scenario], scenario, times_key))
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.", default="block_times")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/*.json")
    show_plots(filenames, args.times_key)
