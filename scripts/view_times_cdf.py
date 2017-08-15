import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys

def convert_values_to_cdf(values):
    """Takes a 1-D list of values and converts it to a CDF representation."""
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

def plot_scenario(benchmarks, name, times_key):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a CDF plot of CPU times for the scenario. See get_benchmark_cdf
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
        cdf_data = get_benchmark_cdf(b, times_key)
        min_value = min(cdf_data[0])
        max_value = max(cdf_data[0])
        if min_value < min_x:
            min_x = min_value
        if max_value > max_x:
            max_x = max_value
        cdfs.append(cdf_data)
    x_range = max_x - min_x
    x_pad = 0.05 * x_range
    figure = plot.figure()
    figure.suptitle(name)
    plot.axis([min_x - x_pad, max_x + x_pad, -5.0, 105.0])
    plot.xticks(numpy.arange(min_x, max_x + x_pad, x_range / 5.0))
    plot.yticks(numpy.arange(0, 105, 100 / 5.0))
    axes = figure.add_subplot(1, 1, 1)
    for i in range(len(cdfs)):
        axes.plot(cdfs[i][0], cdfs[i][1], label=labels[i], lw=3,
            **(style_cycler.next()))
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
    base_directory = "./results"
    if len(sys.argv) > 2:
        print "Usage: python %s [directory containing results (./results)]" % (
            sys.argv[0])
        exit(1)
    if len(sys.argv) == 2:
        base_directory = sys.argv[1]
    filenames = glob.glob(base_directory + "/*.json")
    show_plots(filenames)
