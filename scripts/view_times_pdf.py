import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys
import argparse

def convert_values_to_pdf(values, bin_count):
    """Takes a 1-D list of values and converts it to a PDF representation. The
    PDF is approximated bins of size bin_count. Returns a vector of values for
    each bin, and a vector of the probabilities that a number falls into the
    bin."""
    if len(values) == 0:
        return [[], []]
    if len(values) == 1:
        return [[values[0]], [1.0]]
    values.sort()
    total_count = float(len(values))
    time_range = float(values[len(values) - 1] - values[0])
    bin_size = time_range / bin_count
    start_time = values[0]
    end_time = start_time + bin_size
    time_list = [values[0]]
    probability_list = [0.0]
    count_in_bin = 0
    value_index = 0
    # Count the number of values in each bin, then convert it to a probability
    # by dividing by the total number of values.
    while end_time <= values[len(values) - 1]:
        while (value_index < len(values)) and (values[value_index] < end_time):
            count_in_bin += 1
            value_index += 1
        # We're past the end of the current bin, so record the probability of
        # the bin and move to the next one.
        probability_list.append(count_in_bin / total_count)
        time_list.append((end_time + start_time) / 2.0)
        start_time = end_time
        end_time = start_time + bin_size
        count_in_bin = 0
    return [time_list, probability_list]

def get_benchmark_pdf(benchmark, times_key, bin_count):
    """Takes a parsed benchmark result JSON file and returns a PDF (seconds vs.
    probability of the measurement) of the selected times for the benchmark.
    The times_key is used to specify which specific time measurement (from
    objects in the times array) to use. bin_count specifies the number of
    points required in the resulting PDF approximation. A higher value is more
    precise, but a lower value will make "peaks" more distinguishable."""
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
    return convert_values_to_pdf(raw_values, bin_count)

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

def plot_scenario(benchmarks, name, times_key, bin_count):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a PDF plot of CPU times for the scenario. See get_benchmark_pdf
    for an explanation of the times_key argument."""
    benchmarks = sorted(benchmarks, key = benchmark_sort_key)
    style_cycler = itertools.cycle(get_line_styles())
    pdfs = []
    labels = []
    min_x = 1.0e99
    max_x = -1.0
    min_y = 1.0e99
    max_y = -1.0
    c = 0
    for b in benchmarks:
        c += 1
        label = "%d: %s" % (c, b["benchmark_name"])
        if "label" in b:
            label = b["label"]
        labels.append(label)
        pdf_data = get_benchmark_pdf(b, times_key, bin_count)
        min_time = min(pdf_data[0])
        max_time = max(pdf_data[0])
        min_probability = min(pdf_data[1])
        max_probability = max(pdf_data[1])
        if min_time < min_x:
            min_x = min_time
        if max_time > max_x:
            max_x = max_time
        if min_probability < min_y:
            min_y = min_probability
        if max_probability > max_y:
            max_y = max_probability
        pdfs.append(pdf_data)
    x_range = max_x - min_x
    x_pad = 0.05 * x_range
    y_range = max_y - min_y
    if y_range == 0.0:
        min_y = 0.0
        max_y = 0.001
        y_range = 0.001
    y_pad = 0.05 * y_range
    figure = plot.figure()
    figure.suptitle(name)
    plot.axis([min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad])
    plot.xticks(numpy.arange(min_x, max_x + x_pad, x_range / 5.0))
    plot.yticks(numpy.arange(min_y, max_y + y_pad, y_range / 5.0))
    axes = figure.add_subplot(1, 1, 1)
    for i in range(len(pdfs)):
        axes.plot(pdfs[i][0], pdfs[i][1], label=labels[i], lw=3,
            **(style_cycler.next()))
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("Density")
    legend = plot.legend()
    legend.draggable()
    return figure

def show_plots(filenames, times_key="block_times", bin_count=20):
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
        figures.append(plot_scenario(scenarios[scenario], scenario, times_key,
            bin_count))
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="block_times")
    parser.add_argument("-b", "--pdf_bins",
        help="Number of bins to use for the PDF approximation.",
        default=20)
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/*.json")
    show_plots(filenames, args.times_key, int(args.pdf_bins))
