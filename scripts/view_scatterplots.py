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
    mean block duration]. Durations are converted to milliseconds."""
    block_durations = []
    for t in benchmark["times"]:
        if "block_times" not in t:
            continue
        block_times = t["block_times"]
        i = 0
        while i < len(block_times):
            block_duration = block_times[i + 1] - block_times[i]
            # Convert times to milliseconds here.
            block_durations.append(block_duration)
            i += 2
    minimum = min(block_durations) * 1000.0
    maximum = max(block_durations) * 1000.0
    average = numpy.mean(block_durations) * 1000.0
    return [minimum, maximum, average]

def scenario_to_distribution(scenario):
    """Takes a scenario, mapping numbers to triplets, and re-shapes the data.
    Returns an array of 4 arrays: [[x values], [min y values], [max y values],
    [average y values]]."""
    x_values = []
    for k in scenario:
        x_values.append(k)
    x_values.sort()
    min_y_values = []
    max_y_values = []
    mean_y_values = []
    for k in x_values:
        triplet = scenario[k]
        min_y_values.append(triplet[0])
        max_y_values.append(triplet[1])
        mean_y_values.append(triplet[2])
    return [x_values, min_y_values, max_y_values, mean_y_values]

def plot_scenario(scenario, name):
    data = scenario_to_distribution(scenario)
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
    # Maps benchmark names to benchmark data, where the benchmark data is a map
    # of X-values to y-value triplets.
    all_scenarios = {}
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
            summary_values = benchmark_summary_values(parsed)
            name = parsed["scenario_name"]
            if name not in all_scenarios:
                all_scenarios[name] = {}
            all_scenarios[name][float_value] = summary_values
    # Draw each scenario's plot.
    figures = []
    for scenario in all_scenarios:
        figures.append(plot_scenario(all_scenarios[scenario], scenario))
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
