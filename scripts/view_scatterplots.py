# This scripts looks through JSON result files and uses matplotlib to display
# scatterplots containing the min, max and arithmetic mean for distributions of
# samples. In order for a distribution to be included in a plot, its "label"
# field must consist of a single number (may be floating-point). As with the
# other scripts, one plot will be created for each "name" in the output files.
import argparse
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

def benchmark_summary_values(benchmark, times_key):
    """Takes a single benchmark results (one parsed output file) and returns
    a list containing 3 elements: [min duration, max duration, mean duration].
    Durations are converted to milliseconds."""
    durations = []
    for t in benchmark["times"]:
        if times_key not in t:
            continue
        times = t[times_key]
        i = 0
        while i < len(times):
            duration = times[i + 1] - times[i]
            durations.append(duration)
            i += 2
    minimum = min(durations) * 1000.0
    maximum = max(durations) * 1000.0
    average = numpy.mean(durations) * 1000.0
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

def plot_scenario(scenario, name):
    data = scenario_to_distribution(scenario)
    figure = plot.figure()
    figure.suptitle(name)
    axes = figure.add_subplot(1, 1, 1)
    axes.autoscale(enable=True, axis='both', tight=True)
    axes.plot(data[0], data[2], label="Max", linestyle="None", marker="^",
        fillstyle="full", markeredgewidth=0.0, ms=7)
    axes.plot(data[0], data[3], label="Average", linestyle="None", marker="*",
        fillstyle="full", markeredgewidth=0.0, ms=8)
    axes.plot(data[0], data[1], label="Min", linestyle="None", marker="v",
        fillstyle="full", markeredgewidth=0.0, ms=7)
    add_plot_padding(axes)
    axes.set_ylabel("Duration (ms)")
    axes.set_xlabel("Value")
    legend = plot.legend()
    legend.draggable()
    return figure

def show_plots(filenames, times_key):
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
            summary_values = benchmark_summary_values(parsed, times_key)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="execute_times")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/*.json")
    show_plots(filenames, args.times_key)

