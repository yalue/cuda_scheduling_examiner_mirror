#!/usr/bin/env python3
# This scripts looks through JSON result files and uses matplotlib to display
# scatterplots containing the min, max and arithmetic mean for distributions of
# samples. In order for a distribution to be included in a plot, its "label"
# field must consist of a single number (may be floating-point). As with the
# other scripts, one plot will be created for each "name" in the output files.
import argparse
import copy
import itertools
import glob
import json
import matplotlib.pyplot as plot
import numpy
import os
import sys

# Standard figure styling
plot.rcParams["pdf.use14corefonts"] = "True" # So that it doesn't try to embed Tex Gyre Heros
plot.rcParams["font.sans-serif"] = ["TeX Gyre Heros", "Nimbus Sans", "Helvetica", "Arimo"]
plot.rcParams["font.size"] = 8 # Default IEEEtran footnote size

def convert_to_float(s):
    """Takes a string s and parses it as a floating-point number. If s can not
    be converted to a float, this returns None instead."""
    to_return = None
    try:
        to_return = float(s)
    except:
        to_return = None
    return to_return

def plugin_summary_values(plugin, times_key):
    """Takes a single plugin results (one parsed output file) and returns
    a list containing 3 elements: [min duration, max duration, mean duration].
    Durations are converted to milliseconds."""
    durations = []
    for t in plugin["times"]:
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
    axes.set_yscale("log")
    y_limits = axes.get_ybound()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xbound()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0], y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    #axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
    #    x_range / 5.0))
    #axes.xaxis.set_ticks([1, 15, 30, 45, 60])
    #axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
    #    y_range / 5.0))
    return None

def get_marker_styles():
    """ Returns a list of dicts of marker style kwargs. The plot will cycle
    through these for each scenario that's added to the plot. (In practice, I
    only expect to use this script to plot two scenarios at once, so I'm only
    returning two options here for now. """
    base_style = {
        "linestyle": "-",
        "marker": "o",
#        "markerfacecolor": "k",
#        "markeredgecolor": "k",
#        "markeredgewidth": 0,
        "fillstyle": "full",
        "markersize": 3,
        "color": "#0277bd", # blue
        "drawstyle": "steps-post",
    }
    style_2 = copy.deepcopy(base_style)
    style_2["linestyle"] = ":"
    style_2["marker"] = "s"
    style_2["markersize"] = 5
#    style_2["markeredgecolor"] = "0.7"
#    style_2["markerfacecolor"] = "0.7"
#    style_2["markeredgewidth"] = 0
    style_2["color"] = "#f4b400" # orange
    style_3 = copy.deepcopy(base_style)
    style_3["linestyle"] = "--"
    style_3["marker"] = "x"
    style_3["markersize"] = 6
    style_3["color"] = "#0f9d58" #green
    return [style_3, style_2, base_style]

def add_scenario_to_plot(axes, scenario, name, style_dict):
    data = scenario_to_distribution(scenario)
    # data[0] = x vals, data[1] = min, data[2] = max, data[3] = avg
    if "mps" in name:
        name = "MPS"
    elif "mig" in name:
        name = "MiG"
    else:
        name = "nvsplit"
    #axes.plot(data[0], data[2], label="Max", linestyle="None", marker="^",
    #    fillstyle="full", markeredgewidth=0.0, ms=7)
    axes.plot(data[0], data[3], label=name, **style_dict)
    #axes.plot(data[0], data[1], label="Min", linestyle="None", marker="v",
    #    fillstyle="full", markeredgewidth=0.0, ms=7)
    axes.set_ylabel("Average MM8192 Time (ms)")
    axes.set_xlabel("TPC Partition Size (# of TPCs)")
    legend = plot.legend()
    try:
        legend.draggable()
    except:
        legend.set_draggable(True)
    return None

def show_plots(filenames, times_key, h, w):
    """ Takes a list of filenames and generates one plot. This differs from the
    hip_plugin_framework script in that it only generates a single plot,
    containing only the average times. It will show one distribution per named
    scenario in the files. """
    # Maps plugin names to plugin data, where the plugin data is a map
    # of X-values to y-value triplets.
    all_scenarios = {}
    counter = 1
    for name in filenames:
        print("Parsing file %d / %d: %s" % (counter, len(filenames), name))
        counter += 1
        with open(name) as f:
            parsed = json.loads(f.read())
            if "label" not in parsed:
                print("Skipping %s: no \"label\" field in file." % (name))
                continue
            if len(parsed["times"]) < 2:
                print("Skipping %s: no recorded times in file." % (name))
                continue
            float_value = convert_to_float(parsed["label"])
            if float_value is None:
                print("Skipping %s: label isn't a number." % (name))
                continue
            summary_values = plugin_summary_values(parsed, times_key)
            name = parsed["scenario_name"]
            if name not in all_scenarios:
                all_scenarios[name] = {}
            all_scenarios[name][float_value] = summary_values

    # Add each scenario to the plot.
    style_cycler = itertools.cycle(get_marker_styles())
    px = 1/plot.rcParams['figure.dpi']
    figure = plot.figure(figsize=(w*px, h*px))
    figure.canvas.manager.set_window_title("TPC partition size vs. MM8192 Time")
    axes = figure.add_subplot(1, 1, 1)
    axes.autoscale(enable=True, axis='both', tight=True)
    for name in all_scenarios:
        add_scenario_to_plot(axes, all_scenarios[name], name,
            next(style_cycler))
    add_plot_padding(axes)
    #plot.subplots_adjust(bottom=0.35)
    plot.tight_layout()
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="execute_times")
    parser.add_argument("-v", "--height",
        help="Height (in pixels) of the plot (400 default).", default=400, type=int)
    parser.add_argument("-w", "--width",
        help="Width (in pixels) of the plot (600 default).", default=600, type=int)
    parser.add_argument("result_file_to_plot", nargs="*", default=["./results"],
        help="List of result files, or directories of result files, to plot (./results default)")
    args = parser.parse_args()
    filenames = []
    # If a positional argument is a directory, it's automatically expanded out
    # to include all contained *.json files. This supports the old usage:
    # `python view_blocksbysm.py [results directory (default: ./results)]`
    for f in args.result_file_to_plot:
        if os.path.isdir(f):
            filenames.extend(glob.glob(f + "/*.json"))
        elif os.path.isfile(f):
            filenames.append(f)
        else:
            print("Input path '%s' not found as valid file or directory." % f, file=sys.stderr)
            exit(1)
    show_plots(filenames, args.times_key, args.height, args.width)

