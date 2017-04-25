import glob
import json
import matplotlib.pyplot as plot
import numpy
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
    return [data_list, ratio_list]

def get_benchmark_cdf(benchmark):
    """Takes a parsed benchmark result JSON file and returns a CDF (in seconds
    and percentages) of the CPU (total) times for the benchmark."""
    raw_values = []
    for t in benchmark["times"]:
        if not "cpu_times" in t:
            continue
        times = t["cpu_times"]
        raw_values.append(times[1] - times[0])
    return convert_values_to_cdf(raw_values)

def plot_scenario(benchmarks, name):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a CDF plot of CPU times for the scenario."""
    figure = plot.figure()
    figure.suptitle(name)
    axes = figure.add_subplot(1, 1, 1)
    c = 0
    for b in benchmarks:
        c += 1
        label = "%d: %s" % (c, b["benchmark_name"])
        if "label" in b:
            label = b["label"]
        data = get_benchmark_cdf(b)
        axes.plot(data[0], data[1], label=label, lw=3)
    axes.set_xlabel("Time (seconds)")
    axes.set_ylabel("% <= X")
    legend = plot.legend()
    legend.draggable()
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
