# This script is intended for basic testing of benchmarks and the runner
# program. It will search the bin/ directory for any .so files, then generate
# one simple benchmark configuration for each .so file. To use it, run:
#
#    python scripts/test_all_benchmarks.py
#
# This should produce a <benchmark name>_so.json file for each benchmark in the
# results/ directory. Note that this will leave "additional_info" blank, so all
# benchmakrs *must* fall back to a default (at the very least, report an error
# and exit) if additional_info isn't provided.
import glob
import json
from os import path
from subprocess import Popen, PIPE

configs = []
libraries = glob.glob("bin/*.so")
for library in libraries:
    # For a name, we'll just use the name of the file without the .so extension
    benchmark_name = path.splitext(path.basename(library))[0]
    benchmark_config = {
        "filename": library,
        "log_name": benchmark_name + "_so.json",
        "label": "Test of " + benchmark_name,
        "thread_count": 64,
        "block_count": 1,
        "data_size": 4
    }
    config = {
        "name": "Testing benchmark " + library,
        "max_iterations": 1,
        "max_time": 1,
        "cuda_device": 0,
        "benchmarks": [benchmark_config]
    }
    json_config = json.dumps(config)
    print "Running benchmark " + library + "..."
    process = Popen(["./bin/runner", "-"], stdin=PIPE)
    process.communicate(json_config)

