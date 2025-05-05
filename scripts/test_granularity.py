#!/usr/bin/env python
# Copyright 2024 Joshua Bakita
# Test performance of matrix_multiply when run under every possible number of
# TPCs, with the TPC partitioning method varied.
# This is very similar to test_cu_mask.py, but without striping (as MPS
# doesn't support that).
#
# Supports Python 2 and Python 3
from __future__ import division
import argparse
import io
import json
import math
import os
import pysmctrl
import subprocess
import sys

# Python inexplicably does not provide access to setbuf()/setvbuf(), and is
# inconsistent about buffering on both stdout and stderr between versions, with
# full buffering (even on stderr) when output is redirected in Python 3. This
# results in our progress messages appearing in garbled order relative to the
# benchmark outputs. Attempt to fix this via reconfiguration in Python 3, and
# re-opening stdout to reset the buffer setting in Python 2 (this fix is
# inexplicably blocked on Python 3, which is why branching is required).
try:
    # Python 3
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), line_buffering=True)
    # Python 3.7+ only
    #sys.stdout.reconfigure(line_buffering=True)
except TypeError:
    # Python 2
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# MiG instance geometries are hardware-constrained; the A100 only
# supports 5 configurations for 1 instance, 7, 14, 21, 28, or 49 of 54 TPCs
# (derived from configuration options reported by nvidia-smi, with SM counts
# divided by two to yield the TPC count.)
MIG_CONFIGS = {7:"1g.10gb", 14:"2g.10gb", 21:"3g.20gb", 28:"4g.20gb", 49:"7g.40gb"}

def generate_config(device, part_method, total_tpcs, active_tpcs, iterations):
    """ Returns a JSON string containing a config. The config will use the
    Matrix Multiply plugin with a 8192x8192 matrix, using 32x32 thread blocks.
    The type of partitioning, and number of TPCs enabled, is varied.
    """
    plugin_config = {
        "label": str(active_tpcs),
        "log_name": "eurosys25_%s_%dtpcs.json" % (part_method, active_tpcs),
        "filename": "./bin/matrix_multiply.so",
        "thread_count": [32, 32], # Maximum block size. 32x32=1024
        "block_count": 1, # Ignored/unused by matrix_multiply
        "data_size": 0,
        "additional_info": {
            "matrix_width": 8192,
            "skip_copy": True
        }
    }
    if part_method.lower() == "mps":
        # MPS sets the number of TPCs such that it is at most the specified
        # percentage. This means that we need to specify an epsilon larger
        # percentage to ensure that our desired setting is achieved.
        # We set epsilon as the smallest value that cuda_scheduling_examiner
        # will pass on (it includes at most 4 decimal places of the setting). 
        epsilon = 0.0001
        # This will be a floating-point divide in both Python 2 and 3, due to
        # the `from __future__ import division`.
        percent_active = 100 * active_tpcs / total_tpcs + epsilon
        # Round to maximum precision supported by cuda_scheduling_examiner and
        # limit to at most 100%.
        plugin_config["mps_thread_percentage"] = min(100, round(percent_active, 4))
    elif part_method.lower() == "libsmctrl":
        # Create a bitstring of _enabled_ TPCs, convert to int, print as hex,
        # then prefix with '~' to make clear this is an enable mask.
        plugin_config["sm_mask"] = "~" + hex(int("1"*active_tpcs, 2))
    elif part_method.lower() == "mig":
        # MiG divides the GPU into two levels: GPU Instances, and nested
        # Compute Instances. Compute Instances have the same geometry
        # restrictions as GPU instances, and only provide compute isolation. To
        # the best of my understanding, Compute Instances are only useful
        # insomuch as they could be configured by a tenant.
        # Verify MiG is available.
        smi_status = os.system("nvidia-smi mig --list-gpu-instance-profiles")
        if not os.WIFEXITED(smi_status) or os.WEXITSTATUS(smi_status) == 6:
            raise Exception("MiG-capable GPU required for MiG experiments!")
        # TODO: Check that we're running on the A100 40GB
        config = MIG_CONFIGS[active_tpcs]
        # Delete any preexisting MiG configurations (nested instances first)
        os.system("sudo nvidia-smi mig --destroy-compute-instance")
        os.system("sudo nvidia-smi mig --destroy-gpu-instance")
        # Create a MiG GPU instance, with identical nested compute instance
        smi_status = os.system("sudo nvidia-smi mig --create-gpu-instance " + config + " --default-compute-instance")
        if not os.WIFEXITED(smi_status) or os.WEXITSTATUS(smi_status) != 0:
            raise Exception("Unable to create MiG instance with config " + config + "!")
    else:
        raise Exception("Unknown partitioning type '%s'!"%(part_method))
    name = "TPC Count vs. Performance with " + part_method
    overall_config = {
        "name": name,
        "max_iterations": iterations,
        "max_time": 0,
        "cuda_device": device,
        "use_processes": True, # Required for "mps_thread_percentage"
        "pin_cpus": True,
        "do_warmup": True,
        "benchmarks": [plugin_config]
    }
    return json.dumps(overall_config)

def run_process(device, part_method, total_tpcs, active_tpcs, iterations):
    """ This function starts a process that will run the plugin with the given
    number of active TPCs under the selected partitioning method. """
    config = generate_config(device, part_method, total_tpcs, active_tpcs, iterations)
    print("Starting test with %d TPCs enabled under %s"%(active_tpcs, part_method))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE, encoding='utf8')
    #process = subprocess.Popen(["cat", "-"], stdin=subprocess.PIPE, encoding='utf8')
    process.communicate(input=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpc_count", type=int,
        help="The total number of TPCs on the GPU.")
    parser.add_argument("--start_count", type=int, default=1,
        help="The number of TPCs to start testing from. Can be used to resume "+
             "tests if one hung.")
    parser.add_argument("-m", "--mig",
        help="Run MiG tests instead of MPS and libsmctrl. MiG mode must "+
             "already by enabled via nvidia-smi.", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0,
        help="Which GPU to test on")
    parser.add_argument("-i", "--iterations", type=int, default=10,
        help="How many iterations to run each benchmark? (~15m/iteration)")
    args = parser.parse_args()
    # If a tpc_count is specified, use that, otherwise attempt auto-detection
    if args.tpc_count != None:
        tpc_count = args.tpc_count
        if tpc_count <= 0:
            print("The TPC count must be positive and non-zero.")
            exit(22)
    elif pysmctrl:
        tpc_count = pysmctrl.get_tpc_info_cuda(args.device)
        print("Auto-detected %d available TPCs" % tpc_count)
    if args.start_count != None and args.start_count <= 0:
        print("The starting TPC count must be positive and non-zero.")
        exit(22)

    if args.mig:
        part_methods = ["mig"]
        part_options = MIG_CONFIGS.keys()
    else:
        # If using tpc_count, attempt to validate it
        if (pysmctrl and tpc_count > pysmctrl.get_tpc_info_cuda(args.device)):
            print("The TPC count must not exceed the number of available TPCs.")
            exit(22)
        part_methods = ["mps", "libsmctrl"]
        part_options = range(args.start_count, tpc_count + 1)

    for active_tpcs in part_options:
        for part_method in part_methods:
            print("Running test for %d active TPCs under %s." % (active_tpcs, part_method))
            run_process(args.device, part_method, tpc_count, active_tpcs, args.iterations)
            print()
