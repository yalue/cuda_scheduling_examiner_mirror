# This script generates configs that test the behavior of blocking effects
# across process boundaries. The configs it generates are in a similar format
# to files like configs/rtas_2018_figure_[4567].json. Essentially, the tests
# carry out the following actions:
#
# 1. At t = 0.0s, a single 512x2 kernel is released into its own stream, and
#    runs for 1 second.
# 2. At t = 0.2s, a single 512x2 kernel is released into its own stream, and
#    runs for 2 seconds.
# 3. At t = 0.4s, an action will occur. This action is what is being tested for
#    blocking behavior. If the action is associated with a stream, it will be
#    earlier in the same stream as the kernel in step 4.
# 4. 0.2s after the action in step 3 completes, a 512x2 kernel will be released
#    potentially into a stream shared by step 3 (depending on what step 3 is).
#    If this action *isn't* blocked, it will start running at t = 0.6s.
# 5. At t = 0.8s, a single 512x2 kernel is released into its own stream. If
#    this action isn't blocked, it will start running at t = 0.8s. However, it
#    may be blocked by steps 2, 3, or 4, depending on the action carried out by
#    step 3.
#
# Usage of this script:
#
# rm results/*.json
# python scripts/test_blocking_sources.py
# python scripts/view_blocksbysm.py

import copy
import json
import subprocess

def generate_basic_task(task_number, name, duration, release_time):
    """Returns a short benchmark configuration for a timer_spin benchmark."""
    name = name.replace(" ", "_")
    duration = int(duration)
    to_return = {
        "filename": "./bin/timer_spin.so",
        "log_name": "test_blocking_source_%s_task%d.json" % (name,
            task_number),
        "label": "Task %d" % (task_number),
        "thread_count": 512,
        "block_count": 2,
        "data_size": 0,
        "additional_info": duration,
        "release_time": release_time
    }
    return to_return

def generate_blocked_task(name, blocking_source):
    """Returns a stream_action config, with the given blocking source."""
    name = name.replace(" ", "_")
    kernel_action = {
        "delay": 0.2,
        "type": "kernel",
        "label": "K3",
        "parameters": {
            "duration": 250000000
        }
    }
    actions = []
    if blocking_source is not None:
        actions.append(blocking_source)
    actions.append(kernel_action)
    to_return = {
        "filename": "./bin/stream_action.so",
        "log_name": "test_blocking_source_%s_task3.json" % (name),
        "label": "Task 3",
        "release_time": 0.4,
        "thread_count": 512,
        "block_count": 2,
        "data_size": 0,
        "additional_info": {
            "use_null_stream": False,
            "actions": actions
        }
    }
    return to_return

def generate_config(name, blocking_action):
    benchmarks = []
    benchmarks.append(generate_basic_task(1, name, 1e9, 0.0))
    benchmarks.append(generate_basic_task(2, name, 2e9, 0.2))
    benchmarks.append(generate_blocked_task(name, blocking_action))
    benchmarks.append(generate_basic_task(4, name, 1e9, 0.8))
    top_config = {
        "name": "Testing blocking: " + name,
        "max_iterations": 1,
        "max_time": 0,
        "cuda_device": 0,
        "pin_cpus": True,
        "benchmarks": benchmarks
    }
    return json.dumps(top_config)

def run_process(name, blocking_action):
    """Requires a config for a single action in stream_action.cu, that will be
    run during step 3, as described at the top of the file. blocking_action
    can be None, in which case the action will be a noop."""
    print "Running process with blocking action of " + name + "..."
    config = generate_config(name, blocking_action)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

def run_blocking_tests():
    actions = {
        "Device Synchronize": {
            "type": "synchronize",
            "label": "Device Synchronize",
            "parameters": {
                "device": True
            }
        },
        "Stream Synchronize": {
            "type": "synchronize",
            "label": "Stream Synchronize"
        },
        "Malloc": {
            "type": "malloc",
            "label": "Device Allocation",
            "parameters": {
                "host": False,
                "size": 4096
            }
        },
        "Malloc Host": {
            "type": "malloc",
            "label": "Host Allocation",
            "parameters": {
                "host": True,
                "size": 4096
            }
        },
        "Free": {
            "type": "free",
            "label": "Free",
        },
        "Free Host": {
            "type": "free",
            "label": "Free host",
            "parameters": {
                "host": True,
            }
        },
        "Memset Async": {
            "type": "memset",
            "label": "Memset Async",
            "parameters": {
                "size": 4096
            }
        },
        "Memset (synchronous)": {
            "type": "memset",
            "label": "Memset (synchronous)",
            "parameters": {
                "size": 4096,
                "async": False
            }
        },
        "Memcpy Host-to-Device": {
            "type": "memcpy",
            "label": "Memcpy Host-to-Device",
            "parameters": {
                "size": 4096,
                "direction": "hostToDevice"
            }
        },
        "Memcpy Device-to-Host": {
            "type": "memcpy",
            "label": "Memcpy Device-to-Host",
            "parameters": {
                "size": 4096,
                "direction": "deviceToHost"
            }
        },
        "Memcpy Device-to-Device": {
            "type": "memcpy",
            "label": "Memcpy Device-to-Device",
            "parameters": {
                "size": 4096,
                "direction": "deviceToDevice"
            }
        }
    }
    run_process("No blocking", None)
    for name in actions:
        run_process(name, actions[name])

run_blocking_tests()
