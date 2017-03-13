CUDA Scheduling Viewer
======================

About
-----

This project was intended to provide a tool for examining block-level
scheduling behavior and coscheduling performance on CUDA devices. The tool is
capable of running any benchmark which can be self-contained in a shared
library file exporting specific functions.


Compilation
-----------

This tool can only be run on a computer with a CUDA-capable GPU and with CUDA
installed. The `nvcc` command must be available on your PATH. The tool has not
been tested with devices earlier than compute capability 5.0 or CUDA versions
earlier than 8.0.

Usage
-----

The tool must be provided a JSON configuration file, which will contain
information about which benchmark libraries to run, how to run them, and what
parameters to give file. `configs/simple.json` has been provided as a minimal
example, running one instance of the `mandelbrot.so` benchmark. To run it:

```bash
./bin/runner ./configs/simple.json
```

Additionally, the character `-` may be used in place of a config file name, in
which case the tool will attempt to read a JSON configuration object from
stdin. The file will be read completely before any benchmarks begin execution.

Configuration Files
-------------------

The configuration files specify parameters passed to each benchmark along with
some global settings for the entire program.

The layout of each configuration file is as follows:

```
{
  "max_iterations": <Number. Required. Default cap on the number of iterations
    for each benchmark. 0 = unlimited.>,
  "max_time": <Number. Required. Default cap on the number of number of seconds
    to run each benchmark. 0 = unlimited.>,
  "use_processes": <Boolean, defaulting to false. If this is true, each
    benchmark is run in a separate process. Normally, they run as threads.>
  "cuda_device": <Number. Optional. The default CUDA device to use for
    benchmarks. If unset, benchmarks will not call cudaSetDevice by default.>,
  "base_result_directory": <String, defaulting to "./results". This is the
    directory into which individual JSON files from each benchmark will be
    written. It must already exist.>,
  "benchmarks": [
    {
      "filename": <String. Required. The path to the benchmark file, relative
        to the current working directory.>,
      "log_name": <String. Optional. The filename of the JSON log for this
        particular benchmark. If not provided, this benchmark's log will be
        given a default name based on its filename, process and thread ID. If
        this doesn't start with '/', it will be relative to
        base_result_directory.>,
      "thread_count": <Number. Required, but may be ignored. The number of CUDA
        threads this benchmark should use.>,
      "block_count": <Number. Required, but may be ignored. The number of CUDA
        blocks this benchmark should use.>,
      "data_size": <Number. Required, but may be ignored. The input size, in
        bytes, for the benchmark.>,
      "cuda_device": <Number. Optional. If specified, attempt to run the
        benchmark on the CUDA device with the given ID.>,
      "additional_info": <String. Optional. This can be used to pass additional
        benchmark-specific configuration parameters.>,
      "max_iterations": <Number. Optional. If specified, overrides the default
        max_iterations for this benchmark alone. 0 = unlimited.>,
      "max_time": <Number. Optional. If specified, overrides the default
        max_time for this benchmark alone. 0 = unlimited.>,
      "release_time": <Number. Optional. If set, this benchmark will sleep for
        the given number of seconds (between initialization and the start of
        the first iteration) before beginning execution.>
    },
    {
      <more benchmarks can be listed here>
    }
  ]
}
```

Output File Format
------------------

Each benchmark, when run, will generate a JSON log file at the location
specified in the configuration. If the benchmark did not complete successfully,
the JSON file may be in an invalid state. Times will be recorded as
floating-point numbers of seconds. The format of the log file is:

```
{
  "benchmark_name": "<benchmark name>",
  "thread_count": <thread count>,
  "block_count": <block count>,
  "data_size": <data size>,
  "PID": <pid>,
  "TID": <The thread ID, if benchmarks were run as threads>,
  "times": [
    {},
    {
      "kernel_times": [<start time>, <end time>, ...],
      "block_times": [<start time>, <end time>, ...]
    },
    ...
  ]
}

```

Notice that the first entry in the "times" array will be blank and should be
ignored.

Creating New Benchmarks
-----------------------

Each benchmark must be contained in a shared library and abide by the interface
specified in `src/library_interface.h`. In particular, the library must export
a `RegisterFunctions` function, which provides the addresses of further
functions to the calling program. Benchmarks, preferably, should never use
global state and instead use the `user_data` pointer returned by the
initialize function to track all state. Global state may function if only one
instance of each benchmark is run at a time, but this will never be a
limitation of the default benchmarks included in this project. All benchmarks
must use a user-created CUDA stream in order to avoid unnecessarily blocking
each other.

In general, the comments in `library_interface.h` provide an explanation for
the actions that every library-provided function is expected to carry out.
The existing libraries in `src/mandelbrot.cu` and `src/timer_spin.cu` provide
examples of working library implementations.

Benchmark libraries are invoked by the master process as follows:

 1. The shared library file is loaded using the `dlopen()` function, and the
    `RegisterFunctions` function is located using `dlysym()`.

 2. Depending on the configuration, either a new process or new thread will be
    created for each benchmark.

 3. In its own thread or process, the benchmark's `initialize` function will
    be called, which should allocate and initialize all of the local state
    necessary for one instance of the benchmark.

 4. When the benchmark begins running, a single iteration will consist of the
    benchmark's `copy_in`, `execute`, and `copy_out` functions being called, in
    that order.

 5. When enough time has elapsed or the maximum number of iterations has been
    reached, the benchmark's `cleanup` function will be called, to allow for
    the benchmark to clean up and free its local state.
