// This file defines the interface and types needed when parsing the JSON
// configuration files used by runner.c.
#ifndef PARSE_CONFIG_H
#define PARSE_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#define USE_DEFAULT_CPU_CORE (-1)
#define USE_DEFAULT_STREAM_PRIORITY (-100)

// Holds the configuration for a single benchmark, as specified by the JSON
// format given in the README.
typedef struct {
  // The name of the benchmark's .so file
  char *filename;
  // The name of the benchmark's JSON log, relative to base_result_directory.
  // If NULL, a log name will be used based on the benchmark's filename.
  char *log_name;
  // An extra label or name for the benchmark, included in its JSON log file.
  char *label;
  // The limit on thread resources used by MPS. Only used if the task is run as
  // a process, MPS is active, and a Volta GPU is used.
  double mps_thread_percentage;
  // Specifies the dimensions of a single block, in terms of the number of
  // threads. Contains the x, y, and z dimensions, in that order.
  int block_dim[3];
  // Specifies the number of blocks to use when launching the kernel. Contains
  // the x, y, and z dimensions, in that order.
  int grid_dim[3];
  // The size, in bytes, of the input data the benchmark should generate or use
  uint64_t data_size;
  // A string containing an additional user-defined argument to pass to the
  // benchmark during initialization. May be either NULL or empty if
  // unspecified.
  char *additional_info;
  // The maximum number of iterations for this benchmark alone; will override
  // the global limit if set (0 = unlimited, negative = unset).
  int64_t max_iterations;
  // The maximum number of seconds to run this benchmark alone; will override
  // the global limit if set (0 = unlimited, negative = unset).
  double max_time;
  // The number of seconds for which the benchmark should sleep before
  // starting. If 0 or negative, it won't sleep at all.
  double release_time;
  // The CPU core to pin this benchmark to. Ignored if negative.
  int cpu_core;
  // The stream priority used to create the CUDA stream. Ignored if negative.
  int stream_priority;
  // SM enablement mask
  uint64_t sm_mask;
} BenchmarkConfiguration;

// Holds default settings for all benchmarks, and a list of individual
// benchmarks with their specific settings.
typedef struct {
  // The default cap on the number of iterations each benchmark will run.
  // Unlimited if 0 or lower.
  int64_t max_iterations;
  // The default cap on the number of seconds each benchmark will run.
  // Unlimited if 0 or lower.
  double max_time;
  // Set to 0 if each benchmark should be run in a separate thread. Set to 1 if
  // each should be run in a child process instead.
  int use_processes;
  // The CUDA device to run benchmarks on.
  int cuda_device;
  // The path to the base directory in which benchmark's log files are stored.
  char *base_result_directory;
  // The name of the scenario being tested.
  char *scenario_name;
  // If zero, CPU assignment is either handled by the system or taken from each
  // benchmark's cpu_core setting. If nonzero, benchmarks are distributed
  // evenly accrorss CPU cores.
  int pin_cpus;
  // If nonzero, run each benchmark for one or more iterations without
  // recording performance after initialization, but before syncing with other
  // plugins and starting to take measurements.
  int do_warmup;
  // If zero, iterations of individual benchmarks run as soon as previous
  // iterations complete. If 1, then every benchmark starts each iteration
  // only after the previous iteration of every benchmark has completed.
  int sync_every_iteration;
  // The number of entries in the benchmarks list. Must never be 0.
  int benchmark_count;
  // The list of benchmarks to run.
  BenchmarkConfiguration *benchmarks;
} GlobalConfiguration;

// Parses a JSON configuration string, and allocates and returns a
// GlobalConfiguration struct. Returns NULL on error. When no longer needed,
// the returned pointer must be passed to FreeGlobalConfiguration. May print to
// stdout on error.
GlobalConfiguration* ParseConfiguration(const char *content);

void FreeGlobalConfiguration(GlobalConfiguration *config);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PARSE_CONFIG_H
