// This file defines the tool used for launching GPU benchmarks, contained in
// shared libraries, as either threads or processes. Supported shared libraries
// must implement the RegisterFunctions(...) function as defined in
// library_interface.h.
//
// Usage: ./runner [optional arguments] <list of .so files>
// Running with the --help argument will print more detailed usage information.
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include "library_interface.h"

// Holds data about a given benchmark, in addition to configuration parameters.
typedef struct {
  BenchmarkLibraryFunctions benchmark;
  InitializationParameters parameters;
  // Limits on how long the benchmark should run.
  int max_iterations;
  double max_seconds;
  // The file into which timing information will be written.
  FILE *output_file;
} BenchmarkConfig;

// Returns the current system time in seconds. Exits if an error occurs while
// getting the time.
static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

// Formats the given timing information as a JSON object and appends it to the
// output file. Returns 0 on error and 1 on success. Times will be written in a
// floatig-point number of *seconds*, even though they are recorded in ns. This
// code should not be included in benchmark timing measurements.
static int WriteTimesToOutput(FILE *output, TimingInformation *times) {
  uint64_t i;
  if (fprintf(output, ",\n{\"kernel times\": [") < 0) {
    return 0;
  }
  for (i = 0; i < times->kernel_times_count; i++) {
    if (fprintf(output, "%f%s", (double) times->kernel_times[i] / 1e9,
      (i != (times->kernel_times_count - 1)) ? "," : "") < 0) {
      return 0;
    }
  }
  if (fprintf(output, "], \"block times\": [") < 0) {
    return 0;
  }
  for (i = 0; i < times->block_times_count; i++) {
    if (fprintf(output, "%f%s", (double) times->block_times[i] / 1e9,
      (i != (times->block_times_count - 1)) ? "," : "") < 0) {
      return 0;
    }
  }
  if (fprintf(output, "]}") < 0) return 0;
  fflush(output);
  return 1;
}

// Runs a single benchmark. This is usually called from a separate thread or
// process. It takes a pointer to a BenchmarkConfig struct as an argument.
// It may print a message and return NULL on error. On success, it will simply
// return some value. RegisterFunctions has already been called for this
// benchmark, so the BenchmarkLibraryFunctions have been populated.
static void* RunBenchmark(void *data) {
  BenchmarkConfig *config = (BenchmarkConfig *) data;
  BenchmarkLibraryFunctions *benchmark = &(config->benchmark);
  TimingInformation timing_info;
  void *user_data = NULL;
  const char *name = benchmark->get_name();
  int i;
  double start_time;
  printf("Benchmark %s started. PID %d, TID %d\n", name, getpid(), gettid());
  user_data = benchmark->initialize(&(config->parameters));
  if (!user_data) {
    printf("Benchmark %s initialization failed.\n", name);
    return NULL;
  }
  start_time = CurrentSeconds();
  for (i = 0; i < config->max_iterations; i++) {
    if ((CurrentSeconds() - start_time) >= config->max_seconds) break;
    // The copy_in, execute, and cleanup functions can be NULL. If so, ignore
    // them.
    if (benchmark->copy_in && !benchmark->copy_in(user_data)) {
      printf("Benchmark %s copy in failed.\n", name);
      return NULL;
    }
    if (benchmark->execute && !benchmark->execute(user_data)) {
      printf("Benchmark %s execute failed.\n", name);
      return NULL;
    }
    if (!benchmark->copy_out(user_data, &timing_info)) {
      printf("Benchmark %s failed copying out.\n", name);
      return NULL;
    }
    if (!WriteTimesToOutput(config->output_file, &timing_info)) {
      printf("Benchmark %s failed writing to output file.\n", name);
      return NULL;
    }
  }
  if (benchmark->cleanup) benchmark->cleanup(user_data);
  return NULL;
}

static void* OpenLibrarym

// TODO: Implement main function
// man pthread_create
// man pthread_join
// man dlopen
// man dlsym
//
// Have option to run as processes or run as threads
// Have option to configure each benchmark independently.
