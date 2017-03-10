// This file defines the tool used for launching GPU benchmarks, contained in
// shared libraries, as either threads or processes. Supported shared libraries
// must implement the RegisterFunctions(...) function as defined in
// library_interface.h.
//
// Usage: ./runner [optional arguments] <list of .so files>
// Running with the --help argument will print more detailed usage information.
#include <dlfcn.h>
#include <errno.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "library_interface.h"
#include "parse_config.h"

// If a default log filename must be created, it will be limited to this number
// of bytes.
#define MAX_TEMP_FILENAME_SIZE (1000)

// A function pointer type for the registration function that benchmarks must
// export.
typedef int (*RegisterFunctionsFunction)(BenchmarkLibraryFunctions *functions);

// Holds data about a given benchmark, in addition to configuration parameters.
typedef struct {
  BenchmarkLibraryFunctions benchmark;
  InitializationParameters parameters;
  // Limits on how long the benchmark should run.
  int64_t max_iterations;
  double max_seconds;
  // The file into which timing information will be written.
  FILE *output_file;
  // The handle to the benchmark's shared library file, returned by dlopen.
  void *library_handle;
} ProcessConfig;

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
// process. It takes a pointer to a ProcessConfig struct as an argument.
// It may print a message and return NULL on error. On success, it will simply
// return some value. RegisterFunctions has already been called for this
// benchmark, so the BenchmarkLibraryFunctions have been populated.
static void* RunBenchmark(void *data) {
  ProcessConfig *config = (ProcessConfig *) data;
  BenchmarkLibraryFunctions *benchmark = &(config->benchmark);
  TimingInformation timing_info;
  void *user_data = NULL;
  const char *name = benchmark->get_name();
  int i;
  double start_time;
  printf("Benchmark %s started.\n", name);
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

// Allocates and returns a string containing the path to the benchmark's log
// file. Returns NULL on error. The returned path should be freed once no
// longer needed (e.g. the file has been opened).
static char* GetLogFileName(GlobalConfiguration *config, int benchmark_index) {
  SingleBenchmarkConfiguration *benchmark = config->benchmarks +
    benchmark_index;
  char temp_name[MAX_TEMP_FILENAME_SIZE];
  char *to_return = NULL;
  int total_path_length;
  memset(temp_name, 0, sizeof(temp_name));
  // Set temp_name to either the given log filename or generate one using the
  // name of the so filie along with the benchmark's index for uniqueness.
  if (benchmark->log_name) {
    // Return an absolute path if one was given.
    if (benchmark->log_name[0] == '/') return strdup(benchmark->log_name);
    snprintf(temp_name, sizeof(temp_name), "%s", benchmark->log_name);
  } else {
    snprintf(temp_name, sizeof(temp_name), "%s_%d.json",
      basename(benchmark->filename), benchmark_index);
  }
  // Make sure the null-terminator is still present.
  if (temp_name[sizeof(temp_name) - 1] != 0) {
    printf("The log file name was too long (limit: %d characters).\n",
      MAX_TEMP_FILENAME_SIZE - 1);
    return NULL;
  }
  // Add 2 characters for the path separator and null terminator.
  total_path_length = 2 + strlen(config->base_result_directory) +
    strlen(temp_name);
  to_return = (char *) malloc(total_path_length);
  if (!to_return) {
    printf("Failed allocating space for a log file name.\n");
    return NULL;
  }
  memset(to_return, 0, total_path_length);
  strcat(to_return, config->base_result_directory);
  strcat(to_return, "/");
  strcat(to_return, temp_name);
  return to_return;
}

// Takes the global configuration, then allocates and returns a list of
// ProcessConfig structures, which can then be used to kick off the actual
// benchmark processes. Returns 0 on error. The process_config_list will hold
// exactly config->benchmark_count entries. FreeGlobalConfigruration must not
// be called until after CleanupProcessConfigs has been called.
static ProcessConfig* CreateProcessConfigs(GlobalConfiguration *config) {
  RegisterFunctionsFunction register_functions = NULL;
  SingleBenchmarkConfiguration *benchmark = NULL;
  char *log_name;
  int i = 0;
  ProcessConfig *new_list = NULL;
  new_list = (ProcessConfig *) malloc(config->benchmark_count *
    sizeof(ProcessConfig));
  if (!new_list) {
    printf("Failed allocating process config list.\n");
    return NULL;
  }
  memset(new_list, 0, config->benchmark_count * sizeof(ProcessConfig));
  for (i = 0; i < config->benchmark_count; i++) {
    benchmark = config->benchmarks + i;
    // The time, iterations and cuda device can either be the global setting or
    // something benchmark-specific.
    new_list[i].max_iterations = config->max_iterations;
    if (benchmark->max_iterations >= 0) {
      new_list[i].max_iterations = benchmark->max_iterations;
    }
    new_list[i].max_seconds = config->max_time;
    if (benchmark->max_time >= 0) {
      new_list[i].max_seconds = benchmark->max_time;
    }
    new_list[i].parameters.cuda_device = config->cuda_device;
    if (benchmark->cuda_device != USE_DEFAULT_DEVICE) {
      new_list[i].parameters.cuda_device = benchmark->cuda_device;
    }
    // The thread count, block count, data size, and additional info all must
    // be specified in the benchmark-specific settings.
    new_list[i].parameters.thread_count = benchmark->thread_count;
    new_list[i].parameters.block_count = benchmark->block_count;
    new_list[i].parameters.data_size = benchmark->data_size;
    new_list[i].parameters.additional_info = benchmark->additional_info;
    // Now that all the easy information has been gathered, open the log file.
    log_name = GetLogFileName(config, i);
    if (!log_name) goto ErrorCleanup;
    // TODO: Make sure that no two processes open the same log file.
    new_list[i].output_file = fopen(log_name, "wb");
    if (!new_list[i].output_file) {
      printf("Failed opening output file %s: %s\n", log_name, strerror(errno));
      free(log_name);
      goto ErrorCleanup;
    }
    free(log_name);
    // Finally, open the shared library and get the function pointers.
    new_list[i].library_handle = dlopen(benchmark->filename, RTLD_NOW);
    if (!new_list[i].library_handle) {
      printf("Failed opening shared library %s: %s\n", benchmark->filename,
        dlerror());
      goto ErrorCleanup;
    }
    register_functions = (RegisterFunctionsFunction) dlsym(
      new_list[i].library_handle, "RegisterFunctions");
    if (!register_functions) {
      printf("The shared library %s didn't export RegisterFunctions.\n",
        benchmark->filename);
      goto ErrorCleanup;
    }
    if (!register_functions(&(new_list[i].benchmark))) {
      printf("The shared library %s's RegisterFunctions returned an error.\n",
        benchmark->filename);
    }
  }
  return new_list;
ErrorCleanup:
  for (i = 0; i < config->benchmark_count; i++) {
    if (new_list[i].output_file) fclose(new_list[i].output_file);
    if (new_list[i].library_handle) dlclose(new_list[i].library_handle);
  }
  free(new_list);
  return NULL;
}

// Cleans up and frees the list of configs. Each benchmark should have already
// ended and been cleaned up by the time this is callsed.
static void CleanupProcessConfigs(ProcessConfig *process_config_list,
    int process_config_count) {
  int i;
  for (i = 0; i < process_config_count; i++) {
    fclose(process_config_list[i].output_file);
    dlclose(process_config_list[i].library_handle);
  }
  memset(process_config_list, 0, process_config_count * sizeof(ProcessConfig));
  free(process_config_list);
}

int main(int argc, char **argv) {
  GlobalConfiguration *global_config = NULL;
  ProcessConfig *process_configs = NULL;
  if (argc != 2) {
    printf("Usage: %s <config file.json>\n", argv[0]);
    return 1;
  }
  global_config = ParseConfiguration(argv[1]);
  if (!global_config) return 1;
  printf("Config parsed.\n");
  process_configs = CreateProcessConfigs(global_config);
  if (!process_configs) return 1;
  printf("Process configs created.\n");
  if (argc == 4) RunBenchmark(NULL); //////////////////DUMMY for compilation
  CleanupProcessConfigs(process_configs, global_config->benchmark_count);
  FreeGlobalConfiguration(global_config);
  return 0;
}
