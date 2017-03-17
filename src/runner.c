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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "library_interface.h"
#include "gpu_utilities.h"
#include "parse_config.h"

// If a default log filename must be created, it will be limited to this number
// of bytes.
#define MAX_TEMP_FILENAME_SIZE (1000)

// The size of the buffer used when sanitizing JSON strings for writing to the
// log file. This shouldn't need to be very large--only metadata can contain
// user-defined strings anyway.
#define SANITIZE_JSON_BUFFER_SIZE (512)

// A function pointer type for the registration function that benchmarks must
// export.
typedef int (*RegisterFunctionsFunction)(BenchmarkLibraryFunctions *functions);

// Holds runtime data, apart from user configuration, shared between all child
// processes.
typedef struct {
  // A pointer to the GlobalConfiguration object, which will be mostly
  // redundant information but still useful for determining whether threads or
  // processes are being used.
  GlobalConfiguration *global_config;
  // The approximate GPU global timer at the start of the program.
  uint64_t starting_gpu_clock;
  // The time at the start of the program, as measured by the host, in seconds.
  double starting_seconds;
  // The maximum number of threads resident on the GPU at a time.
  int max_resident_threads;
} ParentState;

// Holds data about a given benchmark, in addition to configuration parameters.
typedef struct {
  BenchmarkLibraryFunctions benchmark;
  InitializationParameters parameters;
  // Limits on how long the benchmark should run.
  int64_t max_iterations;
  double max_seconds;
  // The number of seconds the benchmark should sleep before its first
  // iteration. If negative, the benchmark will begin iterating immediately.
  double release_time;
  // The file into which timing information will be written.
  FILE *output_file;
  // The handle to the benchmark's shared library file, returned by dlopen.
  void *library_handle;
  // The benchmark's label to include in the output file, or NULL.
  char *label;
  // The CPU core to which this process should be pinned, or negative if the
  // CPU affinity should be left as-is.
  int cpu_core;
  // Holds information shared between all benchmarks.
  ParentState *parent_state;
} ProcessConfig;

// Returns the TID of the calling thread.
static pid_t GetThreadID(void) {
  pid_t to_return = syscall(SYS_gettid);
  return to_return;
}

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
static int WriteTimesToOutput(FILE *output, TimingInformation *times,
    uint64_t base_nanoseconds) {
  // Times are printed relative to the program start time in order to make the
  // times smaller in the logs, rather than very large numbers.
  uint64_t since_start;
  uint64_t i;
  if (fprintf(output, ",\n{\"kernel times\": [") < 0) {
    return 0;
  }
  for (i = 0; i < times->kernel_times_count; i++) {
    since_start = times->kernel_times[i] - base_nanoseconds;
    if (fprintf(output, "%f%s", (double) since_start / 1e9,
      (i != (times->kernel_times_count - 1)) ? "," : "") < 0) {
      return 0;
    }
  }
  if (fprintf(output, "], \"block times\": [") < 0) {
    return 0;
  }
  for (i = 0; i < times->block_times_count; i++) {
    since_start = times->block_times[i] - base_nanoseconds;
    if (fprintf(output, "%f%s", (double) since_start / 1e9,
      (i != (times->block_times_count - 1)) ? "," : "") < 0) {
      return 0;
    }
  }
  if (fprintf(output, "]}") < 0) return 0;
  fflush(output);
  return 1;
}

// Takes a standard string and fills the output buffer with a null-terminated
// string with JSON-unsafe values properly escaped.
static void SanitizeJSONString(const char *input, char *output,
  int output_size) {
  int output_index = 0;
  memset(output, 0, output_size);
  while (*input) {
    // Ensure that we have enough space for at least one full escaped value.
    if ((output_index - 2) == output_size) break;
    // Block any non-ASCII characters
    if (*input >= 0x7f) {
      output[output_index] = '?';
      input++;
      output_index++;
      continue;
    }
    // Copy or escape acceptable characters.
    switch (*input) {
    // Backspace character
    case 0x08:
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'b';
      break;
    // Form feed character
    case 0x0c:
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'f';
      break;
    case '\r':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'r';
      break;
    case '\t':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 't';
      break;
    case '\\':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = '\\';
      break;
    case '"':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = '"';
      break;
    case '\n':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'n';
      break;
    default:
      output[output_index] = *input;
      break;
    }
    input++;
    output_index++;
  }
}

// Writes a block of metadata entries to the output JSON file. Returns 0 on
// error.
static int WriteOutputHeader(ProcessConfig *config) {
  FILE *output = config->output_file;
  char buffer[SANITIZE_JSON_BUFFER_SIZE];
  if (fprintf(output, "{\n") < 0) {
    return 0;
  }
  SanitizeJSONString(config->parent_state->global_config->scenario_name,
    buffer, sizeof(buffer));
  if (fprintf(output, "\"scenario_name\": \"%s\",\n", buffer) < 0) {
    return 0;
  }
  SanitizeJSONString(config->benchmark.get_name(), buffer, sizeof(buffer));
  if (fprintf(output, "\"benchmark_name\": \"%s\",\n", buffer) < 0) {
    return 0;
  }
  if (config->label) {
    SanitizeJSONString(config->label, buffer, sizeof(buffer));
    if (fprintf(output, "\"label\": \"%s\",\n", buffer) < 0) {
      return 0;
    }
  }
  if (fprintf(output, "\"thread_count\": %d,\n",
    config->parameters.thread_count) < 0) {
    return 0;
  }
  if (fprintf(output, "\"block_count\": %d,\n", config->parameters.block_count)
    < 0) {
    return 0;
  }
  if (fprintf(output, "\"max_resident_threads\": %d,\n",
    config->parent_state->max_resident_threads) < 0) {
    return 0;
  }
  if (fprintf(output, "\"data_size\": %lld,\n",
    (long long int) config->parameters.data_size) < 0) {
    return 0;
  }
  if (fprintf(output, "\"release_time\": %f,\n", config->release_time) < 0) {
    return 0;
  }
  if (fprintf(output, "\"PID\": %d,\n", getpid()) < 0) {
    return 0;
  }
  // Only include the POSIX thread ID if threads are used.
  if (!config->parent_state->global_config->use_processes) {
    if (fprintf(output, "\"TID\": %d,\n", (int) GetThreadID()) < 0) {
      return 0;
    }
  }
  if (fprintf(output, "\"times\": [{}") < 0) {
    return 0;
  }
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
  uint64_t i = 0;
  double start_time;
  printf("Benchmark %s started.\n", name);
  user_data = benchmark->initialize(&(config->parameters));
  if (!user_data) {
    printf("Benchmark %s initialization failed.\n", name);
    return NULL;
  }
  if (!WriteOutputHeader(config)) {
    printf("Failed writing metadata to log file.\n");
    return NULL;
  }
  if (config->release_time > 0) {
    // Convert the release time in seconds to microseconds.
    usleep(config->release_time * 1e6);
  }
  start_time = CurrentSeconds();
  while (1) {
    // Iterations and times are unlimited if they're zero.
    if (config->max_iterations > 0) {
      i++;
      if (i > config->max_iterations) break;
    }
    if (config->max_seconds > 0) {
      if ((CurrentSeconds() - start_time) >= config->max_seconds) break;
    }
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
    if (!WriteTimesToOutput(config->output_file, &timing_info,
      config->parent_state->starting_gpu_clock)) {
      printf("Benchmark %s failed writing to output file.\n", name);
      return NULL;
    }
  }
  if (benchmark->cleanup) benchmark->cleanup(user_data);
  if (fprintf(config->output_file, "\n]}") < 0) {
    printf("Failed writing footer to output file.\n");
    return NULL;
  }
  return ((void *) 1);
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

// Takes the shared program state, then allocates and returns a list of
// ProcessConfig structures, which can then be used to kick off the actual
// benchmark processes. Returns 0 on error. The process_config_list will hold
// exactly config->benchmark_count entries. FreeGlobalConfigruration must not
// be called until after CleanupProcessConfigs has been called.
static ProcessConfig* CreateProcessConfigs(ParentState *parent_state) {
  RegisterFunctionsFunction register_functions = NULL;
  SingleBenchmarkConfiguration *benchmark = NULL;
  char *log_name;
  int i = 0;
  ProcessConfig *new_list = NULL;
  GlobalConfiguration *config = parent_state->global_config;
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
    new_list[i].label = benchmark->label;
    new_list[i].parameters.cuda_device = config->cuda_device;
    // These settings must all be specified in the benchmark-specific settings.
    new_list[i].parameters.thread_count = benchmark->thread_count;
    new_list[i].parameters.block_count = benchmark->block_count;
    new_list[i].parameters.data_size = benchmark->data_size;
    new_list[i].parameters.additional_info = benchmark->additional_info;
    new_list[i].release_time = benchmark->release_time;
    // Retain a pointer to the global configuration.
    new_list[i].parent_state = parent_state;
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
      fflush(stdout);
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

// Runs the list of configurations as threads. Returns 0 on error.
static int RunAsThreads(ParentState *parent_state,
    ProcessConfig *process_configs) {
  pthread_t *threads = NULL;
  int i, result, to_return;
  void *thread_result;
  int process_config_count = parent_state->global_config->benchmark_count;
  threads = (pthread_t *) malloc(process_config_count * sizeof(pthread_t));
  if (!threads) {
    printf("Failed allocating space to hold thread IDs.\n");
    return 0;
  }
  to_return = 1;
  memset(threads, 0, process_config_count * sizeof(pthread_t));
  for (i = 0; i < process_config_count; i++) {
    result = pthread_create(threads + i, NULL, RunBenchmark,
      process_configs + i);
    if (result != 0) {
      printf("Failed starting a thread: %d\n", result);
      to_return = 0;
      break;
    }
  }
  // Wait on threads in reverse order, in case not all of them were created.
  i--;
  for (; i >= 0; i--) {
    result = pthread_join(threads[i], &thread_result);
    if (result != 0) {
      printf("Failed joining thread %d: %d\n", i, result);
      to_return = 0;
      continue;
    }
    if (!thread_result) {
      printf("A child thread exited with an error.\n");
      to_return = 0;
    }
  }
  free(threads);
  return to_return;
}

// Like RunAsThreads, but runs benchmarks in separate processes instead.
static int RunAsProcesses(ParentState *parent_state,
    ProcessConfig *process_configs) {
  pid_t *pids = NULL;
  int i;
  pid_t child_pid = 0;
  int child_status;
  int all_ok = 1;
  int process_config_count = parent_state->global_config->benchmark_count;
  pids = (pid_t *) malloc(process_config_count * sizeof(pid_t));
  if (!pids) {
    printf("Failed allocating space to hold PIDs.\n");
    return 0;
  }
  memset(pids, 0, process_config_count * sizeof(pid_t));
  for (i = 0; i < process_config_count; i++) {
    child_pid = fork();
    // The parent process can keep generating child processes
    if (child_pid != 0) {
      pids[i] = child_pid;
      continue;
    }
    // The child process will run its benchmark and exit with a success if
    // everything went OK.
    if (!RunBenchmark(process_configs + i)) {
      exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
  }
  // As the parent, ensure that each child exited and exited with EXIT_SUCCESS.
  for (i = 0; i < process_config_count; i++) {
    waitpid(pids[i], &child_status, 0);
    if (!WIFEXITED(child_status)) {
      printf("A child process ended without exiting properly.\n");
      all_ok = 0;
    } else if (WEXITSTATUS(child_status) != EXIT_SUCCESS) {
      printf("A child process exited with an error.\n");
      all_ok = 0;
    }
  }
  free(pids);
  return all_ok;
}

int main(int argc, char **argv) {
  int result;
  ParentState parent_state;
  ProcessConfig *process_configs = NULL;
  GlobalConfiguration *global_config = NULL;
  memset(&parent_state, 0, sizeof(parent_state));
  if (argc != 2) {
    printf("Usage: %s <config file.json>\n", argv[0]);
    return 1;
  }
  // First, load the configuration file
  global_config = ParseConfiguration(argv[1]);
  if (!global_config) return 1;
  printf("Config parsed.\n");
  parent_state.global_config = global_config;
  // Next, get information about the GPU being used.
  parent_state.max_resident_threads = GetMaxResidentThreads(
    global_config->cuda_device);
  if (parent_state.max_resident_threads == 0) {
    printf("Couldn't get max resident GPU threads. Continuing anyway...\n");
  }
  // Next, create the structures that will be passed to each thread or process.
  process_configs = CreateProcessConfigs(&parent_state);
  if (!process_configs) return 1;
  printf("Process configs created.\n");
  // After the heavy initialization work has been done, record an approximate
  // GPU time and system time.
  parent_state.starting_gpu_clock = GetCurrentGPUNanoseconds(
    global_config->cuda_device);
  if (parent_state.starting_gpu_clock == 0) {
    printf("Couldn't read initial GPU time. Continuing anyway...\n");
  }
  parent_state.starting_seconds = CurrentSeconds();
  // Finally, run the benchmarks in threads or processes
  if (global_config->use_processes) {
    result = RunAsProcesses(&parent_state, process_configs);
  } else {
    result = RunAsThreads(&parent_state, process_configs);
  }
  // Last, clean up allocated state.
  CleanupProcessConfigs(process_configs, global_config->benchmark_count);
  FreeGlobalConfiguration(global_config);
  if (!result) {
    printf("An error occurred in one or more benchmarks.\n");
    return 1;
  }
  printf("All benchmarks completed OK.\n");
  return 0;
}
