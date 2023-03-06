// This file implements the functions for parsing JSON configuration files
// defined in parse_config.h and README.md.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "library_interface.h"
#include "third_party/cJSON.h"
#include "parse_config.h"

// Config files are read in chunks containing this many bytes.
#define FILE_CHUNK_SIZE (4096)

// Set to 1 to use processes rather than threads by default. The readme says
// to use threads by default, so leave this as 0.
#define DEFAULT_USE_PROCESSES (0)

#define DEFAULT_BASE_RESULT_DIRECTORY "./results"

// Returns 0 if any key in the given cJSON config isn't in the given list of
// valid keys, and nonzero otherwise. The cJSON object must refer to the first
// sibling.
static int VerifyConfigKeys(cJSON *config, char **valid_keys,
  int valid_keys_count) {
  int i, found;
  while (config != NULL) {
    found = 0;
    if (!config->string) {
      printf("Found a setting without a name in the config.\n");
      return 0;
    }
    for (i = 0; i < valid_keys_count; i++) {
      if (strncmp(config->string, valid_keys[i], strlen(valid_keys[i])) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("Unknown setting in config: %s\n", config->string);
      return 0;
    }
    config = config->next;
  }
  return 1;
}

// Ensures that all JSON key names in global config are known. Returns 0 if an
// unknown setting is found, and nonzero otherwise.
static int VerifyGlobalConfigKeys(cJSON *main_config) {
  int keys_count = 0;
  char *valid_keys[] = {
    "name",
    "max_iterations",
    "max_time",
    "use_processes",
    "cuda_device",
    "base_result_directory",
    "pin_cpus",
    "do_warmup",
    "benchmarks",
    "comment",
    "sync_every_iteration",
  };
  keys_count = sizeof(valid_keys) / sizeof(char*);
  return VerifyConfigKeys(main_config, valid_keys, keys_count);
}

// Ensures that all JSON key names in a benchmark config are known. Returns 0
// if an unknown setting is found, and nonzero otherwise.
static int VerifyBenchmarkConfigKeys(cJSON *benchmark_config) {
  int keys_count = 0;
  char *valid_keys[] = {
    "filename",
    "log_name",
    "label",
    "thread_count",
    "block_count",
    "data_size",
    "additional_info",
    "max_iterations",
    "max_time",
    "release_time",
    "cpu_core",
    "stream_priority",
    "mps_thread_percentage",
    "comment",
  };
  keys_count = sizeof(valid_keys) / sizeof(char*);
  return VerifyConfigKeys(benchmark_config, valid_keys, keys_count);
}

// Used for parsing block_count and thread_count. The cJSON object must either
// be a single number or an array containing 1, 2, or 3 entries, all of which
// must be numbers. Returns 0 on error, including if any numbers are negative,
// the entry isn't a number or an array, if the array is too large or empty,
// etc.
static int ParseDim3OrInt(cJSON *entry, int *dim3) {
  cJSON *element = NULL;
  int i, array_length;
  // All entries in the dimensions default to 1, so setting only lower values
  // will be valid.
  for (i = 0; i < 3; i++) {
    dim3[i] = 1;
  }

  // If it's a number, we can just return right away.
  if (entry->type == cJSON_Number) {
    if (entry->valueint <= 0) {
      printf("Block and grid dims must be positive.\n");
      return 0;
    }
    dim3[0] = entry->valueint;
    return 1;
  }

  if ((entry->type != cJSON_Array) || (!entry->child)) {
    printf("Block and grid dims must either be a number or non-empty array\n");
    return 0;
  }
  array_length = 1;
  element = entry->child;
  element = element->next;
  // Walk the list to figure out the length.
  while (element) {
    array_length++;
    if (array_length > 3) {
      printf("Block and grid dims may have at most 3 entries.\n");
      return 0;
    }
    element = element->next;
  }

  // "Rewind" back to the first array element and parse the values.
  element = entry->child;
  for (i = 0; i < array_length; i++) {
    if (element->type != cJSON_Number) {
      printf("Block and grid dim array entries must be numbers.\n");
      return 0;
    }
    if (element->valueint <= 0) {
      printf("Block and grid dim array entries must be positive.\n");
      return 0;
    }
    dim3[i] = element->valueint;
  }

  return 1;
}

// Parses the list of individaul benchmark settings, starting with the entry
// given by list_start. The list_start entry must have already been valideated
// when this is called. On error, this will return 0 and leave the config
// object unmodified. On success, this returns 1.
static int ParseBenchmarkList(GlobalConfiguration *config, cJSON *list_start) {
  cJSON *current_benchmark = NULL;
  cJSON *entry = NULL;
  int benchmark_count = 1;
  int i;
  size_t benchmarks_size = 0;
  BenchmarkConfiguration *benchmarks = NULL;
  // Start by counting the number of benchmarks in the array and allocating
  // memory.
  entry = list_start;
  while (entry->next) {
    benchmark_count++;
    entry = entry->next;
  }
  benchmarks_size = benchmark_count * sizeof(BenchmarkConfiguration);
  benchmarks = (BenchmarkConfiguration *) calloc(1, benchmarks_size);
  if (!benchmarks) {
    printf("Failed allocating space for the benchmark list.\n");
    return 0;
  }
  // Next, traverse the array and fill in our parsed copy.
  current_benchmark = list_start;
  for (i = 0; i < benchmark_count; i++) {
    if (!VerifyBenchmarkConfigKeys(current_benchmark->child)) {
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "filename");
    if (!entry || (entry->type != cJSON_String)) {
      printf("Missing/invalid benchmark filename in the config file.\n");
      goto ErrorCleanup;
    }
    benchmarks[i].filename = strdup(entry->valuestring);
    if (!benchmarks[i].filename) {
      printf("Failed copying benchmark filename.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "log_name");
    if (entry) {
      if (entry->type != cJSON_String) {
        printf("Invalid benchmark log_name in the config file.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].log_name = strdup(entry->valuestring);
      if (!benchmarks[i].log_name) {
        printf("Failed copying benchmark log file name.\n");
        goto ErrorCleanup;
      }
    }
    entry = cJSON_GetObjectItem(current_benchmark, "label");
    if (entry) {
      if (entry->type != cJSON_String) {
        printf("Invalid benchmark label in the config file.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].label = strdup(entry->valuestring);
      if (!benchmarks[i].label) {
        printf("Failed copying benchmark label.\n");
        goto ErrorCleanup;
      }
    }
    benchmarks[i].mps_thread_percentage = 100.0;
    entry = cJSON_GetObjectItem(current_benchmark, "mps_thread_percentage");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid mps_thread_percentage setting.\n");
        goto ErrorCleanup;
      }
      if ((entry->valuedouble <= 0) || (entry->valuedouble > 100)) {
        printf("Invalid mps_thread_percentage: %f\n", entry->valuedouble);
        goto ErrorCleanup;
      }
      benchmarks[i].mps_thread_percentage = entry->valuedouble;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "thread_count");
    if (!ParseDim3OrInt(entry, benchmarks[i].block_dim)) {
      printf("Missing/invalid benchmark thread_count in config.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "block_count");
    if (!ParseDim3OrInt(entry, benchmarks[i].grid_dim)) {
      printf("Missing/invalid benchmark block_count in config.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "data_size");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid benchmark data_size in config.\n");
      goto ErrorCleanup;
    }
    // As with max iterations (both benchmark-specific and default), use
    // valuedouble for a better range. valueint is just a cast double already.
    benchmarks[i].data_size = entry->valuedouble;
    entry = cJSON_GetObjectItem(current_benchmark, "additional_info");
    if (entry) {
      benchmarks[i].additional_info = cJSON_PrintUnformatted(entry);
      if (!benchmarks[i].additional_info) {
        printf("Error copying additional info JSON.\n");
        goto ErrorCleanup;
      }
    }
    entry = cJSON_GetObjectItem(current_benchmark, "max_iterations");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid benchmark max_iterations in config.\n");
        goto ErrorCleanup;
      }
      // As with data_size, valuedouble provides a better range than valueint.
      benchmarks[i].max_iterations = entry->valuedouble;
      // We can't sync every iteration if some benchmarks will never reach the
      // barrier due to different numbers of iterations.
      if (config->sync_every_iteration) {
        printf("sync_every_iteration must be false if benchmark-specific "
          "iteration counts are used.\n");
        goto ErrorCleanup;
      }
      // We can't sync every iteration if different benchmarks run different
      // numbers of iterations.
      config->sync_every_iteration = 0;
    } else {
      // Remember, 0 means unlimited, negative means unset.
      benchmarks[i].max_iterations = -1;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "max_time");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid benchmark max_time in config.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].max_time = entry->valuedouble;
    } else {
      // As with max_iterations, negative means the value wasn't set.
      benchmarks[i].max_time = -1;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "release_time");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid benchmark release_time in config.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].release_time = entry->valuedouble;
    } else {
      benchmarks[i].release_time = 0;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "cpu_core");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid benchmark CPU core in config.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].cpu_core = entry->valueint;
    } else {
      benchmarks[i].cpu_core = USE_DEFAULT_CPU_CORE;
    }
    entry = cJSON_GetObjectItem(current_benchmark, "stream_priority");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid stream priority in config.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].stream_priority = entry->valueint;
    } else {
      benchmarks[i].stream_priority = USE_DEFAULT_STREAM_PRIORITY;
    }
    current_benchmark = current_benchmark->next;
  }
  config->benchmarks = benchmarks;
  config->benchmark_count = benchmark_count;
  return 1;
ErrorCleanup:
  // This won't free anything we didn't allocate, because we zero the entire
  // benchmarks array after allocating it.
  for (i = 0; i < benchmark_count; i++) {
    if (benchmarks[i].filename) free(benchmarks[i].filename);
    if (benchmarks[i].log_name) free(benchmarks[i].log_name);
    if (benchmarks[i].additional_info) free(benchmarks[i].additional_info);
    if (benchmarks[i].label) free(benchmarks[i].label);
  }
  free(benchmarks);
  return 0;
}

GlobalConfiguration* ParseConfiguration(const char *config) {
  GlobalConfiguration *to_return = NULL;
  cJSON *root = NULL;
  cJSON *entry = NULL;
  int tmp;
  to_return = (GlobalConfiguration *) calloc(1, sizeof(*to_return));
  if (!to_return) {
    printf("Failed allocating config memory.\n");
    return NULL;
  }
  root = cJSON_Parse(config);
  if (!root) {
    printf("Failed parsing JSON.\n");
    free(to_return);
    return NULL;
  }
  if (!VerifyGlobalConfigKeys(root->child)) {
    goto ErrorCleanup;
  }
  // Begin reading the global settings values.
  entry = cJSON_GetObjectItem(root, "max_iterations");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max_iterations in config.\n");
    goto ErrorCleanup;
  }
  // Use valuedouble here, since valueint is just a double cast to an int
  // already. Casting valuedouble to a uint64_t will be just as good, and will
  // have a bigger range.
  to_return->max_iterations = entry->valuedouble;
  if (to_return->max_iterations < 0) {
    printf("Invalid(negative) default max_iterations in config.\n");
    goto ErrorCleanup;
  }
  entry = cJSON_GetObjectItem(root, "max_time");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max_time in config.\n");
    goto ErrorCleanup;
  }
  to_return->max_time = entry->valuedouble;
  entry = cJSON_GetObjectItem(root, "use_processes");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid use_processes setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->use_processes = tmp == cJSON_True;
  } else {
    to_return->use_processes = DEFAULT_USE_PROCESSES;
  }
  entry = cJSON_GetObjectItem(root, "cuda_device");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid cuda_device number in config.\n");
    goto ErrorCleanup;
  }
  to_return->cuda_device = entry->valueint;
  // Any string entries will be copied--we have to assume that freeing cJSON
  // and/or the config content will free them otherwise.
  entry = cJSON_GetObjectItem(root, "name");
  if (!entry || (entry->type != cJSON_String)) {
    printf("Missing scenario name in config.\n");
    goto ErrorCleanup;
  }
  to_return->scenario_name = strdup(entry->valuestring);
  if (!to_return->scenario_name) {
    printf("Failed allocating memory for the scenario name.\n");
    goto ErrorCleanup;
  }
  entry = cJSON_GetObjectItem(root, "base_result_directory");
  // Like the scenario_name entry, the result directory must also be copied.
  // However, it is optional so we'll copy the default if it's not present.
  if (entry) {
    if (entry->type != cJSON_String) {
      printf("Invalid base_result_directory in config.\n");
      goto ErrorCleanup;
    }
    to_return->base_result_directory = strdup(entry->valuestring);
  } else {
    to_return->base_result_directory = strdup(DEFAULT_BASE_RESULT_DIRECTORY);
  }
  if (!to_return->base_result_directory) {
    printf("Failed allocating memory for result path.\n");
    goto ErrorCleanup;
  }
  // The pin_cpus setting defaults to 0 (false)
  entry  = cJSON_GetObjectItem(root, "pin_cpus");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid pin_cpus setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->pin_cpus = tmp == cJSON_True;
  } else {
    to_return->pin_cpus = 0;
  }
  // The sync_every_iteration setting defaults to 0 (false). This MUST be
  // parsed before checking benchmark-specific configs, to ensure that no
  // benchmark has a specific iteration count while sync_every_iteration is
  // true.
  entry = cJSON_GetObjectItem(root, "sync_every_iteration");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid sync_every_iteration setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->sync_every_iteration = tmp == cJSON_True;
  } else {
    to_return->sync_every_iteration = 0;
  }
  // The do_warmup setting defaults to 0.
  entry = cJSON_GetObjectItem(root, "do_warmup");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid do_warmup setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->do_warmup = tmp == cJSON_True;
  } else {
    to_return->do_warmup = 0;
  }
  // Finally, parse the benchmark list. Ensure that we've obtained a valid JSON
  // array for the benchmarks before calling ParseBenchmarkList.
  entry = cJSON_GetObjectItem(root, "benchmarks");
  if (!entry || (entry->type != cJSON_Array) || !entry->child) {
    printf("Missing/invalid list of benchmarks in config.\n");
    goto ErrorCleanup;
  }
  entry = entry->child;
  if (!ParseBenchmarkList(to_return, entry)) {
    goto ErrorCleanup;
  }
  // Clean up the JSON, we don't need it anymore since all the data was copied.
  cJSON_Delete(root);
  return to_return;
ErrorCleanup:
  if (to_return->base_result_directory) free(to_return->base_result_directory);
  if (to_return->scenario_name) free(to_return->scenario_name);
  free(to_return);
  cJSON_Delete(root);
  return NULL;
}

void FreeGlobalConfiguration(GlobalConfiguration *config) {
  int i;
  BenchmarkConfiguration *benchmarks = config->benchmarks;
  for (i = 0; i < config->benchmark_count; i++) {
    free(benchmarks[i].filename);
    free(benchmarks[i].log_name);
    if (benchmarks[i].additional_info) free(benchmarks[i].additional_info);
    if (benchmarks[i].label) free(benchmarks[i].label);
  }
  free(benchmarks);
  free(config->base_result_directory);
  free(config->scenario_name);
  memset(config, 0, sizeof(*config));
  free(config);
}

