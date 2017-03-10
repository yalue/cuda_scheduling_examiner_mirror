// This file implements the functions for parsing JSON configuration files
// defined in parse_config.h and README.md.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "third_party/cJSON.h"
#include "parse_config.h"

// Config files are read in chunks containing this many bytes.
#define FILE_CHUNK_SIZE (4096)

// Set to 1 to use processes rather than threads by default. The readme says
// to use threads by default, so leave this as 0.
#define DEFAULT_USE_PROCESSES (0)

#define DEFAULT_BASE_RESULT_DIRECTORY "./results"

// Wraps realloc, and attempts to resize the given buffer to the new_size.
// Returns 0 on error and leaves buffer unchanged. Returns 1 on success. If
// buffer is NULL, this will allocate memory. If new_size is 0, this wil free
// memory and set buffer to NULL.
static int SetBufferSize(void **buffer, size_t new_size) {
  void *new_pointer = NULL;
  if (new_size == 0) {
    free(*buffer);
    *buffer = NULL;
    return 1;
  }
  new_pointer = realloc(old_pointer, new_size);
  if (!new_pointer) return 0;
  *buffer = new_pointer;
  return 1;
}

// Takes the name of the configuration file and returns a pointer to a buffer
// containing its content. This will return NULL on error. On success, the
// returned buffer must be passed to free(...) when no longer needed. May print
// a message to stdout if an error occurs.
static uint8_t* GetConfigFileContent(const char *filename) {
  FILE *config_file = NULL;
  uint8_t *raw_content = NULL;
  uint8_t *current_chunk_start = NULL;
  size_t total_bytes_read = 0;
  size_t last_bytes_read = 0;
  // Remember that a filename of "-" indicates to use stdin.
  if (strncmp(filename, "-", 2) == 0) {
    config_file = stdin;
  } else {
    config_file = fopen(filename, "rb");
    if (!config_file) {
      printf("Failed opening config file.\n");
      return NULL;
    }
  }
  if (!SetBufferSize(&raw_content, FILE_CHUNK_SIZE)) {
    printf("Failed allocating buffer for config file content.\n");
    if (config_file != stdin) fclose(config_file);
    return NULL;
  }
  // It would be far nicer to just allocate a chunk of memory at once, but then
  // there's no way to use stdin, since we don't know the size ahead of time.
  // Also, we need to full buffer in order to parse the JSON.
  while (1) {
    current_chunk_start = raw_content + bytes_read;
    last_bytes_read = fread(current_chunk_start, 1, FILE_CHUNK_SIZE,
      config_file);
    // If we failed to read an entire chunk, we're either at the end of the
    // file or we encountered an error.
    if (last_bytes_read != FILE_CHUNK_SIZE) {
      if (!feof(config_file) || ferror(config_file)) {
        printf("Error reading the config file: %s\n", strerror(errno));
        free(raw_content);
        if (config_file != stdin) fclose(config_file);
        return NULL;
      }
      total_bytes_read += last_bytes_read;
      break;
    }
    // Allocate space for another chunk of the file to be read.
    bytes_read += FILE_CHUNK_SIZE;
    if (!SetBufferSize(&raw_content, bytes_read + FILE_CHUNK_SIZE)) {
      printf("Failed obtaining more memory for the config file.\n");
      free(raw_content);
      if (config_file != stdin) fclose(config_file);
      return NULL;
    }
  }
  if (config_file != stdin) fclose(config_file);
  return raw_content;
}

// Parses the list of individaul benchmark settings, starting with the entry
// given by list_start. The list_start entry must have already been valideated
// when this is called. On error, this will return 0 and leave the config
// object unmodified. On success, this returns 1.
static int ParseBenchmarkList(GlobalConfig *config, cJSON *list_start) {
  cJSON *current_benchmark = NULL;
  cJSON *entry = NULL;
  int benchmark_count = 1;
  size_t benchmarks_size = 0;
  SingleBenchmarkConfiguration *benchmarks = NULL;
  // Start by counting the number of benchmarks in the array and allocating
  // memory.
  entry = list_start;
  while (entry->next) {
    benchmark_count++;
  }
  benchmarks_size = benchmark_count * sizeof(SingleBenchmarkConfiguration);
  benchmarks = (SingleBenchmarkConfiguration *) malloc(benchmarks_size);
  if (!benchmarks) {
    printf("Failed allocating space for the benchmark list.\n");
    return 0;
  }
  memset(benchmarks, 0, benchmarks_size);
  // Next, traverse the array and fill in our parsed copy.
  current_benchmark = list_start;
  for (i = 0; i < benchmark_count; i++) {
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
        printf("Invalid benchmark log name in the config file.\n");
        goto ErrorCleanup;
      }
      benchmarks[i].log_name = strdup(entry->valuestring);
      if (!benchmarks[i].log_name) {
        printf("Failed copying benchmark log file name.\n");
        goto ErrorCleanup;
      }
    }
    // TODO: Continue parsing individual benchmark JSON configs:
    //  - thread_count, reqd
    //  - block_count, reqd
    //  - data_size, reqd
    //  - cuda_device, optional
    //  - additional_info, optional string
    //  - max_iterations, optional int64
    //  - max_time, optional double
    //  - release_time, optional double
    current_benchmark = current_benchmark->next;
  }
  return 1;
ErrorCleanup:
  // This won't free anything we didn't allocate, because we zero the entire
  // benchmarks array after allocating it.
  for (i = 0; i < benchmark_count; i++) {
    if (benchmarks[i].filename) free(benchmarks[i].filename);
    if (benchmarks[i].log_name) free(benchmarks[i].log_name);
    if (benchmarks[i].additional_info) free(benchmarks[i].additional_info);
  }
  free(benchmarks);
  return 0;
}

GlobalConfiguration* ParseConfiguration(const char *filename) {
  GlobalConfiguration *to_return = NULL;
  cJSON *root = NULL;
  cJSON *entry = NULL;
  uint8_t *raw_content = GetConfigFileContent(filename);
  int tmp;
  if (!raw_content) return NULL;
  to_return = (GlobalConfiguration *) malloc(sizeof(*to_return));
  if (!to_return) {
    printf("Failed allocating config memory.\n");
    free(raw_content);
    return NULL;
  }
  memset(to_return, 0, sizeof(*to_return));
  root = cJSON_Parse(raw_content);
  if (!root) {
    printf("Failed parsing JSON.\n");
    free(raw_content);
    free(to_return);
    return NULL;
  }
  // Begin reading the global settings values.
  entry = cJSON_GetObjectItem(root, "max_iterations");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max iterations in config.\n");
    goto ErrorCleanup;
  }
  // TODO: Use valuedouble here for a better range?
  to_return->max_iterations = entry->valueint;
  entry = cJSON_GetObjectItem(root, "max_time");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max time in config.\n");
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
  entry = cJSON_GetObjectItem(root, "base_result_directory");
  // Any string entries will be copied--we have to assume that freeing cJSON
  // and/or the raw_content will free them otherwise.
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
  // Clean up the raw content and JSON--we don't need them anymore since all
  // the data was copied.
  free(raw_content);
  cJSON_Delete(root);
  return to_return;
ErrorCleanup:
  if (to_return->base_result_directory) free(to_return->base_result_directory);
  free(raw_content);
  free(to_return);
  cJSON_Delete(root);
  return NULL;
}

void FreeGlobalConfiguration(GlobalConfiguration *config) {
  int i;
  SingleBenchmarkConfiguration *benchmarks = config->benchmarks;
  for (i = 0; i < config->benchmark_count; i++) {
    free(benchmarks[i].filename);
    free(benchmarks[i].log_name);
    if (benchmarks[i].additional_info) free(benchmarks[i].additional_info);
  }
  free(benchmarks);
  free(config->base_result_directory);
  memset(config, 0, sizeof(*config));
  free(config);
}

