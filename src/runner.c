// This file defines the tool used for launching GPU benchmarks, contained in
// shared libraries, as either threads or processes. Supported shared libraries
// must implement the RegisterFunctions(...) function as defined in
// library_interface.h.
//
// Usage: ./runner <path to JSON config file>
// Supplying - in place of a JSON config file will cause the program to read a
// config from stdin.
#include <dlfcn.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "library_interface.h"

// Config files are read in chunks containing this many bytes.
#define FILE_CHUNK_SIZE (4096)

// A function pointer type for the registration function that benchmarks must
// export.
typedef int (*RegisterFunctionsFunction)(BenchmarkLibraryFunctions *functions);

// Wraps realloc, and attempts to resize the given buffer to the new_size.
// Returns 0 on error and leaves buffer unchanged. Returns 1 on success. If
// buffer is NULL, this will allocate memory. If new_size is 0, this wil free
// memory and set buffer to NULL. Used when reading config files, including
// stdin.
static int SetBufferSize(void **buffer, size_t new_size) {
  void *new_pointer = NULL;
  if (new_size == 0) {
    free(*buffer);
    *buffer = NULL;
    return 1;
  }
  new_pointer = realloc(*buffer, new_size);
  if (!new_pointer) return 0;
  *buffer = new_pointer;
  return 1;
}

// Takes the name of the configuration file and returns a pointer to a buffer
// containing its content. This will return NULL on error. On success, the
// returned buffer must be passed to free(...) when no longer needed. May print
// a message to stdout if an error occurs. The complexity of this function is
// due to handling stdin (with the special name "-") in addition to standard
// files.
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
  if (!SetBufferSize((void **) (&raw_content), FILE_CHUNK_SIZE)) {
    printf("Failed allocating buffer for config file content.\n");
    if (config_file != stdin) fclose(config_file);
    return NULL;
  }
  // It would be far nicer to just allocate a chunk of memory at once, but then
  // there's no way to use stdin, since we don't know the size ahead of time.
  // Also, we need to fully buffer a file in order to parse the JSON later.
  while (1) {
    current_chunk_start = raw_content + total_bytes_read;
    last_bytes_read = fread(current_chunk_start, 1, FILE_CHUNK_SIZE,
      config_file);
    // If we failed to read an entire chunk, we're either at the end of the
    // file or we encountered an error.
    if (last_bytes_read != FILE_CHUNK_SIZE) {
      if (!feof(config_file) || ferror(config_file)) {
        printf("Error reading the config.\n");
        free(raw_content);
        if (config_file != stdin) fclose(config_file);
        return NULL;
      }
      total_bytes_read += last_bytes_read;
      break;
    }
    // Allocate space for another chunk of the file to be read.
    total_bytes_read += FILE_CHUNK_SIZE;
    if (!SetBufferSize((void **) (&raw_content), total_bytes_read +
      FILE_CHUNK_SIZE)) {
      printf("Failed obtaining more memory for the config file.\n");
      free(raw_content);
      if (config_file != stdin) fclose(config_file);
      return NULL;
    }
  }
  if (config_file != stdin) fclose(config_file);
  return raw_content;
}

// Loads task_host.so. On success, this sets library_handle, which must be
// passed to dlclose when the library is no longer needed. This also fills in
// the functions struct. task_host.so is loaded from a path relative to the
// runner binary, so the executable path (argv[0]) is needed, too. Returns 0 on
// error.
static int LoadTaskHostLibrary(void **library_handle,
    BenchmarkLibraryFunctions *functions, const char *executable_path) {
  RegisterFunctionsFunction register_functions = NULL;
  char library_path[2048];
  char *executable_path_copy = NULL;
  void *handle = NULL;
  // We need to copy this string, because the dumb dirname function modifies
  // the original string.
  executable_path_copy = strdup(executable_path);
  if (!executable_path_copy) return 0;
  // Build a path string, in the same directory as the executable.
  snprintf(library_path, sizeof(library_path), "%s/task_host.so",
    dirname(executable_path_copy));
  free(executable_path_copy);
  handle = dlopen(library_path, RTLD_NOW);
  if (!handle) {
    printf("Couldn't load task host library %s: %s\n", library_path,
      dlerror());
    return 0;
  }
  register_functions = (RegisterFunctionsFunction) dlsym(handle,
    "RegisterFunctions");
  if (!register_functions) {
    printf("%s didn't export RegisterFunctions.\n", library_path);
    dlclose(handle);
    return 0;
  }
  if (!register_functions(functions)) {
    printf("%s's RegisterFunctions returned an error.\n", library_path);
    dlclose(handle);
    return 0;
  }
  *library_handle = handle;
  return 1;
}

int main(int argc, char **argv) {
  BenchmarkLibraryFunctions functions;
  InitializationParameters params;
  TimingInformation times;
  void *library_handle = NULL;
  void *task_host_data = NULL;
  char *config_content = NULL;
  int result;
  if (argc != 2) {
    printf("Usage: %s <path to JSON config file>\n", argv[0]);
    return 1;
  }
  // Read the config file and generate a set of initialization parameters. The
  // only meaningful field we set is additional_info.
  config_content = (char *) GetConfigFileContent(argv[1]);
  if (!config_content) return 1;
  memset(&params, 0, sizeof(params));
  params.additional_info = config_content;
  // Next, load the task_host library, which is always the top-level task.
  memset(&functions, 0, sizeof(functions));
  if (!LoadTaskHostLibrary(&library_handle, &functions, argv[0])) {
    printf("Failed loading the task host library.\n");
    goto ErrorCleanup;
  }
  // Initialize the top-level task. This is where the JSON actually gets parsed.
  task_host_data = functions.initialize(&params);
  if (!task_host_data) {
    printf("Failed initializing the task host.\n");
    goto ErrorCleanup;
  }
  printf("Successfully loaded %s\n", argv[1]);
  // copy_in shouldn't do anything in task_host.so, but we'll call it for
  // consistency.
  result = functions.copy_in(task_host_data);
  if (!result) {
    printf("The task host's copy_in failed.\n");
    goto ErrorCleanup;
  }
  // This is where all of the processing should take place.
  result = functions.execute(task_host_data);
  if (!result) {
    printf("Task execution encountered an error.\n");
    goto ErrorCleanup;
  }
  // Like copy_in, copy_out shouldn't do anything but we'll call it anyway.
  result = functions.copy_out(task_host_data, &times);
  if (!result) {
    printf("The task host's copy_out failed.\n");
    goto ErrorCleanup;
  }
  printf("All tasks completed successfully.\n");
  // Finally, clean up. Everything was successful.
  functions.cleanup(task_host_data);
  dlclose(library_handle);
  free(config_content);
  return 0;
ErrorCleanup:
  if (task_host_data) functions.cleanup(task_host_data);
  if (library_handle) dlclose(library_handle);
  if (config_content) free(config_content);
  return 1;
}
