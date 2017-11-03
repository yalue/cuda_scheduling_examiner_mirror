// This file defines a CUDA benchmark which issues multiple kernels to a single
// stream before waiting for all kernels to complete. The configuration for the
// kernels is taken exclusively from the additional_info field in the
// InitializationParameters struct. The actual kernels will simply be instances
// of the same kernel as in the timer_spin benchmark. This benchmark ignores
// all fields in its initialization parameters apart from cuda_device and
// additional_info.
//
// The format of the necessary additional_info field is as follows. Each object
// in the "actions" list must have a type that is one of "launchKernel",
// "cudaMalloc", "cudaFree", "cudaMemset", "cudaMemcpy", or "synchronize".
// Memory operations such as malloc, free memset, and memcpy operate on buffers
// separate from each other. For example, a malloc doesn't need to precede a
// memset, because memset buffers will be allocated during initialization.
// "Free" can't come before a malloc, though. Any unfreed mallocs from these
// actions will be freed during benchmark cleanup. Synchronization actions are
// available solely to experiment with scheduling, and are not necessary for
// the task. A stream-synchronization request will be issued at the end of all
// actions regardless of whether an explicit, additional synchronization action
// was carried out.

// For more details about parameters for eac action, see the annotated JSON
// structure below:
/*
"additional_info": {
  "use_null_stream": <Boolean, defaults to false, set to true to use the
    null stream rather than the default stream>,
  "actions": [
    {
      "delay": <A floating-point number of seconds to sleep before starting
        this action. Defaults to 0.0, which will insert no sleep at all.>,
      "type": <A string, from the list given above.>,
      "label": <A string, a label for this action.>,
      "parameters": <A JSON object with action-specific parameters.>
    },
    {
      "type": "launchKernel",
      "label": "Kernel 1",
      "parameters": {
        "type": <A string: "timer_spin" or "counter_spin". Defaults to
          "timer_spin">,
        "duration": <If "type" is "timer_spin", this will be the number of
          nanoseconds to run the kernel. If type is "counter_spin", this
          will be the number of loop iterations to run.>,
        "shared_memory_size": <The number of shared 32-bit integers to use.
          Defaults to 0. Must be 0, 4096, 8192, or 10240.>,
        "block_count": <The number of thread blocks to use. Defaults to the
          value given in the benchmark parameters.>,
        "thread_count": <The number of threads per block to use. Defaults to
          the value given in the benchmark parameters.>
      },
      {
        "type": "cudaMalloc",
        "label": "Malloc 1",
        "parameters": {
          "host": <Boolean. Defaults to false. If true, will allocate host
            memory.>,
          "size": <Number of bytes to allocate>
        }
      },
      {
        "type": "cudaFree",
        "label": "Free 1",
        "parameters": {
          "host": <Boolean. Defaults to false. If true, will free host memory.
            The entire "parameters" block can be omitted here for the default.>
        }
      },
      {
        "type": "cudaMemset",
        "label": "Memset 1",
        "parameters": {
          "async": <Boolean. Defaults to true. If false, will issue a
            null-stream memset regardless of use_null_stream's value.>,
          "size": <Number of bytes to set to 0>
        }
      },
      {
        "type": "cudaMemcpy",
        "label": Memcpy 1",
        "parameters": {
          "async": <Boolean. Defaults to true. If false, issues a null-stream
            memcpy regardless of use_null_stream's value.>,
          "size": <Number of byte to copy>,
          "direction": <Either "deviceToDevice", "deviceToHost", or
            "hostToDevice">
        }
      },
      {
        "type": "synchronize",
        "label": "Sync 1",
        "parameters": {
          "device": <Boolean. Defaults to false (parameters can be omitted here
            entirely, too). If true, runs a cudaDeviceSynchronize rather than
            cudaStreamSynchronize.>
        }
      }
    }
  ]
}
*/
// Actions are issued to the stream in the same order that they're specified
// in the "actions" list.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"
#include "third_party/cJSON.h"
#include "stream_action.h"

// This specifies the maximum number of un-freed malloc actions that can occur
// before further allocations return an error instead. Any list with this many
// or fewer (balanced) malloc and free actions can run indefinitely.
#define MAX_MEMORY_ALLOCATION_COUNT (10)

// Use the macros defined in stream_action.h to generate a set of kernels using
// various amounts of static shared memory.
GENERATE_SPIN_KERNEL(4096);
GENERATE_SPIN_KERNEL(8192);
GENERATE_SPIN_KERNEL(10240);

// A basic kernel that wastes GPU cycles without using shared memory. The
// duration parameter specifies the number of nanoseconds to wait if
// use_counter is 0. If use_counter is nonzero, duration specifies a number of
// loop iterations to spin instead. The junk parameter must be NULL and is used
// to prevent optimization.
static __global__ void GPUSpin(int use_counter, uint64_t duration,
    uint64_t *block_times, uint32_t *block_smids, uint64_t *junk) {
  uint64_t i, accumulator;
  uint64_t start_time = GlobalTimer64();
  // Have one thread record the block's start time and SM ID.
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  if (use_counter) {
    // Write to the accumulator (which must be potentially returned) to prevent
    // this loop from being optimized out.
    for (i = 0; i < duration; i++) {
      accumulator += i;
    }
  } else {
    // Wait until the specified number of nanoseconds has elapsed.
    while ((GlobalTimer64() - start_time) < duration) {
      continue;
    }
  }
  // Make it look like the junk value can be used to prevent the loop updating
  // the accumulator from being removed by the optimizer.
  if (junk) *junk = accumulator;
  // Have one thread write the block end time (simple, but may be slightly
  // inaccurate if other warps finish later).
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}

// Frees any data and clears out an ActionConfig struct. For use during
// cleanup.
static void CleanupAction(ActionConfig *action) {
  uint64_t *tmp64;
  uint32_t *tmp32;
  if (action->label) free(action->label);
  if (action->type == ACTION_KERNEL) {
    // For now, only kernel actions require extra cleanup.
    tmp64 = action->parameters.kernel.device_block_times;
    if (tmp64) CheckCUDAError(cudaFree(tmp64));
    tmp64 = action->parameters.kernel.host_block_times;
    if (tmp64) CheckCUDAError(cudaFreeHost(tmp64));
    tmp32 = action->parameters.kernel.device_smids;
    if (tmp32) CheckCUDAError(cudaFree(tmp32));
    tmp32 = action->parameters.kernel.host_smids;
    if (tmp32) CheckCUDAError(cudaFreeHost(tmp32));
  }
  memset(action, 0, sizeof(*action));
}

// Implements the cleanup fucntion required by the interface, but is also used
// internally to clean up during a faulty Initialize(). That's why all of the
// pointers are checked to be non-NULL. This is also why it's very important to
// ensure that any fields and pointers are zero before any initialization.
static void Cleanup(void *data) {
  TaskState *state = (TaskState *) data;
  int i;
  ActionConfig *action = NULL;
  for (i = 0; i < state->action_count; i++) {
    action = state->actions + i;
    CleanupAction(action);
  }
  if (state->actions) free(state->actions);
  if (state->kernel_times) free(state->kernel_times);
  // The CheckCUDAError macros here are just to print a message on error, since
  // we can't really do any additional error handling during cleanup.
  if (state->stream_created) {
    // Remember that state->stream may be the NULL stream or may be another
    // reference to this same stream. In either case, we don't need to destroy
    // it.
    CheckCUDAError(cudaStreamDestroy(state->copy_out_stream));
  }
  if (state->host_copy_buffer) {
    CheckCUDAError(cudaFreeHost(state->host_copy_buffer));
  }
  if (state->device_copy_buffer) {
    CheckCUDAError(cudaFree(state->device_copy_buffer));
  }
  if (state->device_secondary_buffer) {
    CheckCUDAError(cudaFree(state->device_secondary_buffer));
  }
  for (i = 0; i < state->device_memory_allocation_count; i++) {
    CheckCUDAError(cudaFree(state->device_memory_allocations[i]));
  }
  if (state->device_memory_allocations) free(state->device_memory_allocations);
  for (i = 0; i < state->host_memory_allocation_count; i++) {
    CheckCUDAError(cudaFreeHost(state->host_memory_allocations[i]));
  }
  if (state->host_memory_allocations) free(state->host_memory_allocations);
  memset(state, 0, sizeof(*state));
  free(state);
}

// Returns nonzero if all of the keys in the JSON object are in the list of
// valid keys.
static int VerifyJSONKeys(cJSON *object, const char* const valid_keys[],
    int valid_count) {
  int i, found;
  // We'll be passed a top-level object here.
  object = object->child;
  while (object != NULL) {
    found = 0;
    if (!object->string) {
      printf("Found JSON object without a name in stream_action settings.\n");
      return 0;
    }
    for (i = 0; i < valid_count; i++) {
      if (strcmp(object->string, valid_keys[i]) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("Unexpected setting in stream_action.so settings: %s\n",
        object->string);
      return 0;
    }
    object = object->next;
  }
  return 1;
}

// Takes a cJSON object and returns 1 if it's true, 0 if it's false, and -1 if
// it's invalid or not a boolean. Returns -1 if object is NULL.
static int GetCJSONBoolean(cJSON *object) {
  if (!object) return -1;
  if (object->type == cJSON_True) return 1;
  if (object->type == cJSON_False) return 0;
  return -1;
}

// Since this is such a long string of code, it gets moved into a separate
// function. Parses the parameters for a kernel action. Requires the cJSON
// *parameters* object for a kernel action, and fills in the KernelParameters.
// Returns 0 on error.
static int ParseKernelParameters(cJSON *json_parameters,
    KernelParameters *kernel_config, int default_block_count,
    int default_thread_count) {
  cJSON *entry = NULL;
  // Due to the complexity of this config, this can forestall confusing errors
  // by pointing out misspelled keys.
  static const char* const valid_keys[] = {
    "type",
    "thread_count",
    "block_count",
    "shared_memory_size",
    "comment",
    "duration",
  };
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  // Determine whether the kernel should be a timer spin (constant time) or
  // counter spin (constant effort). The default is constant time, if the
  // setting isn't provided.
  entry = cJSON_GetObjectItem(json_parameters, "type");
  if (entry) {
    if (entry->type != cJSON_String) {
      printf("Invalid kernel type for kernel action.\n");
      return 0;
    }
    if (strcmp(entry->valuestring, "timer_spin") == 0) {
      kernel_config->use_counter_spin = 0;
    } else if (strcmp(entry->valuestring, "counter_spin") == 0) {
      kernel_config->use_counter_spin = 1;
    } else {
      printf("Unsupported kernel type for kernel action: %s\n",
        entry->valuestring);
      return 0;
    }
  } else {
    kernel_config->use_counter_spin = 0;
  }
  // Get the one required numerical parameter: duration.
  entry = cJSON_GetObjectItem(json_parameters, "duration");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid duration for kernel action.\n");
    return 0;
  }
  kernel_config->duration = (uint64_t) entry->valuedouble;
  // Get the block and thread counts, which default to the benchmark setting
  // if they aren't provided.
  kernel_config->block_count = default_block_count;
  entry = cJSON_GetObjectItem(json_parameters, "block_count");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("Invalid block count for kernel action.\n");
      return 0;
    }
    kernel_config->block_count = entry->valueint;
  }
  kernel_config->thread_count = default_thread_count;
  entry = cJSON_GetObjectItem(json_parameters, "thread_count");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("Invalid thread count for kernel action.\n");
      return 0;
    }
    kernel_config->thread_count = entry->valueint;
  }
  // Unlike the other numbers, the shared_memory_count is optional and needs
  // validation.
  entry = cJSON_GetObjectItem(json_parameters, "shared_memory_size");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("Invalid shared memory size for kernel action.\n");
      return 0;
    }
    kernel_config->shared_memory_count = entry->valueint;
  } else {
    kernel_config->shared_memory_count = 0;
  }
  switch (kernel_config->shared_memory_count) {
    case 0:
    case 4096:
    case 8192:
    case 10240:
      break;
    default:
      printf("Unsupported shared memory size for kernel action: %d\n",
        kernel_config->shared_memory_count);
      return 0;
  }
  return 1;
}

// Parses parameters for the malloc action. Returns 0 on error.
static int ParseMallocParameters(cJSON *json_parameters,
    MallocParameters *malloc_config) {
  cJSON *entry = NULL;
  int host = 0;
  static const char* const valid_keys[] = {
    "size",
    "host",
    "comment",
  };
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  entry = cJSON_GetObjectItem(json_parameters, "host");
  if (entry) {
    host = GetCJSONBoolean(entry);
  }
  if (host < 0) {
    printf("Invalid host setting for malloc action.\n");
    return 0;
  }
  malloc_config->allocate_host_memory = host;
  entry = cJSON_GetObjectItem(json_parameters, "size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid size setting for malloc action.\n");
    return 0;
  }
  malloc_config->size = (uint64_t) entry->valuedouble;
  return 1;
}

// Parses the given (optional) parameters for the cudaFree action. Returns 0
// on error. Since the parameters are optional, json_parameters can be NULL.
static int ParseFreeParameters(cJSON *json_parameters,
    FreeParameters *free_config) {
  cJSON *entry = NULL;
  int host = 0;
  static const char* const valid_keys[] = {
    "host",
    "comment",
  };
  // The config here is optional.
  free_config->free_host_memory = 0;
  if (!json_parameters) return 1;
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  entry = cJSON_GetObjectItem(json_parameters, "host");
  if (entry) {
    host = GetCJSONBoolean(entry);
  }
  if (host < 0) {
    printf("Invalid host setting for malloc action.\n");
    return 0;
  }
  free_config->free_host_memory = host;
  return 1;
}

// Parses JSON parameters for the memset action. Returns 0 on error.
static int ParseMemsetParameters(cJSON *json_parameters,
    MemsetParameters *memset_config) {
  cJSON *entry = NULL;
  int async = 0;
  static const char* const valid_keys[] = {
    "async",
    "size",
    "comment",
  };
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  entry = cJSON_GetObjectItem(json_parameters, "size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid size for memset action.\n");
    return 0;
  }
  memset_config->size = (uint64_t) entry->valuedouble;
  entry = cJSON_GetObjectItem(json_parameters, "async");
  if (entry) {
    async = GetCJSONBoolean(entry);
  }
  if (async < 0) {
    printf("Invalid async setting for memset action.\n");
    return 0;
  }
  memset_config->synchronous = !async;
  return 1;
}

// Parses JSON parameters for the memcpy action. Returns 0 on error.
static int ParseMemcpyParameters(cJSON *json_parameters,
    MemcpyParameters *memcpy_config) {
  cJSON *entry = NULL;
  int async = 0;
  static const char* const valid_keys[] = {
    "async",
    "size",
    "direction",
    "comment",
  };
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  entry = cJSON_GetObjectItem(json_parameters, "size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid size for memcpy action.\n");
    return 0;
  }
  memcpy_config->size = (uint64_t) entry->valuedouble;
  entry = cJSON_GetObjectItem(json_parameters, "async");
  if (entry) {
    async = GetCJSONBoolean(entry);
  }
  if (async < 0) {
    printf("Invalid async setting for memcpy action.\n");
    return 0;
  }
  memcpy_config->synchronous = !async;
  entry = cJSON_GetObjectItem(json_parameters, "direction");
  if (!entry || (entry->type != cJSON_String)) {
    printf("Missing/invalid direction for memcpy action.\n");
    return 0;
  }
  if (strcmp(entry->valuestring, "deviceToHost") == 0) {
    memcpy_config->direction = cudaMemcpyDeviceToHost;
  } else if (strcmp(entry->valuestring, "hostToDevice") == 0) {
    memcpy_config->direction = cudaMemcpyHostToDevice;
  } else if (strcmp(entry->valuestring, "deviceToDevice") == 0) {
    memcpy_config->direction = cudaMemcpyDeviceToDevice;
  } else {
    printf("Unsupported direction for memcpy action: %s\n",
      entry->valuestring);
    return 0;
  }
  return 1;
}

// Parses the JSON parameters for the "synchronize" action. Returns 0 on error.
// The json_parameters can be NULL, in which case the sync parameters will take
// their default values.
static int ParseSyncParameters(cJSON *json_parameters,
    SyncParameters *sync_config) {
  cJSON *entry = NULL;
  static const char* const valid_keys[] = {
    "device",
    "comment",
  };
  sync_config->sync_device = 0;
  if (!json_parameters) return 1;
  if (!VerifyJSONKeys(json_parameters, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  entry = cJSON_GetObjectItem(json_parameters, "device");
  if (entry) {
    sync_config->sync_device = GetCJSONBoolean(entry);
  }
  if (sync_config->sync_device < 0) {
    sync_config->sync_device = 0;
    printf("Invalid device setting for sync action.\n");
    return 0;
  }
  return 1;
}

// Parses a JSON action object in order to fill in the given ActionConfig.
// Returns 0 on error and 1 on success. May partially initialize action on
// error, so the caller may need to clean it up. However, the action type is
// guaranteed to be valid if any other fields are set.
static int ParseSingleAction(cJSON *object, ActionConfig *action,
    InitializationParameters *params) {
  cJSON *entry = NULL;
  ActionType type = ACTION_UNINITIALIZED;
  static const char* const valid_keys[] = {
    "type",
    "label",
    "delay",
    "parameters",
    "comment",
  };
  // Validate keys to find confusing spelling mistakes that may make a setting
  // take its default value unintentionally.
  if (!VerifyJSONKeys(object, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    return 0;
  }
  // Start with the hardest property: the action's type.
  entry = cJSON_GetObjectItem(object, "type");
  if (!entry || (entry->type != cJSON_String)) {
    printf("Missing/invalid action type for stream_action.so.\n");
    return 0;
  }
  if (strcmp(entry->valuestring, "launchKernel") == 0) {
    type = ACTION_KERNEL;
  } else if (strcmp(entry->valuestring, "cudaMalloc") == 0) {
    type = ACTION_MALLOC;
  } else if (strcmp(entry->valuestring, "cudaFree") == 0) {
    type = ACTION_FREE;
  } else if (strcmp(entry->valuestring, "cudaMemset") == 0) {
    type = ACTION_MEMSET;
  } else if (strcmp(entry->valuestring, "cudaMemcpy") == 0) {
    type = ACTION_MEMCPY;
  } else if (strcmp(entry->valuestring, "synchronize") == 0) {
    type = ACTION_SYNC;
  } else {
    printf("Unsupported action type for stream_action.so: %s\n",
      entry->valuestring);
    return 0;
  }
  action->type = type;
  entry = cJSON_GetObjectItem(object, "label");
  if (!entry || (entry->type != cJSON_String)) {
    printf("Missing/invalid action label for stream_action.so.\n");
    return 0;
  }
  action->label = strdup(entry->valuestring);
  if (!action->label) return 0;
  // Last, parse the action-specific parameters. Remember that additional
  // parameters are optional for some actions, so only ensure that the
  // parameters are an object if they're non-NULL.
  entry = cJSON_GetObjectItem(object, "parameters");
  if (entry && (entry->type != cJSON_Object)) {
    printf("Invalid action parameters for stream_action.so.\n");
    return 0;
  }
  // Get kernel config parsing over with first, since it's the most complex.
  if (type == ACTION_KERNEL) {
    if (!entry) {
      printf("Missing kernel parameters for stream_action.so.\n");
      return 0;
    }
    if (!ParseKernelParameters(entry, &(action->parameters.kernel),
      params->block_count, params->thread_count)) {
      return 0;
    }
  }
  if (type == ACTION_MALLOC) {
    if (!entry) {
      printf("Missing malloc parameters for stream_action.so.\n");
      return 0;
    }
    if (!ParseMallocParameters(entry, &(action->parameters.malloc))) return 0;
  }
  if (type == ACTION_FREE) {
    // It's okay for "entry" to be NULL here.
    if (!ParseFreeParameters(entry, &(action->parameters.free))) return 0;
  }
  if (type == ACTION_MEMSET) {
    if (!entry) {
      printf("Missing memset parameters for stream_action.so.\n");
      return 0;
    }
    if (!ParseMemsetParameters(entry, &(action->parameters.memset))) return 0;
  }
  if (type == ACTION_MEMCPY) {
    if (!entry) {
      printf("Missing memcpy parameters for stream_action.so.\n");
      return 0;
    }
    if (!ParseMemcpyParameters(entry, &(action->parameters.memcpy))) return 0;
  }
  if (type == ACTION_SYNC) {
    // It's okay for "entry" to be NULL here, too.
    if (!ParseSyncParameters(entry, &(action->parameters.sync))) return 0;
  }
  return 1;
}

// Takes a TaskState struct to be initialized and a JSON configuration string.
// Parses the JSON configuration and fills the appropriate fields in the state
// struct. The stream_priority value is needed, because this function will
// create the CUDA stream if the use_null_stream setting is not true. Returns 0
// on error.
static int ParseParameters(TaskState *state,
    InitializationParameters *params) {
  cJSON *json_root = NULL;
  cJSON *list_head = NULL;
  cJSON *entry = NULL;
  ActionConfig *actions = NULL;
  ActionConfig *action = NULL;
  int i = 0, action_count = 0, use_null_stream = 0;
  static const char* const valid_keys[] = {
    "actions",
    "use_null_stream",
    "comment",
  };
  json_root = cJSON_Parse(params->additional_info);
  if (!json_root || (json_root->type != cJSON_Object)) {
    printf("Missing/invalid additional_info for stream_action.so.\n");
    goto ErrorCleanup;
  }
  if (!VerifyJSONKeys(json_root, valid_keys, sizeof(valid_keys) /
    sizeof(char*))) {
    goto ErrorCleanup;
  }
  // First, check for the "use_null_stream" setting.
  entry = cJSON_GetObjectItem(json_root, "use_null_stream");
  if (entry) use_null_stream = GetCJSONBoolean(entry);
  if (use_null_stream < 0) {
    printf("Invalid use_null_stream setting in stream_action.so.\n");
    goto ErrorCleanup;
  }
  // Always use a user-defined stream for copy_out operations.
  if (!CheckCUDAError(CreateCUDAStreamWithPriority(params->stream_priority,
    &(state->copy_out_stream)))) {
    goto ErrorCleanup;
  }
  state->stream_created = 1;
  // If the NULL stream wasn't speicified, then use the user-defined stream
  // for all other operations, too.
  if (use_null_stream) {
    state->stream = 0;
  } else {
    state->stream = state->copy_out_stream;
  }
  // Get the actions list, ensuring it's an array with at least one element.
  list_head = cJSON_GetObjectItem(json_root, "actions");
  if (!list_head || (list_head->type != cJSON_Array) || !list_head->child) {
    printf("Missing/invalid list of actions for stream_action.so.\n");
    goto ErrorCleanup;
  }
  // Count the number of actions in the list.
  entry = list_head->child;
  action_count = 1;
  while (entry->next) {
    action_count++;
    entry = entry->next;
  }
  // Allocate and initialize the internal list of ActionConfig structs.
  actions = (ActionConfig *) calloc(action_count, sizeof(*actions));
  if (!actions) goto ErrorCleanup;
  entry = list_head->child;
  for (i = 0; i < action_count; i++) {
    action = actions + i;
    if (!ParseSingleAction(entry, action, params)) goto ErrorCleanup;
    entry = entry->next;
  }
  // Clean up and return success.
  state->actions = actions;
  state->action_count = action_count;
  cJSON_Delete(json_root);
  return 1;
ErrorCleanup:
  if (json_root) cJSON_Delete(json_root);
  if (actions) {
    for (i = 0; i < action_count; i++) {
      CleanupAction(actions + i);
    }
    free(actions);
  }
  return 0;
}

// Allocates buffers needed by a single kernel action. Returns 0 on error.
static int AllocateKernelActionMemory(KernelParameters *parameters) {
  size_t block_times_size = 2 * parameters->block_count * sizeof(uint64_t);
  size_t smids_size = parameters->block_count * sizeof(uint32_t);
  if (!CheckCUDAError(cudaMalloc(&parameters->device_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&parameters->device_smids, smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&parameters->host_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&parameters->host_smids, smids_size))) {
    return 0;
  }
  return 1;
}

// Takes a TaskState after fully parsing InitializationParameters (i.e. the
// actions list is populated). Allocates necessary buffers for kernel actions,
// memory sets and copies, holding pointers for malloc actions, and buffers of
// data to report to the calling process during copy_out. Returns 0 on error.
static int AllocateMemory(TaskState *state) {
  int i;
  uint64_t current_size;
  uint64_t max_size = 0;
  int secondary_buffer_needed = 0;
  int malloc_action_exists = 0;
  int kernel_count = 0;
  ActionConfig *action = NULL;
  // Collect aggregate information about all actions, and allocate the kernel
  // action's buffers while we're at it.
  for (i = 0; i < state->action_count; i++) {
    action = state->actions + i;
    switch (action->type) {
      case ACTION_KERNEL:
        kernel_count++;
        if (!AllocateKernelActionMemory(&(action->parameters.kernel))) {
          return 0;
        }
        break;
      case ACTION_MALLOC:
        malloc_action_exists = 1;
        break;
      case ACTION_MEMSET:
        current_size = action->parameters.memset.size;
        if (current_size > max_size) max_size = current_size;
        break;
      case ACTION_MEMCPY:
        current_size = action->parameters.memcpy.size;
        if (current_size > max_size) max_size = current_size;
        if (action->parameters.memcpy.direction == cudaMemcpyDeviceToDevice) {
          secondary_buffer_needed = 1;
        }
        break;
      default:
        break;
    }
  }
  // Start by allocating device memory.
  if (!CheckCUDAError(cudaMalloc(&state->device_copy_buffer, max_size))) {
    return 0;
  }
  // Only allocate a second device buffer if a device-to-device memcpy action
  // is present.
  if (secondary_buffer_needed) {
    if (!CheckCUDAError(cudaMalloc(&state->device_secondary_buffer,
      max_size))) {
      return 0;
    }
  }
  // Now allocate host memory.
  if (!CheckCUDAError(cudaMallocHost(&state->host_copy_buffer, max_size))) {
    return 0;
  }
  if (malloc_action_exists) {
    state->device_memory_allocations = (uint8_t**) calloc(
      MAX_MEMORY_ALLOCATION_COUNT, sizeof(uint8_t*));
    if (!state->device_memory_allocations) {
      printf("Failed allocating list of device memory allocation pointers.\n");
      return 0;
    }
    state->host_memory_allocations = (uint8_t**) calloc(
      MAX_MEMORY_ALLOCATION_COUNT, sizeof(uint8_t*));
    if (state->host_memory_allocations) {
      printf("Failed allocating list of host memory allocation pointers.\n");
      return 0;
    }
  }
  // Any pointers contained in the individual KernelTimes entries are simply
  // copied from KernelParameters structs after execution--they don't need to
  // be allocated here.
  state->kernel_times = (KernelTimes*) calloc(kernel_count,
    sizeof(KernelTimes));
  if (!state->kernel_times) {
    printf("Failed allocating list of kernel times.\n");
    return 0;
  }
  state->kernel_count = kernel_count;
  return 1;
}

// Initializes the tasks' kernel_times array. Must be called after memory
// allocation. This is done once because most of the fields in the kernel_times
// array never change, apart from cuda_launch_times. Returns 0 on error.
static int InitializeKernelTimes(TaskState *state) {
  int i;
  int kernel_index = 0;
  KernelTimes *current_times = NULL;
  ActionConfig *action = NULL;
  KernelParameters *params = NULL;
  for (i = 0; i < state->kernel_count; i++) {
    action = state->actions + i;
    if (action->type != ACTION_KERNEL) continue;
    params = &(action->parameters.kernel);
    current_times = state->kernel_times + kernel_index;
    current_times->kernel_name = action->label;
    current_times->block_count = params->block_count;
    current_times->thread_count = params->thread_count;
    current_times->shared_memory = params->shared_memory_count * 4;
    current_times->block_times = params->host_block_times;
    current_times->block_smids = params->host_smids;
  }
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
  state = (TaskState *) malloc(sizeof(*state));
  if (!state) {
    printf("Error allocating memory for stream_action task state.\n");
    return NULL;
  }
  memset(state, 0, sizeof(*state));
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) {
    Cleanup(state);
    return NULL;
  }
  // Parse the configuration string, initialize the action configs, and create
  // the CUDA stream (if a non-NULL stream is used).
  if (!ParseParameters(state, params)) {
    Cleanup(state);
    return NULL;
  }
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  if (!InitializeKernelTimes(state)) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// Nothing needs to be copied to the GPU at this stage in the benchmark.
static int CopyIn(void *data) {
  return 1;
}

// Copies device data to host buffers for a single kernel action. Requires a
// stream on which the copy should run. Returns 0 on error.
static int CopyKernelActionMemoryOut(KernelParameters *kernel,
    cudaStream_t stream) {
  size_t block_times_size = 2 * kernel->block_count * sizeof(uint64_t);
  size_t block_smids_size = 2 * kernel->block_count * sizeof(uint64_t);
  if (!CheckCUDAError(cudaMemcpyAsync(kernel->host_block_times,
    kernel->device_block_times, block_times_size, cudaMemcpyDeviceToHost,
    stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(kernel->host_smids, kernel->device_smids,
    block_smids_size, cudaMemcpyDeviceToHost, stream))) {
    return 0;
  }
  return 1;
}

// Provides the caller with information about the kernel actions.
static int CopyOut(void *data, TimingInformation *times) {
  TaskState *state = (TaskState *) data;
  int i;
  for (i = 0; i < state->action_count; i++) {
    if (state->actions[i].type != ACTION_KERNEL) continue;
    if (!CopyKernelActionMemoryOut(&(state->actions[i].parameters.kernel),
      state->copy_out_stream)) {
      return 0;
    }
  }
  // The kernel_times structs were already filled in with the correct pointers
  // during initialization, and the cuda_launch_times were filled in during the
  // execute phase. So now, all that needs to be done is provide the correct
  // pointer.
  times->kernel_count = state->kernel_count;
  times->kernel_info = state->kernel_times;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

// Executes a kernel action. Requires the index of the kernel action (its order
// relative only to other kernel actions) in order to fill in the CUDA launch
// times in the correct entry in the kernel_times array. Returns 0 on error.
static int ExecuteKernelAction(TaskState *state, KernelParameters *params,
    int kernel_index) {
  state->kernel_times[kernel_index].cuda_launch_times[0] = CurrentSeconds();
  switch (params->shared_memory_count) {
    case 0:
      GPUSpin<<<params->block_count, params->thread_count, 0, state->stream>>>(
        params->use_counter_spin, params->duration, params->device_block_times,
        params->device_smids, NULL);
      break;
    case 4096:
      SharedMemGPUSpin_4096<<<params->block_count, params->thread_count, 0,
        state->stream>>>(params->use_counter_spin, params->duration,
        params->device_block_times, params->device_smids, NULL);
      break;
    case 8192:
      SharedMemGPUSpin_8192<<<params->block_count, params->thread_count, 0,
        state->stream>>>(params->use_counter_spin, params->duration,
        params->device_block_times, params->device_smids, NULL);
      break;
    case 10240:
      SharedMemGPUSpin_10240<<<params->block_count, params->thread_count, 0,
        state->stream>>>(params->use_counter_spin, params->duration,
        params->device_block_times, params->device_smids, NULL);
      break;
    default:
      printf("Unsupported kernel shared memory count: %d\n",
        params->shared_memory_count);
      return 0;
  }
  // Record the time after the kernel launch returns, but we don't know when
  // synchronization will complete in this benchmark, so set that entry to 0.
  state->kernel_times[kernel_index].cuda_launch_times[1] = CurrentSeconds();
  state->kernel_times[kernel_index].cuda_launch_times[2] = 0;
  return 1;
}

// Executes a malloc action. Returns 0 on error.
static int ExecuteMallocAction(TaskState *state, MallocParameters *params) {
  int next_index = 0;
  uint8_t **destination = NULL;
  if (params->allocate_host_memory) {
    next_index = state->host_memory_allocation_count;
  } else {
    next_index = state->device_memory_allocation_count;
  }
  if (next_index >= MAX_MEMORY_ALLOCATION_COUNT) {
    printf("Can't execute malloc action: too many unfreed %s allocations.\n",
      params->allocate_host_memory ? "host" : "device");
    return 0;
  }
  if (params->allocate_host_memory) {
    destination = &(state->host_memory_allocations[next_index]);
    if (!CheckCUDAError(cudaMallocHost(destination, params->size))) return 0;
    state->host_memory_allocation_count++;
    return 1;
  }
  destination = &(state->device_memory_allocations[next_index]);
  if (!CheckCUDAError(cudaMalloc(destination, params->size))) return 0;
  state->device_memory_allocation_count++;
  return 1;
}

// Executes a free action. Returns 0 on error.
static int ExecuteFreeAction(TaskState *state, FreeParameters *params) {
  if (params->free_host_memory) {
    if (state->host_memory_allocation_count == 0) {
      printf("Can't execute free action: No host memory allocations.\n");
      return 0;
    }
    state->host_memory_allocation_count--;
    if (!CheckCUDAError(cudaFreeHost(state->host_memory_allocations[
      state->host_memory_allocation_count]))) {
      return 0;
    }
    return 1;
  }
  if (state->device_memory_allocation_count == 0) {
    printf("Can't execute free action: No device memory allocations.\n");
    return 0;
  }
  state->device_memory_allocation_count--;
  if (!CheckCUDAError(cudaFree(state->device_memory_allocations[
    state->device_memory_allocation_count]))) {
    return 0;
  }
  return 1;
}

// Executes a memset action. Fills a device buffer with a random value. Returns
// 0 on error.
static int ExecuteMemsetAction(TaskState *state, MemsetParameters *params) {
  if (params->synchronous) {
    if (!CheckCUDAError(cudaMemset(state->device_copy_buffer, rand(),
      params->size))) {
      return 0;
    }
    return 1;
  }
  if (!CheckCUDAError(cudaMemsetAsync(state->device_copy_buffer, rand(),
    params->size, state->stream))) {
    return 0;
  }
  return 1;
}

// Executes a memcpy action. Returns 0 on error.
static int ExecuteMemcpyAction(TaskState *state, MemcpyParameters *params) {
  uint8_t *src = NULL;
  uint8_t *dest = NULL;
  switch (params->direction) {
    case cudaMemcpyDeviceToDevice:
      src = state->device_copy_buffer;
      dest = state->device_secondary_buffer;
      break;
    case cudaMemcpyDeviceToHost:
      src = state->device_copy_buffer;
      dest = state->host_copy_buffer;
      break;
    case cudaMemcpyHostToDevice:
      src = state->host_copy_buffer;
      dest = state->device_copy_buffer;
      break;
    default:
      printf("Unsupported direction for memcpy action: %d\n",
        (int) params->direction);
      return 0;
  }
  if (params->synchronous) {
    if (!CheckCUDAError(cudaMemcpy(dest, src, params->size,
      params->direction))) {
      return 0;
    }
    return 1;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(dest, src, params->size,
    params->direction, state->stream))) {
    return 0;
  }
  return 1;
}

// Executes a synchronization action. Returns 0 on error.
static int ExecuteSyncAction(TaskState *state, SyncParameters *params) {
  if (params->sync_device) {
    if (!CheckCUDAError(cudaDeviceSynchronize())) return 0;
    return 1;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

// Sleeps for at least the given number of seconds, with a microsecond
// granularity.
static void SleepSeconds(double seconds) {
  uint64_t to_sleep = (uint64_t) (seconds * 1e6);
  usleep(to_sleep);
}

// Executes each action in the order it appears in the list.
static int Execute(void *data) {
  TaskState *state = (TaskState *) data;
  ActionConfig *action = NULL;
  int kernel_index = 0;
  int i;
  for (i = 0; i < state->action_count; i++) {
    action = state->actions + i;
    if (action->delay > 0.0) {
      // The delay is only applied after all of the previous stream work has
      // completed.
      if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
      SleepSeconds(state->actions[i].delay);
    }
    switch (action->type) {
      case ACTION_KERNEL:
        if (!ExecuteKernelAction(state, &(action->parameters.kernel),
          kernel_index)) {
          return 0;
        }
        kernel_index++;
        break;
      case ACTION_MALLOC:
        if (!ExecuteMallocAction(state, &(action->parameters.malloc))) {
          return 0;
        }
        break;
      case ACTION_FREE:
        if (!ExecuteFreeAction(state, &(action->parameters.free))) {
          return 0;
        }
        break;
      case ACTION_MEMSET:
        if (!ExecuteMemsetAction(state, &(action->parameters.memset))) {
          return 0;
        }
        break;
      case ACTION_MEMCPY:
        if (!ExecuteMemcpyAction(state, &(action->parameters.memcpy))) {
          return 0;
        }
        break;
      case ACTION_SYNC:
        if (!ExecuteSyncAction(state, &(action->parameters.sync))) {
          return 0;
        }
        break;
      default:
        printf("Attempted to execute invalid action: %d\n", action->type);
        return 0;
    }
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Sequential action execution";
}

int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  functions->cleanup = Cleanup;
  functions->get_name = GetName;
  return 1;
}
