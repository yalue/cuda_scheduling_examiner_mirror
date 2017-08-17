// This file defines a CUDA benchmark which issues multiple kernels to a single
// stream before waiting for all kernels to complete. The configuration for the
// kernels is taken exclusively from the additional_info field in the
// InitializationParameters struct. The actual kernels will simply be instances
// of the same kernel as in the timer_spin benchmark. This benchmark ignores
// all fields in its initialization parameters apart from cuda_device and
// additional_info.
//
// The format of the necessary additional_info field is as follows:
// "additional_info": [
//   {
//     "kernel_label": "<Label string to give this kernel launch>",
//     "duration": <number of nanoseconds to run this kernel>,
//     "block_count": <number of blocks to launch for this kernel>,
//     "thread_count": <number of threads per block for this kernel>,
//     "delay": <number of seconds to sleep after the previous kernel, but
//              before releasing this kernel. Defaults to 0.0.>,
//     "shared_memory_size": <# of shared 32-bit integers. Must be one of
//                           0, 4096, 8192, or 10240. The shared memory
//                           usage in bytes will be one of those values
//                           multiplied by 4. Defaults to 0.>,
//     "copy_in_count": <# of 32-bit integers to copy from the host to the
//                      device before invoking the kernel. The amount of data
//                      copied in bytes will be this value multiplied by 4.
//                      Defaults to 0.>,
//     "copy_out_count": <# of 32-bit integers to copy from the device to the
//                       host after invoking the kernel. The amount of data
//                       copied in bytes will be this value multiplied by 4.
//                       Defaults to 0.>
//   },
//   {... <kernel 2 info> ...},
//   ...
// ]
//
// Kernels are issued to the stream in the same order that they're specified
// in the additional_info list.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"
#include "third_party/cJSON.h"

// Holds the parameters for a single kernel's execution.
typedef struct {
  // The name given to this kernel.
  char *name;
  // The number of nanoseconds this kernel should spin for.
  uint64_t spin_duration;
  // The grid dimensions for this kernel
  int block_count;
  int thread_count;
  // The amount of shared memory used by the kernel.
  int shared_memory_count;
  // The number of 32-bit integers to copy to the device before executing
  // this kernel.
  int copy_in_count;
  // The number of 32-bit integers to copy from the device after executing
  // this kernel. This should be <= copy_in_count.
  int copy_out_count;
  // The number of seconds to sleep after the previous kernel's completion,
  // before executing this kernel.
  double delay;
  // The host and device memory buffers for the kernel.
  double cuda_launch_times[3];
  uint64_t *host_block_times;
  uint64_t *device_block_times;
  uint32_t *host_smids;
  uint32_t *device_smids;
  uint32_t *host_copy_data;
  uint32_t *device_copy_data;
} SingleKernelData;

// Holds the local state for one instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // This will be set to 0 if the CUDA stream hasn't been created yet. This is
  // useful because it allows us to unconditionally call Cleanup on error
  // without needing to worry about calling cudaStreamDestroy twice.
  int stream_created;
  // The number of kernels that will be created.
  int kernel_count;
  // Holds the parameters used to invoke each kernel.
  SingleKernelData *kernel_configs;
  // Holds the resulting times for each kernel that are passed to the calling
  // process.
  KernelTimes *kernel_times;
} BenchmarkState;

// Frees memory in a single kernel config struct.
static void CleanupKernelConfig(SingleKernelData *kernel_config) {
  uint64_t *tmp64 = NULL;
  uint32_t *tmp32 = NULL;
  // Free host memory for this kernel.
  tmp64 = kernel_config->host_block_times;
  if (tmp64) cudaFreeHost(tmp64);
  tmp32 = kernel_config->host_smids;
  if (tmp32) cudaFreeHost(tmp32);
  tmp32 = kernel_config->host_copy_data;
  if (tmp32) cudaFreeHost(tmp32);
  if (kernel_config->name) free(kernel_config->name);
  // Free GPU memory for this kernel.
  tmp64 = kernel_config->device_block_times;
  if (tmp64) cudaFree(tmp64);
  tmp32 = kernel_config->device_smids;
  if (tmp32) cudaFree(tmp32);
  tmp32 = kernel_config->device_copy_data;
  if (tmp32) cudaFree(tmp32);
  memset(kernel_config, 0, sizeof(*kernel_config));
}

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  SingleKernelData *kernel_config = NULL;
  int i;
  for (i = 0; i < state->kernel_count; i++) {
    kernel_config = state->kernel_configs + i;
    CleanupKernelConfig(kernel_config);
  }
  if (state->stream_created) {
    // Call CheckCUDAError here to print a message, even though we won't check
    // the return value.
    CheckCUDAError(cudaStreamDestroy(state->stream));
  }
  if (state->kernel_configs) free(state->kernel_configs);
  if (state->kernel_times) free(state->kernel_times);
  memset(state, 0, sizeof(*state));
  free(state);
}

// Allocates host and device memory for a single kernel config struct. Returns
// 0 on error and nonzero on success. Must be called after block size has been
// set.
static int AllocateKernelDataMemory(SingleKernelData *config) {
  size_t block_times_size = 2 * config->block_count * sizeof(uint64_t);
  size_t smids_size = config->block_count * sizeof(uint32_t);
  uint64_t copy_data_size = config->copy_in_count > config->copy_out_count ?
                            config->copy_in_count * sizeof(uint32_t) :
                            config->copy_out_count * sizeof(uint32_t);
  // Allocate host memory
  if (!CheckCUDAError(cudaMallocHost(&config->host_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&config->host_smids, smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&config->host_copy_data,
    copy_data_size))) {
    return 0;
  }
  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&config->device_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&config->device_smids, smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(config->device_copy_data),
    copy_data_size))) {
    return 0;
  }
  return 1;
}

// Copies out device memory for the given config, using the given stream. This
// will not call cudaStreamSynchronize. Returns 0 on error.
static int CopyKernelMemoryOut(SingleKernelData *config,
    cudaStream_t stream) {
  size_t block_times_size = 2 * config->block_count * sizeof(uint64_t);
  size_t block_smids_size = config->block_count * sizeof(uint32_t);
  if (!CheckCUDAError(cudaMemcpyAsync(config->host_block_times,
    config->device_block_times, block_times_size, cudaMemcpyDeviceToHost,
    stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(config->host_smids, config->device_smids,
    block_smids_size, cudaMemcpyDeviceToHost, stream))) {
    return 0;
  }
  return 1;
}

// Parses the additional info setting and allocates/initializes the
// kernel_configs array in the state struct. Returns 0 on error.
static int InitializeKernelConfigs(BenchmarkState *state, char *info) {
  cJSON *parsed = NULL;
  cJSON *list_entry = NULL;
  cJSON *entry = NULL;
  SingleKernelData *kernel_configs = NULL;
  int i = 0;
  int kernel_count = 0;
  parsed = cJSON_Parse(info);
  if (!parsed || (parsed->type != cJSON_Array) || !parsed->child) {
    printf("Missing/invalid list of kernels for multikernel.so.\n");
    goto ErrorCleanup;
  }
  // First, calculate the number of kernels so we can allocate the
  // kernel_configs array.
  list_entry = parsed->child;
  kernel_count = 1;
  while (list_entry->next) {
    kernel_count++;
    list_entry = list_entry->next;
  }
  kernel_configs = (SingleKernelData *) calloc(kernel_count,
    sizeof(*kernel_configs));
  if (!kernel_configs) goto ErrorCleanup;

  // Now, loop over each JSON object and read the kernel config data.
  list_entry = parsed->child;
  for (i = 0; i < kernel_count; i++) {
    entry = cJSON_GetObjectItem(list_entry, "kernel_label");
    if (!entry || (entry->type != cJSON_String)) {
      printf("Missing/invalid kernel label for multikernel.so.\n");
      goto ErrorCleanup;
    }
    kernel_configs[i].name = strdup(entry->valuestring);
    if (!kernel_configs[i].name) goto ErrorCleanup;
    entry = cJSON_GetObjectItem(list_entry, "duration");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid kernel duration for multikernel.so.\n");
      goto ErrorCleanup;
    }
    kernel_configs[i].spin_duration = entry->valuedouble;
    entry = cJSON_GetObjectItem(list_entry, "block_count");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid block count for multikernel.so.\n");
      goto ErrorCleanup;
    }
    kernel_configs[i].block_count = entry->valueint;
    entry = cJSON_GetObjectItem(list_entry, "thread_count");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid thread count for multikernel.so.\n");
      goto ErrorCleanup;
    }
    kernel_configs[i].thread_count = entry->valueint;
    entry = cJSON_GetObjectItem(list_entry, "shared_memory_size");
    if (!entry || (entry->type != cJSON_Number)) {
      kernel_configs[i].shared_memory_count = 0;
    } else {
      kernel_configs[i].shared_memory_count = entry->valueint;
    }
    entry = cJSON_GetObjectItem(list_entry, "copy_in_count");
    if (!entry || (entry->type != cJSON_Number)) {
      kernel_configs[i].copy_in_count = 0;
    } else {
      kernel_configs[i].copy_in_count = entry->valueint;
    }
    entry = cJSON_GetObjectItem(list_entry, "copy_out_count");
    if (!entry || (entry->type != cJSON_Number)) {
      kernel_configs[i].copy_out_count = 0;
    } else {
      kernel_configs[i].copy_out_count = entry->valueint;
    }
    entry = cJSON_GetObjectItem(list_entry, "delay");
    kernel_configs[i].delay = 0.0;
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid delay for multikernel.so.\n");
        goto ErrorCleanup;
      }
      kernel_configs[i].delay = entry->valuedouble;
    }
    if (!AllocateKernelDataMemory(kernel_configs + i)) goto ErrorCleanup;
    list_entry = list_entry->next;
  }
  cJSON_Delete(parsed);
  parsed = NULL;
  for (i = 0; i < kernel_count; i++) {
    switch (kernel_configs[i].shared_memory_count) {
      case 0:
      case 4096:
      case 8192:
      case 10240:
        break;
      default:
        printf("Unsupported shared memory size: %d\n",
               (int) kernel_configs[i].shared_memory_count);
        goto ErrorCleanup;
    }
  } 
  state->kernel_configs = kernel_configs;
  state->kernel_count = kernel_count;
  return 1;
ErrorCleanup:
  if (parsed) cJSON_Delete(parsed);
  if (kernel_configs) {
    for (i = 0; i < kernel_count; i++) {
      if (kernel_configs[i].name) free(kernel_configs[i].name);
    }
    free(kernel_configs);
  }
  return 0;
}

static void* Initialize(InitializationParameters *params) {
  BenchmarkState *state = NULL;
  // Allocate the local data for this benchmark and associate with a GPU.
  state = (BenchmarkState *) malloc(sizeof(*state));
  memset(state, 0, sizeof(*state));
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  // Parse the configuration string and allocate most of the memory.
  if (!InitializeKernelConfigs(state, params->additional_info)) return NULL;
  // Allocate the structures passed to the caller during CopyOut.
  state->kernel_times = (KernelTimes *) malloc(state->kernel_count *
    sizeof(*(state->kernel_times)));
  if (!state->kernel_times) {
    Cleanup(state);
    return NULL;
  }
  // Create the stream
  if (!CheckCUDAError(CreateCUDAStreamWithPriority(params->stream_priority,
    &(state->stream)))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  return state;
}

// Nothing needs to be copied in for this benchmark.
static int CopyIn(void *data) {
  return 1;
}

// Uses 4096 bytes of statically-defined shared memory.
static __device__ uint32_t UseSharedMemory4096(void) {
  __shared__ uint32_t shared_mem_arr[4096];
  uint32_t num_threads, elts_per_thread, i;
  num_threads = blockDim.x;
  elts_per_thread = 4096 / num_threads;
  for (i = 0; i < elts_per_thread; i++) {
    shared_mem_arr[threadIdx.x * elts_per_thread + i] = threadIdx.x;
  }
  return shared_mem_arr[threadIdx.x * elts_per_thread];
}

// Uses 8192 bytes of statically-defined shared memory.
static __device__ uint32_t UseSharedMemory8192(void) {
  __shared__ uint32_t shared_mem_arr[8192];
  uint32_t num_threads, elts_per_thread, i;
  num_threads = blockDim.x;
  elts_per_thread = 8192 / num_threads;
  for (i = 0; i < elts_per_thread; i++) {
    shared_mem_arr[threadIdx.x * elts_per_thread + i] = threadIdx.x;
  }
  return shared_mem_arr[threadIdx.x * elts_per_thread];
}

// Uses 10240 bytes of statically-defined shared memory.
static __device__ uint32_t UseSharedMemory10240(void) {
  __shared__ uint32_t shared_mem_arr[10240];
  uint32_t num_threads, elts_per_thread, i;
  num_threads = blockDim.x;
  elts_per_thread = 10240 / num_threads;
  for (i = 0; i < elts_per_thread; i++) {
    shared_mem_arr[threadIdx.x * elts_per_thread + i] = threadIdx.x;
  }
  return shared_mem_arr[threadIdx.x * elts_per_thread];
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void SharedMem_GPUSpin4096(uint64_t spin_duration,
    uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  // In the shared memory loop, set the value for a window of elements.
  shared_mem_res = UseSharedMemory4096();
  // The actual spin loop--most of this kernel code is for recording block and
  // kernel times.
  while ((GlobalTimer64() - start_time) < spin_duration) {
    continue;
  }
  // Record the kernel and block end times.
  if (shared_mem_res == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void SharedMem_GPUSpin8192(uint64_t spin_duration,
    uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  // In the shared memory loop, set the value for a window of elements.
  shared_mem_res = UseSharedMemory8192();
  // The actual spin loop--most of this kernel code is for recording block and
  // kernel times.
  while ((GlobalTimer64() - start_time) < spin_duration) {
    continue;
  }
  // Record the kernel and block end times.
  if (shared_mem_res == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void SharedMem_GPUSpin10240(uint64_t spin_duration,
    uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  // In the shared memory loop, set the value for a window of elements.
  shared_mem_res = UseSharedMemory10240();
  // The actual spin loop--most of this kernel code is for recording block and
  // kernel times.
  while ((GlobalTimer64() - start_time) < spin_duration) {
    continue;
  }
  // Record the kernel and block end times.
  if (shared_mem_res == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}

// Spins in a loop until at least spin_duration nanoseconds have elapsed.
static __global__ void GPUSpin(uint64_t spin_duration, uint64_t *block_times,
    uint32_t *block_smids) {
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  // The actual spin loop--most of this kernel code is for recording block and
  // kernel times.
  while ((GlobalTimer64() - start_time) < spin_duration) {
    continue;
  }
  // Record the kernel and block end times.
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}

// Sleeps for at least the given number of seconds, with a microsecond
// granularity.
static void SleepSeconds(double seconds) {
  uint64_t to_sleep = (uint64_t) (seconds * 1e6);
  usleep(to_sleep);
}

static int Execute(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  SingleKernelData *config = NULL;
  int i;
  // Submit all of the kernels before calling cudaStreamSynchronize
  for (i = 0; i < state->kernel_count; i++) {
    config = state->kernel_configs + i;
    if (config->delay > 0.0) {
      if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
      SleepSeconds(config->delay);
    }
    if (config->copy_in_count > 0) {
      if (!CheckCUDAError(cudaMemcpyAsync(config->device_copy_data,
        config->host_copy_data, config->copy_in_count * sizeof(uint32_t),
        cudaMemcpyHostToDevice, state->stream))) {
        return 0;
      }
    }
    config->cuda_launch_times[0] = CurrentSeconds();
    if (config->shared_memory_count == 0) {
      GPUSpin<<<config->block_count, config->thread_count, 0, state->stream>>>(
        config->spin_duration, config->device_block_times,
        config->device_smids);
    } else if (config->shared_memory_count == 4096) {
      SharedMem_GPUSpin4096<<<config->block_count, config->thread_count, 0,
        state->stream>>>(config->spin_duration, config->device_block_times,
        config->device_smids);
    } else if (config->shared_memory_count == 8192) {
      SharedMem_GPUSpin8192<<<config->block_count, config->thread_count, 0,
        state->stream>>>(config->spin_duration, config->device_block_times,
        config->device_smids);
    } else if (config->shared_memory_count == 10240) {
      SharedMem_GPUSpin10240<<<config->block_count, config->thread_count, 0,
        state->stream>>>(config->spin_duration, config->device_block_times,
        config->device_smids);
    }
    config->cuda_launch_times[1] = CurrentSeconds();
    if (config->copy_out_count > 0) {
      if (!CheckCUDAError(cudaMemcpyAsync(config->host_copy_data,
        config->device_copy_data, config->copy_out_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, state->stream))) {
        return 0;
      }
    }
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  config->cuda_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  BenchmarkState *state = (BenchmarkState *) data;
  SingleKernelData *config = NULL;
  KernelTimes *time_to_return = NULL;
  int i;
  // As a reminder (my naming scheme sucks, but I don't know what would be
  // better): kernel_*times* is the array of structures shared with the caller
  // and kernel_*configs* is the internal copy of the data containing device
  // pointers and host pointers. The kernel_configs array is what we use for
  // bookkeeping for memory management, but kernel_times needs copies of the
  // pointers to host memory.
  for (i = 0; i < state->kernel_count; i++) {
    // Do all the copy outs in one shot before calling synchronize.
    config = state->kernel_configs + i;
    if (!CopyKernelMemoryOut(config, state->stream)) {
      return 0;
    }
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  for (i = 0; i < state->kernel_count; i++) {
    config = state->kernel_configs + i;
    time_to_return = state->kernel_times + i;
    memcpy(time_to_return->cuda_launch_times, config->cuda_launch_times,
      sizeof(config->cuda_launch_times));
    time_to_return->block_times = config->host_block_times;
    time_to_return->block_smids = config->host_smids;
    time_to_return->block_count = config->block_count;
    time_to_return->thread_count = config->thread_count;
    time_to_return->kernel_name = config->name;
    time_to_return->shared_memory = config->shared_memory_count * sizeof(uint32_t);
  }
  times->kernel_info = state->kernel_times;
  times->kernel_count = state->kernel_count;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

static const char* GetName(void) {
  return "Multi-kernel submission";
}

// This should be the only function we export from the library, to provide
// pointers to all of the other functions.
int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  functions->cleanup = Cleanup;
  functions->get_name = GetName;
  return 1;
}
