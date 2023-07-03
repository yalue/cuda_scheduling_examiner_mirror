// This file defines a benchmark similar to timer_spin.cu, but with a
// somewhat-configurable way to specify shared memory usage: the
// additional_info now has the following format:
//
// additional_info: {
//   "duration": <ns to spin>,
//   "shared_memory_size": <# of shared 32-bit integers>
// }
//
// The shared_memory_size must be one of 4096, 8192, or 10240. The shared
// memory usage in bytes will therefore be one of those values multiplied by 4.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"
#include "third_party/cJSON.h"

// Holds the local state for one instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // This will be set to 0 if the CUDA stream hasn't been created yet. This is
  // useful because it allows us to unconditionally call Cleanup on error
  // without needing to worry about calling cudaStreamDestroy twice.
  int stream_created;
  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids;
  // The number of nanoseconds for which each CUDA thread should spin.
  uint64_t spin_duration;
  // Holds the grid dimension to use, set during initialization.
  int block_count;
  int thread_count;
  int sharedmem_count;
  // Holds host-side times that are shared with the calling process.
  KernelTimes spin_kernel_times;
} TaskState;

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->spin_kernel_times;
  // Free device memory.
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->device_block_smids) cudaFree(state->device_block_smids);
  // Free host memory.
  if (host_times->block_times) cudaFreeHost(host_times->block_times);
  if (host_times->block_smids) cudaFreeHost(host_times->block_smids);
  if (state->stream_created) {
    // Call CheckCUDAError here to print a message, even though we won't check
    // the return value.
    CheckCUDAError(cudaStreamDestroy(state->stream));
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Allocates GPU and CPU memory. Returns 0 on error, 1 otherwise.
static int AllocateMemory(TaskState *state) {
  uint64_t block_times_size = state->block_count * sizeof(uint64_t) * 2;
  uint64_t block_smids_size = state->block_count * sizeof(uint32_t);
  KernelTimes *host_times = &state->spin_kernel_times;
  // Allocate device memory.
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times),
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids),
    block_smids_size))) {
    return 0;
  }
  // Allocate host memory.
  if (!CheckCUDAError(cudaMallocHost(&host_times->block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&host_times->block_smids,
    block_smids_size))) {
    return 0;
  }
  return 1;
}

// Parses the additional_info JSON object, which should contain two integer
// keys: "duration", the spin duration and "shared_memory_size", the shared
// memory size. Returns nonzero on success. The shared memory size must be one
// of [4096, 8192, 10240].
static int InitializeKernelConfig(TaskState *state, char *info) {
  cJSON *parsed = NULL;
  cJSON *entry = NULL;
  parsed = cJSON_Parse(info);
  if (!parsed) {
    // This should actually never happen, because it will have already been
    // successfully parsed once when the top-level file was parsed.
    printf("Failed parsing additional_info JSON\n");
    return 0;
  }
  entry = cJSON_GetObjectItem(parsed, "duration");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid duration for sharedmem_timer_spin.\n");
    cJSON_Delete(parsed);
    return 0;
  }
  // Once again, use valuedouble for better precision.
  state->spin_duration = entry->valuedouble;
  entry = cJSON_GetObjectItem(parsed, "shared_memory_size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid shared_memory_size for sharedmem_timer_spin.\n");
    cJSON_Delete(parsed);
    return 0;
  }
  state->sharedmem_count = entry->valueint;
  // Free the parsed JSON now that we've obtained what we need.
  entry = NULL;
  cJSON_Delete(parsed);
  parsed = NULL;
  switch (state->sharedmem_count) {
    case 4096:
    case 8192:
    case 10240:
      return 1;
  }
  printf("Unsupported shared memory size: %d\n", (int) state->sharedmem_count);
  state->spin_duration = 0;
  state->sharedmem_count = 0;
  return 0;
}

static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
  // First allocate space for local data.
  state = (TaskState *) calloc(1, sizeof(*state));
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  if (!GetSingleBlockAndGridDimensions(params, &state->thread_count,
    &state->block_count)) {
    Cleanup(state);
    return NULL;
  }
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  // Parse the additional_info JSON (spin duration, shared memory count)
  if (!InitializeKernelConfig(state, params->additional_info)) return NULL;
  if (!CheckCUDAError(CreateCUDAStreamWithPriorityAndMask(
    params->stream_priority, params->sm_mask, &(state->stream)))) {
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

static int Execute(void *data) {
  TaskState *state = (TaskState *) data;
  state->spin_kernel_times.cuda_launch_times[0] = CurrentSeconds();
  if (state->sharedmem_count == 4096) {
    SharedMem_GPUSpin4096<<<state->block_count, state->thread_count, 0,
      state->stream>>>(state->spin_duration, state->device_block_times,
      state->device_block_smids);
  } else if (state->sharedmem_count == 8192) {
    SharedMem_GPUSpin8192<<<state->block_count, state->thread_count, 0,
      state->stream>>>(state->spin_duration, state->device_block_times,
      state->device_block_smids);
  } else if (state->sharedmem_count == 10240) {
    SharedMem_GPUSpin10240<<<state->block_count, state->thread_count, 0,
      state->stream>>>(state->spin_duration, state->device_block_times,
      state->device_block_smids);
  }
  state->spin_kernel_times.cuda_launch_times[1] = CurrentSeconds();
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  state->spin_kernel_times.cuda_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->spin_kernel_times;
  uint64_t block_times_count = state->block_count * 2;
  uint64_t block_smids_count = state->block_count;
  memset(times, 0, sizeof(*times));
  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_times,
    state->device_block_times, block_times_count * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_smids,
    state->device_block_smids, block_smids_count * sizeof(uint32_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  host_times->kernel_name = "SharedMem_GPUSpin";
  host_times->block_count = state->block_count;
  host_times->thread_count = state->thread_count;
  host_times->shared_memory = state->sharedmem_count * sizeof(uint32_t);
  times->kernel_count = 1;
  times->kernel_info = host_times;
  return 1;
}

static const char* GetName(void) {
  return "Timer Spin (shared memory)";
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
