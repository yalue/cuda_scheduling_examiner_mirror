// This file defines a CUDA benchmark in which a kernel will perform a sequence
// of memory reads over a buffer in GPU memory.
//
// By default, this benchmark performs an arbitrary constant number of memory
// references, but a specific number of references can be provided as an
// "additional_info" string.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// The default number of memory reads to perform in each iteration
#define DEFAULT_MEMORY_ACCESS_COUNT (1000 * 1000)

// Holds the local state for one instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // This will be set to 0 if the CUDA stream hasn't been created yet. This is
  // useful because it allows us to unconditionally call Cleanup on error
  // without needing to worry about calling cudaStreamDestroy twice.
  int stream_created;
  // Holds the device copy of the overall start and end time of the kernel.
  uint64_t *device_kernel_times;
  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids;
  // Holds the buffer of GPU memory traversed during the walk.
  uint32_t *device_walk_buffer;
  // Use an accumulator that the host reads in order to prevent CUDA from
  // optimizing out our loop.
  uint64_t *device_accumulator;
  uint64_t host_accumulator;
  // Holds the grid dimension to use, set during initialization.
  int block_count;
  int thread_count;
  // The number of steps to make in the walk.
  uint64_t memory_access_count;
  // The number of 32-bit elements in the walk buffer.
  uint64_t walk_buffer_length;
  // Holds host-side times that are shared with the calling process.
  KernelTimes walk_kernel_times;
} BenchmarkState;

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  KernelTimes *host_times = &state->walk_kernel_times;
  // Free device memory.
  if (state->device_kernel_times) cudaFree(state->device_kernel_times);
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->device_block_smids) cudaFree(state->device_block_smids);
  if (state->device_walk_buffer) cudaFree(state->device_walk_buffer);
  if (state->device_accumulator) cudaFree(state->device_accumulator);
  // Free host memory.
  if (host_times->kernel_times) cudaFreeHost(host_times->kernel_times);
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
static int AllocateMemory(BenchmarkState *state) {
  uint64_t block_times_size = state->block_count * sizeof(uint64_t) * 2;
  uint64_t block_smids_size = state->block_count * sizeof(uint32_t);
  KernelTimes *host_times = &state->walk_kernel_times;
  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&(state->device_kernel_times),
    2 * sizeof(uint64_t)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times),
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids),
    block_smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_walk_buffer),
    state->walk_buffer_length * sizeof(uint32_t)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_accumulator),
    sizeof(uint64_t)))) {
    return 0;
  }
  // Allocate host memory.
  if (!CheckCUDAError(cudaMallocHost(&host_times->kernel_times, 2 *
    sizeof(uint64_t)))) {
    return 0;
  }
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

// If the given argument is a non-NULL, non-empty string, attempts to set
// memory_access_count by parsing it. Otherwise, this function will set the
// count to a default value. Returns 0 on error.
static int SetMemoryAccessCount(const char *arg, BenchmarkState *state) {
  int64_t parsed_value;
  if (!arg || (strlen(arg) == 0)) {
    state->memory_access_count = DEFAULT_MEMORY_ACCESS_COUNT;
    return 1;
  }
  char *end = NULL;
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid memory access count: %s\n", arg);
    return 0;
  }
  state->memory_access_count = (uint64_t) parsed_value;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  BenchmarkState *state = NULL;
  uint64_t i;
  uint32_t *host_initial_buffer = NULL;
  // First allocate space for local data.
  state = (BenchmarkState *) malloc(sizeof(*state));
  if (!state) return NULL;
  memset(state, 0, sizeof(*state));
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  // Round the thread count up to a value evenly divisible by 32.
  if ((params->thread_count % WARP_SIZE) != 0) {
    params->thread_count += WARP_SIZE - (params->thread_count % WARP_SIZE);
  }
  state->thread_count = params->thread_count;
  state->block_count = params->block_count;
  state->walk_buffer_length = params->data_size / 4;
  if (state->walk_buffer_length <= 0) {
    printf("Memory walks require a data_size of at least 4.\n");
    Cleanup(state);
    return NULL;
  }
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  // Now that the device buffer is allocated, initialize it for an in-order
  // walk.
  host_initial_buffer = (uint32_t *) malloc(state->walk_buffer_length *
    sizeof(uint32_t));
  if (!host_initial_buffer) {
    printf("Failed allocating host buffer for initializing GPU memory.\n");
    Cleanup(state);
    return NULL;
  }
  // Initialize a cyclic in-order walk using the indices of the array.
  for (i = 0; i < (state->walk_buffer_length - 1); i++) {
    host_initial_buffer[i] = i + 1;
  }
  host_initial_buffer[state->walk_buffer_length - 1] = 0;
  if (!CheckCUDAError(cudaMemcpy(state->device_walk_buffer,
    host_initial_buffer, state->walk_buffer_length * sizeof(uint32_t),
    cudaMemcpyHostToDevice))) {
    free(host_initial_buffer);
    Cleanup(state);
    return NULL;
  }
  free(host_initial_buffer);
  host_initial_buffer = NULL;
  // Check additional_info to see if a custom walk count has been provided.
  if (!SetMemoryAccessCount(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
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

// Returns the ID of the SM this is executed on.
static __device__ __inline__ uint32_t GetSMID(void) {
  uint32_t to_return;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(to_return));
  return to_return;
}

// Returns the value of CUDA's global nanosecond timer.
static __device__ __inline__ uint64_t GlobalTimer64(void) {
  uint64_t to_return;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(to_return));
  return to_return;
}

// Traverses the walk_buffer, taking each subsequent index to visit as the
// value stored in the current index.
static __global__ void WalkKernel(uint64_t access_count,
    uint64_t *accumulator, uint32_t *walk_buffer, uint64_t walk_buffer_length,
    uint64_t *kernel_times, uint64_t *block_times, uint32_t *block_smids) {
  uint64_t start_time = GlobalTimer64();
  uint64_t i = 0;
  uint64_t walk_index = threadIdx.x % walk_buffer_length;
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) kernel_times[0] = start_time;
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  // The actual walk loop.
  for (i = 0; i < access_count; i++) {
    walk_index = walk_buffer[walk_index];
    *accumulator = *accumulator + walk_index;
  }
  // Record the kernel and block end times.
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
  kernel_times[1] = GlobalTimer64();
}

static int Execute(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  WalkKernel<<<state->block_count, state->thread_count, 0, state->stream>>>(
    state->memory_access_count, state->device_accumulator,
    state->device_walk_buffer, state->walk_buffer_length,
    state->device_kernel_times, state->device_block_times,
    state->device_block_smids);
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  BenchmarkState *state = (BenchmarkState *) data;
  KernelTimes *host_times = &state->walk_kernel_times;
  uint64_t block_times_count = state->block_count * 2;
  uint64_t block_smids_count = state->block_count;
  memset(times, 0, sizeof(*times));
  if (!CheckCUDAError(cudaMemcpyAsync(&(state->host_accumulator),
    state->device_accumulator, sizeof(uint64_t), cudaMemcpyDeviceToHost,
    state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(host_times->kernel_times,
    state->device_kernel_times, 2 * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
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
  host_times->kernel_name = "WalkKernel";
  host_times->block_count = state->block_count;
  host_times->thread_count = state->thread_count;
  times->kernel_count = 1;
  times->kernel_info = host_times;
  times->resulting_data = &(state->host_accumulator);
  times->resulting_data_size = sizeof(state->host_accumulator);
  return 1;
}

static const char* GetName(void) {
  return "In-order Walk";
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
