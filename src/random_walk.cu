// This file defines a CUDA benchmark in which a kernel will perform a sequence
// of memory reads in a random cycle over a buffer in GPU memory.
//
// By default, this benchmark performs an arbitrary constant number of memory
// references, but a specific number of references can be provided in the
// "additional_info" field.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// The default number of memory reads to perform in each iteration.
#define DEFAULT_MEMORY_ACCESS_COUNT (1000)

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
  // Holds the buffer of GPU memory traversed during the walk.
  uint64_t *device_walk_buffer;
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
} TaskState;

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->walk_kernel_times;
  // Free device memory.
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->device_block_smids) cudaFree(state->device_block_smids);
  if (state->device_walk_buffer) cudaFree(state->device_walk_buffer);
  if (state->device_accumulator) cudaFree(state->device_accumulator);
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
  KernelTimes *host_times = &state->walk_kernel_times;
  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times),
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids),
    block_smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_walk_buffer),
    state->walk_buffer_length * sizeof(uint64_t)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_accumulator),
    sizeof(uint64_t)))) {
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

// If the given argument is a non-NULL, non-empty string, attempts to set
// memory_access_count by parsing it. Otherwise, this function will set the
// count to a default value. Returns 0 on error.
static int SetMemoryAccessCount(const char *arg, TaskState *state) {
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

// Returns a single random 64-bit value.
static uint64_t Random64(void) {
  int i;
  uint64_t to_return = 0;
  // Get a random number in 16-bit chunks
  for (i = 0; i < 4; i++) {
    to_return = to_return << 16;
    to_return |= rand() & 0xffff;
  }
  return to_return;
}

// Returns a random 64-bit integer in the range [base, limit)
static uint64_t RandomRange(uint64_t base, uint64_t limit) {
  if (limit <= base) return base;
  return (Random64() % (limit - base)) + base;
}

// Shuffles an array of 32-bit values.
static void ShuffleArray(uint64_t *buffer, uint64_t element_count) {
  uint64_t tmp;
  uint64_t i, dst;
  for (i = 0; i < element_count; i++) {
    dst = RandomRange(i, element_count);
    tmp = buffer[i];
    buffer[i] = buffer[dst];
    buffer[dst] = tmp;
  }
}

// This kernel uses a single thread to access every element in the given
// walk_buffer, which should bring (small) buffers into the GPU cache. The
// accumulator can be NULL, and is used to prevent optimizations from removing
// the kernel entirely.
static __global__ void InitialWalk(uint64_t *walk_buffer,
    uint64_t buffer_length, uint64_t *accumulator) {
  uint64_t i = 0;
  uint64_t result = 0;
  if (blockIdx.x != 0) return;
  if (threadIdx.x != 0) return;
  for (i = 0; i < buffer_length; i++) {
    result += walk_buffer[i];
  }
  if (accumulator != NULL) *accumulator = result;
}

static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
  uint64_t i;
  uint64_t *host_initial_buffer = NULL;
  // First allocate space for local data.
  state = (TaskState *) calloc(1, sizeof(*state));
  if (!state) return NULL;
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  if (!GetSingleBlockAndGridDimensions(params, &state->thread_count,
    &state->block_count)) {
    Cleanup(state);
    return NULL;
  }
  state->walk_buffer_length = params->data_size / sizeof(uint64_t);
  if (state->walk_buffer_length <= 0) {
    printf("Memory walks require a data_size of at least %d.\n",
      (int) sizeof(uint64_t));
    Cleanup(state);
    return NULL;
  }
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  // Now that the device buffer is allocated, initialize it for a random walk.
  // This is the only change over the in-order walk.
  host_initial_buffer = (uint64_t *) malloc(state->walk_buffer_length *
    sizeof(uint64_t));
  if (!host_initial_buffer) {
    printf("Failed allocating host buffer for initializing GPU memory.\n");
    Cleanup(state);
    return NULL;
  }
  // Initialize a random cycle by shuffling array indices.
  for (i = 0; i < state->walk_buffer_length; i++) {
    host_initial_buffer[i] = i;
  }
  ShuffleArray(host_initial_buffer, state->walk_buffer_length);
  if (!CheckCUDAError(cudaMemcpy(state->device_walk_buffer,
    host_initial_buffer, state->walk_buffer_length * sizeof(uint64_t),
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
  if (!CheckCUDAError(CreateCUDAStreamWithPriorityAndMask(
    params->stream_priority, params->sm_mask, &(state->stream)))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  // Bring the buffer into the GPU cache by accessing it once.
  InitialWalk<<<1, 1, 0, state->stream>>>(state->device_walk_buffer,
    state->walk_buffer_length, NULL);
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// Nothing needs to be copied in for this benchmark.
static int CopyIn(void *data) {
  return 1;
}

// Traverses the walk_buffer, taking each subsequent index to visit as the
// value stored in the current index.
static __global__ void WalkKernel(uint64_t access_count,
    uint64_t *accumulator, uint64_t *walk_buffer, uint64_t walk_buffer_length,
    uint64_t *block_times, uint32_t *block_smids) {
  uint64_t start_time = GlobalTimer64();
  uint64_t i = 0;
  uint64_t walk_index = threadIdx.x % walk_buffer_length;
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
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
}

static int Execute(void *data) {
  TaskState *state = (TaskState *) data;
  state->walk_kernel_times.cuda_launch_times[0] = CurrentSeconds();
  WalkKernel<<<state->block_count, state->thread_count, 0, state->stream>>>(
    state->memory_access_count, state->device_accumulator,
    state->device_walk_buffer, state->walk_buffer_length,
    state->device_block_times, state->device_block_smids);
  state->walk_kernel_times.cuda_launch_times[1] = CurrentSeconds();
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  state->walk_kernel_times.cuda_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->walk_kernel_times;
  uint64_t block_times_count = state->block_count * 2;
  uint64_t block_smids_count = state->block_count;
  memset(times, 0, sizeof(*times));
  if (!CheckCUDAError(cudaMemcpyAsync(&(state->host_accumulator),
    state->device_accumulator, sizeof(uint64_t), cudaMemcpyDeviceToHost,
    state->stream))) {
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
  return "Random Walk";
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
