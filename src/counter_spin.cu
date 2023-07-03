// This file defines a CUDA benchmark which spins for a set number of
// iterations. Like timer_spin, it is very simple, but unlike timer_spin it
// should perform a constant amount of processing work, rather than simply
// waiting for a set amount of time. Therefore, this benchmark's runtime should
// be subject to other workloads running on the GPU.
//
// The specific number of loop iterations to run is given as an integer value
// in the "additional_info" configuration object. If this value isn't set, then
// the benchmark will execute an arbitrary constant number of operations.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// If no number is provided, execute this number of operations.
#define DEFAULT_LOOP_ITERATIONS (1 * 1000 * 1000)

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
  // The number of iterations the kernel's loop should spin for.
  uint64_t loop_iterations;
  // Holds the grid dimension to use, set during initialization.
  int block_count;
  int thread_count;
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
  // Allocate device memory
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

// If the given argument is a non-NULL, non-empty string, attempts to set the
// loop_iterations by parsing it as a number of operations. Otherwise, this
// function will set loop_iterations to a default value. Returns 0 if the
// argument has been set to an invalid number, or nonzero on success.
static int SetLoopIterations(const char *arg, TaskState *state) {
  int64_t parsed_value;
  if (!arg || (strlen(arg) == 0)) {
    state->loop_iterations = DEFAULT_LOOP_ITERATIONS;
    return 1;
  }
  char *end = NULL;
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid operations count: %s\n", arg);
    return 0;
  }
  state->loop_iterations = (uint64_t) parsed_value;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
  state = (TaskState *) calloc(1, sizeof(*state));
  if (!state) return NULL;
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
  if (!SetLoopIterations(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
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

// Spins in a loop until the set number of loop iterations have completed. The
// throwaway argument can be NULL; it's only used to prevent optimizing out the
// loop body.
static __global__ void CounterSpin(uint64_t iterations, uint64_t *block_times,
    uint32_t *block_smids, uint64_t *throwaway) {
  uint64_t start_time = GlobalTimer64();
  uint64_t i, accumulator;
  // Start by recording the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();
  for (i = 0; i < iterations; i++) {
    accumulator += blockIdx.x;
  }
  // By leaving the possibility that the value may be used, we prevent the loop
  // from being removed.
  if (throwaway) *throwaway = accumulator;
  block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
}

static int Execute(void *data) {
  TaskState *state = (TaskState *) data;
  state->spin_kernel_times.cuda_launch_times[0] = CurrentSeconds();
  CounterSpin<<<state->block_count, state->thread_count, 0, state->stream>>>(
    state->loop_iterations, state->device_block_times,
    state->device_block_smids, NULL);
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
  host_times->kernel_name = "CounterSpin";
  host_times->block_count = state->block_count;
  host_times->thread_count = state->thread_count;
  times->kernel_count = 1;
  times->kernel_info = host_times;
  return 1;
}

static const char* GetName(void) {
  return "Counter Spin";
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
