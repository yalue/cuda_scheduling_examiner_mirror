// This file defines a bare-bones CUDA benchmark which spins waiting for a
// user-specified amount of time to complete. While the benchmark itself is
// simpler than the mandelbrot-set benchmark, the boilerplate is relatively
// similar.
//
// While this benchmark will spin for an arbitrary default number of
// nanoseconds, the specific amount of time to spin may be given as a number
// of nanoseconds provided in the "additional_info" configuration field.
//
// This benchmark differs from the regular timer_spin only in that it issues
// all work to the default stream, rather than a user-defined stream.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// If no number is provided, spin for this number of nanoseconds.
#define DEFAULT_SPIN_DURATION (10 * 1000 * 1000)

// Holds the local state for one instance of this benchmark.
typedef struct {
  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids;
  // The number of nanoseconds for which each CUDA thread should spin.
  uint64_t spin_duration;
  // Holds the grid dimension to use, set during initialization.
  int block_count;
  int thread_count;
  // Holds host-side times that are shared with the calling process.
  KernelTimes spin_kernel_times;
  // The kernel launch will still use the default stream--this one is only for
  // copying the runtime data from the GPU.
  cudaStream_t stream;
  // Nonzero if stream for copying results has been created.
  int stream_created;
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
// spin_duration by parsing it as a number of nanoseconds. Otherwise, this
// function will set spin_duration to a default value. Returns 0 if the
// argument has been set to an invalid number, or nonzero on success.
static int SetSpinDuration(const char *arg, TaskState *state) {
  int64_t parsed_value;
  if (!arg || (strlen(arg) == 0)) {
    state->spin_duration = DEFAULT_SPIN_DURATION;
    return 1;
  }
  char *end = NULL;
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid spin duration: %s\n", arg);
    return 0;
  }
  state->spin_duration = (uint64_t) parsed_value;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
  // First allocate space for local data.
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
  if (!SetSpinDuration(params->additional_info, state)) {
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

static int Execute(void *data) {
  TaskState *state = (TaskState *) data;
  state->spin_kernel_times.cuda_launch_times[0] = CurrentSeconds();
  GPUSpin<<<state->block_count, state->thread_count>>>(state->spin_duration,
    state->device_block_times, state->device_block_smids);
  state->spin_kernel_times.cuda_launch_times[1] = CurrentSeconds();
  state->spin_kernel_times.cuda_launch_times[2] = 0;
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
  host_times->kernel_name = "GPUSpin";
  host_times->block_count = state->block_count;
  host_times->thread_count = state->thread_count;
  times->kernel_count = 1;
  times->kernel_info = host_times;
  return 1;
}

static const char* GetName(void) {
  return "Timer Spin (default stream)";
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
