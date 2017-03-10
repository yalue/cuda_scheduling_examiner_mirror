// This file defines a bare-bones CUDA benchmark which spins waiting for a
// user-specified amount of time to complete. While the benchmark itself is
// simpler than the mandelbrot-set benchmark, the boilerplate is relatively
// similar.
//
// While this benchmark will spin for an arbitrary default number of
// nanoseconds, the specific amount of time to spin may be given as a number
// of nanoseconds provided as a string "additional_info" configuration field.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "library_interface.h"

// If no number is provided, spin for this number of nanoseconds.
#define DEFAULT_SPIN_DURATION (10 * 1000 * 1000)

// This macro takes a cudaError_t value. It prints an error message and returns
// 0 if the cudaError_t isn't cudaSuccess. Otherwise, it returns nonzero.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Prints an error message and returns 0 if the given CUDA result is an error.
static int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
    const char *file, int line) {
  if (result == cudaSuccess) return 1;
  printf("CUDA error %d in %s, line %d (%s)\n", (int) result, file, line, fn);
  return 0;
}

// Holds the local state for one instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // This will be set to 0 if the CUDA stream hasn't been created yet. This is
  // useful because it allows us to unconditionally call Cleanup on error
  // without needing to worry about calling cudaStreamDestroy twice.
  int stream_created;
  // Holds host and device copies of the start and end times of the most recent
  // kernel to have completed.
  uint64_t *device_kernel_times;
  uint64_t host_kernel_times[2];
  // Holds host and device copies of the start and end times of each thread
  // block from the most recent kernel to have completed.
  uint64_t *device_block_times;
  uint64_t *host_block_times;
  // The number of nanoseconds for which each CUDA thread should spin.
  uint64_t spin_duration;
  // Holds the grid dimension to use, set during initialization.
  int block_count;
  int thread_count;
} BenchmarkState;

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  if (state->device_kernel_times) cudaFree(state->device_kernel_times);
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->host_block_times) free(state->host_block_times);
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
  if (!CheckCUDAError(cudaMalloc(&(state->device_kernel_times),
    sizeof(state->host_kernel_times)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times),
    block_times_size))) {
    return 0;
  }
  state->host_block_times = (uint64_t *) malloc(block_times_size);
  if (!state->host_block_times) {
    return 0;
  }
  return 1;
}

// If the given argument is a non-NULL, non-empty string, attempts to set the
// spin_duration by parsing it as a number of nanoseconds. Otherwise, this
// function will set spin_duration to a default value. Returns 0 if the
// argument has been set to an invalid number, or nonzero on success.
static int SetSpinDuration(const char *arg, BenchmarkState *state) {
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
  BenchmarkState *state = NULL;
  // First allocate space for local data.
  state = (BenchmarkState *) malloc(sizeof(*state));
  memset(state, 0, sizeof(*state));
  if (params->cuda_device != USE_DEFAULT_DEVICE) {
    if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  }
  // Round the thread count up to a value evenly divisible by 32.
  if ((params->thread_count % WARP_SIZE) != 0) {
    params->thread_count += WARP_SIZE - (params->thread_count % WARP_SIZE);
  }
  state->thread_count = params->thread_count;
  state->block_count = params->block_count;
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  if (!SetSpinDuration(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  if (!CheckCUDAError(cudaStreamCreate(&(state->stream)))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  return state;
}

// This sets up a simple mechanism for detecting overall kernel start time,
// which requires initializing the device's copy of the kernel start time to
// the maximum 64-bit integer.
static int CopyIn(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  // Set the host's start kernel time to the highest-possible time by setting
  // it to 0 and subtracting 1.
  memset(state->host_kernel_times, 0, sizeof(state->host_kernel_times));
  state->host_kernel_times[0]--;
  if (!CheckCUDAError(cudaMemcpyAsync(state->device_kernel_times,
    state->host_kernel_times, sizeof(state->host_kernel_times),
    cudaMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

// Returns the value of CUDA's global nanosecond timer.
static __device__ __inline__ uint64_t GlobalTimer64(void) {
  uint64_t to_return;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(to_return));
  return to_return;
}

// Spins in a loop until at least spin_duration nanoseconds have elapsed.
static __global__ void GPUSpin(uint64_t spin_duration, uint64_t *kernel_times,
    uint64_t *block_times) {
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (kernel_times[0] > start_time) kernel_times[0] = start_time;
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
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
  kernel_times[1] = GlobalTimer64();
}

static int Execute(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  GPUSpin<<<state->block_count, state->thread_count, 0, state->stream>>>(
    state->spin_duration, state->device_kernel_times,
    state->device_block_times);
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  BenchmarkState *state = (BenchmarkState *) data;
  uint64_t block_times_count = state->block_count * 2;
  memset(times, 0, sizeof(*times));
  if (!CheckCUDAError(cudaMemcpyAsync(state->host_kernel_times,
    state->device_kernel_times, sizeof(state->host_kernel_times),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(state->host_block_times,
    state->device_block_times, block_times_count * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  times->kernel_times_count = 2;
  times->kernel_times = state->host_kernel_times;
  times->block_times_count = block_times_count;
  times->block_times = state->host_block_times;
  return 1;
}

static const char* GetName(void) {
  return "Timer Spin";
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
