// This file defines a bare-bones CUDA benchmark which spins accesses shared
// memory and then waits for a user-specified amount of time to complete.
// While the benchmark itself is simpler than the mandelbrot-set benchmark,
// the boilerplate is relatively similar.
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
  // Holds the device copy of the overall start and end time of the kernel.
  uint64_t *device_kernel_times;
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
} BenchmarkState;

// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  KernelTimes *host_times = &state->spin_kernel_times;
  // Free device memory.
  if (state->device_kernel_times) cudaFree(state->device_kernel_times);
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->device_block_smids) cudaFree(state->device_block_smids);
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
  KernelTimes *host_times = &state->spin_kernel_times;
  // Allocate device memory.
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

// Splits the additional_input field into comma-separated tokens. Sets the
// tokens argument to a pointer to an array of strings, and the token_count
// argument to the number of strings in the tokens array. Returns 0 on error,
// or nonzero on success. Both the individual token pointers and the list of
// pointers returned by this function must be freed by the caller. Fails if the
// number of tokens is not divisible by 4.
static int SplitAdditionalInfoTokens(const char *input, char ***tokens,
    int *token_count) {
  char **to_return = NULL;
  char *input_copy = NULL;
  size_t input_length = strlen(input);
  char *token_start = NULL;
  uint32_t temp_token_count = 0;
  size_t i = 0;
  char c = 0;
  *tokens = NULL;
  *token_count = 0;
  if (input_length == 0) {
    printf("The multikernel benchmark was created with no kernels to run.");
    return 0;
  }
  // First, count the number of tokens in the input string, and build a copy of
  // the input will commas replaced with null bytes.
  input_copy = (char *) malloc(input_length + 1);
  if (!input_copy) return 0;
  memset(input_copy, 0, input_length + 1);
  for (i = 0; i < input_length; i++) {
    c = input[i];
    // This should never happen...
    if (c == 0) break;
    // Copy non-comma characters
    if (c != ',') {
      input_copy[i] = c;
      continue;
    }
    // If a comma is seen, increment the token count and replace it with a null
    // byte.
    input_copy[i] = 0;
    temp_token_count++;
  }
  // The last token won't have had a comma, so increment the count of tokens.
  temp_token_count++;
  if (temp_token_count != 2) {
    printf("The benchmark requires a comma-separated list of arguments with "
      "two entries.\n");
    goto ErrorCleanup;
  }
  // Allocate the list of pointers. The ErrorCleanup code needs this to be
  // zeroed, so use calloc.
  to_return = (char **) calloc(temp_token_count, sizeof(char *));
  if (!to_return) goto ErrorCleanup;
  // Finally, duplicate the strings by seeking past the null characters and
  // using strdup.
  token_start = input_copy;
  for (i = 0; i < temp_token_count; i++) {
    to_return[i] = strdup(token_start);
    if (!to_return[i]) goto ErrorCleanup;
    // Advance to the byte past the next null byte
    while (*token_start != 0) {
      token_start++;
    }
    token_start++;
  }
  free(input_copy);
  *tokens = to_return;
  *token_count = temp_token_count;
  return 1;
ErrorCleanup:
  if (input_copy) free(input_copy);
  // Free any existing string copies in addition to the list of pointers.
  if (to_return) {
    for (i = 0; i < temp_token_count; i++) {
      if (to_return[i]) free(to_return[i]);
    }
    free(to_return);
  }
  return 0;
}

// Parses an input value to a uint64_t. Returns 0 on error or nonzero on
// success.
static int StringToUint64(const char *input, uint64_t *parsed) {
  char *end = NULL;
  uint64_t parsed_value;
  *parsed = 0;
  parsed_value = strtoull(input, &end, 10);
  if ((end == input) || (*end != 0)) {
    printf("Invalid integer: %s\n", input);
    return 0;
  }
  *parsed = parsed_value;
  return 1;
}

// Parses the additional info setting to get the spin duration and the
// amount of static shared memory to use. Attempts to set the spin_duration
// by parsing it as a number of nanoseconds. For shared memory, it must
// be one of [4096, 8192, 10240], as these are statically-defined.
// Returns 0 if the arguments have been set to invalid numbers, or nonzero
// on success.
static int InitializeKernelConfigs(BenchmarkState *state, char *info) {
  int token_count = 0;
  int i = 0;
  uint64_t parsed_number = 0;
  char **tokens = NULL;
  if (!SplitAdditionalInfoTokens(info, &tokens, &token_count)) return 0;
  // Spin duration
  if (!StringToUint64(tokens[0], &parsed_number)) goto ErrorCleanup;
  state->spin_duration = parsed_number;
  // Shared memory
  if (!StringToUint64(tokens[1], &parsed_number)) goto ErrorCleanup;
  if ((parsed_number != 4096) && (parsed_number != 8192) &&
      (parsed_number != 10240)) goto ErrorCleanup; 
  state->sharedmem_count = parsed_number;
  for (i = 0; i < token_count; i++) {
    free(tokens[i]);
  }
  free(tokens);
  return 1;
ErrorCleanup:
  if (tokens) {
    for (i = 0; i < token_count; i++) {
      free(tokens[i]);
    }
    free(tokens);
  }
  return 0;
}

static void* Initialize(InitializationParameters *params) {
  BenchmarkState *state = NULL;
  // First allocate space for local data.
  state = (BenchmarkState *) malloc(sizeof(*state));
  memset(state, 0, sizeof(*state));
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
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
  // Parse the configuration string (spin duration, shared memory count)
  if (!InitializeKernelConfigs(state, params->additional_info)) return NULL;
  // Create the stream
  if (!CheckCUDAError(cudaStreamCreate(&(state->stream)))) {
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
    uint64_t *kernel_times, uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) kernel_times[0] = start_time;
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
  kernel_times[1] = GlobalTimer64();
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void SharedMem_GPUSpin8192(uint64_t spin_duration,
    uint64_t *kernel_times, uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) kernel_times[0] = start_time;
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
  kernel_times[1] = GlobalTimer64();
}

// Accesses shared memory and spins in a loop until at least
// spin_duration nanoseconds have elapsed.
static __global__ void SharedMem_GPUSpin10240(uint64_t spin_duration,
    uint64_t *kernel_times, uint64_t *block_times, uint32_t *block_smids) {
  uint32_t shared_mem_res;
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) kernel_times[0] = start_time;
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
  kernel_times[1] = GlobalTimer64();
}

static int Execute(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  if (state->sharedmem_count == 4096) {
    SharedMem_GPUSpin4096<<<state->block_count, state->thread_count, 0, state->stream>>>(
      state->spin_duration, state->device_kernel_times,
      state->device_block_times, state->device_block_smids);
  } else if (state->sharedmem_count == 8192) {
    SharedMem_GPUSpin8192<<<state->block_count, state->thread_count, 0, state->stream>>>(
      state->spin_duration, state->device_kernel_times,
      state->device_block_times, state->device_block_smids);
  } else if (state->sharedmem_count == 10240) {
    SharedMem_GPUSpin10240<<<state->block_count, state->thread_count, 0, state->stream>>>(
      state->spin_duration, state->device_kernel_times,
      state->device_block_times, state->device_block_smids);
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  BenchmarkState *state = (BenchmarkState *) data;
  KernelTimes *host_times = &state->spin_kernel_times;
  uint64_t block_times_count = state->block_count * 2;
  uint64_t block_smids_count = state->block_count;
  memset(times, 0, sizeof(*times));
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
  host_times->kernel_name = "SharedMem_GPUSpin";
  host_times->block_count = state->block_count;
  host_times->thread_count = state->thread_count;
  host_times->sharedmem = state->sharedmem_count * sizeof(uint32_t);
  times->kernel_count = 1;
  times->kernel_info = host_times;
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
