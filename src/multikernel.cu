// This file defines a CUDA benchmark which issues multiple kernels to a single
// stream before waiting for all kernels to complete. The configuration for the
// kernels is taken exclusively from the additional_info field in the
// InitializationParameters struct. The actual kernels will simply be instances
// of the same kernel as in the timer_spin benchmark. This benchmark ignores
// all fields in its initialization parameters apart from cuda_device and
// additional_info.
//
// The format of the necessary additional_info string is as follows:
// "kernel_1_name,<ns to spin>,<# blocks>,<# threads>,kernel_2_name,...".
//
// Essentially, the string will be a comma-separated list of arguments, the
// first of which is a name to be given to the kernel, the second is the number
// of nanoseconds the kernel should spin, the third is the block count, and the
// fourth is the thread count. One kernel is created for every group of 4
// arguments. Kernel names can't contain commas. Kernels are issued to the
// stream in the same order that they're specified in additional_info.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// Holds the parameters for a single kernel's execution.
typedef struct {
  // The name given to this kernel.
  char *name;
  // The number of nanoseconds this kernel should spin for.
  uint64_t spin_duration;
  // The grid dimensions for this kernel
  int block_count;
  int thread_count;
  // The host and device memory buffers for the kernel.
  uint64_t *host_kernel_times;
  uint64_t *device_kernel_times;
  uint64_t *host_block_times;
  uint64_t *device_block_times;
  uint32_t *host_smids;
  uint32_t *device_smids;
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
  tmp64 = kernel_config->host_kernel_times;
  if (tmp64) cudaFreeHost(tmp64);
  tmp64 = kernel_config->host_block_times;
  if (tmp64) cudaFreeHost(tmp64);
  tmp32 = kernel_config->host_smids;
  if (tmp32) cudaFreeHost(tmp32);
  if (kernel_config->name) free(kernel_config->name);
  // Free GPU memory for this kernel.
  tmp64 = kernel_config->device_kernel_times;
  if (tmp64) cudaFree(tmp64);
  tmp64 = kernel_config->device_block_times;
  if (tmp64) cudaFree(tmp64);
  tmp32 = kernel_config->device_smids;
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
  if ((temp_token_count % 4) != 0) {
    printf("The benchmark requires a comma-separated list of arguments with a "
      "number of entries divisible by 4.\n");
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

// Allocates host and device memory for a single kernel config struct. Returns
// 0 on error and nonzero on success. Must be called after block size has been
// set.
static int AllocateKernelDataMemory(SingleKernelData *config) {
  size_t block_times_size = 2 * config->block_count * sizeof(uint64_t);
  size_t smids_size = config->block_count * sizeof(uint32_t);
  // Allocate host memory
  if (!CheckCUDAError(cudaMallocHost(&config->host_kernel_times, 2 *
    sizeof(uint64_t)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&config->host_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&config->host_smids, smids_size))) {
    return 0;
  }
  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&config->device_kernel_times, 2 *
    sizeof(uint64_t)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&config->device_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&config->device_smids, smids_size))) {
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
  if (!CheckCUDAError(cudaMemcpyAsync(config->host_kernel_times,
    config->device_kernel_times, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost,
    stream))) {
    return 0;
  }
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
  int token_count = 0;
  int kernel_count = 0;
  int i = 0;
  int j;
  uint64_t parsed_number = 0;
  char **tokens = NULL;
  SingleKernelData *kernel_configs = NULL;
  state->kernel_configs = NULL;
  state->kernel_count = 0;
  if (!SplitAdditionalInfoTokens(info, &tokens, &token_count)) return 0;
  kernel_count = token_count / 4;
  // I don't think SplitAdditionalTokens will report a success in this case,
  // but it still shouldn't hurt at this point if 0 kernels were specified.
  if (kernel_count == 0) return 1;
  kernel_configs = (SingleKernelData *) calloc(kernel_count,
    sizeof(*kernel_configs));
  if (!kernel_configs) goto ErrorCleanup;
  for (i = 0; i < kernel_count; i++) {
    j = i * 4;
    kernel_configs[i].name = strdup(tokens[j]);
    if (!kernel_configs[i].name) goto ErrorCleanup;
    if (!StringToUint64(tokens[j + 1], &parsed_number)) goto ErrorCleanup;
    kernel_configs[i].spin_duration = parsed_number;
    if (!StringToUint64(tokens[j + 2], &parsed_number)) goto ErrorCleanup;
    kernel_configs[i].block_count = parsed_number;
    if (!StringToUint64(tokens[j + 3], &parsed_number)) goto ErrorCleanup;
    kernel_configs[i].thread_count = parsed_number;
    // Round thread count up to an amount evenly divisible by WARP_SIZE
    if ((kernel_configs[i].thread_count % WARP_SIZE) != 0) {
      kernel_configs[i].thread_count += WARP_SIZE -
        (kernel_configs[i].thread_count % WARP_SIZE);
    }
    if (!AllocateKernelDataMemory(kernel_configs + i)) goto ErrorCleanup;
  }
  state->kernel_configs = kernel_configs;
  state->kernel_count = kernel_count;
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
  if (kernel_configs) {
    for (i = 0; i < kernel_count; i++) {
      CleanupKernelConfig(kernel_configs + i);
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

// Spins in a loop until at least spin_duration nanoseconds have elapsed.
static __global__ void GPUSpin(uint64_t spin_duration, uint64_t *kernel_times,
    uint64_t *block_times, uint32_t *block_smids) {
  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) kernel_times[0] = start_time;
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
  kernel_times[1] = GlobalTimer64();
}

static int Execute(void *data) {
  BenchmarkState *state = (BenchmarkState *) data;
  SingleKernelData *config = NULL;
  int i;
  // Submit all of the kernels before calling cudaStreamSynchronize
  for (i = 0; i < state->kernel_count; i++) {
    config = state->kernel_configs + i;
    GPUSpin<<<config->block_count, config->thread_count, 0, state->stream>>>(
      config->spin_duration, config->device_kernel_times,
      config->device_block_times, config->device_smids);
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
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
    time_to_return->kernel_times = config->host_kernel_times;
    time_to_return->block_times = config->host_block_times;
    time_to_return->block_smids = config->host_smids;
    time_to_return->block_count = config->block_count;
    time_to_return->thread_count = config->thread_count;
    time_to_return->kernel_name = config->name;
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
