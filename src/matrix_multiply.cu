// This plugin multiplies two square matrices, using a not-particularly-
// optimized matrix multiply kernel. This uses a 2D grid of 2D blocks, mapping
// one thread per resulting matrix element. The block_count setting from the
// config is ignored, and thread_count must be 2 dimensional. The
// additional_info must be a JSON object with the following keys:
//
// - "matrix_width": The width of the square matrix of floating-point numbers
//   to multiply.
//
// - "skip_copy": A boolean. Optional, defaults to false. If it's set to true,
//   then the "copy_in" and "copy_out" phases will not copy the matrix data to
//   or from the GPU. Instead, the input matrices will only be copied once,
//   during the initialize function, and output data will never be copied. May
//   be useful if you want a simple workload without as many memory transfers.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "third_party/cJSON.h"
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// Holds the state of an instance of this plugin.
typedef struct {
  cudaStream_t stream;
  // Tracking whether the stream is created simplifies cleanup code.
  int stream_created;
  // The grid size is computed based on the matrix width.
  dim3 grid_size;
  // The block size is determined by the thread_count setting in the config.
  dim3 block_size;
  // Set this to nonzero in order to not copy input or output matrices, apart
  // from during initialization.
  int skip_copy;
  // The width of each square matrix.
  int matrix_width;
  // The device-side matrices. The computation will be d_c = d_a x d_b.
  float *d_a, *d_b, *d_c;
  // The three host-side copies of the matrices.
  float *h_a, *h_b, *h_c;
  // The recordings of the start and end GPU clock cycle for each block.
  uint64_t *device_block_times;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (!state) return;
  cudaFree(state->d_a);
  cudaFree(state->d_b);
  cudaFree(state->d_c);
  cudaFreeHost(state->h_a);
  cudaFreeHost(state->h_b);
  cudaFreeHost(state->h_c);
  cudaFreeHost(state->kernel_times.block_times);
  cudaFreeHost(state->kernel_times.block_smids);
  if (state->stream_created) {
    CheckCUDAError(cudaStreamDestroy(state->stream));
  }
  // Free device memory.
  if (state->device_block_times) cudaFree(state->device_block_times);
  if (state->device_block_smids) cudaFree(state->device_block_smids);

  memset(state, 0, sizeof(*state));
  free(state);
}

static float RandomFloat(void) {
  // Maybe replace this with something faster?
  float to_return = ((float) rand()) / ((float) RAND_MAX);
  return to_return;
}

// Allocates the host and device matrices, and randomly initializes them.
// Returns 0 on error. Must be called after grid_size has been initialized.
static int AllocateMemory(PluginState *state) {
  int i, j, width, block_count;
  size_t size, smids_size;
  width = state->matrix_width;
  block_count = state->grid_size.x * state->grid_size.y;

  // Allocate host and device memory for block times.
  size = block_count * 2 * sizeof(uint64_t);
  if (!CheckCUDAError(cudaMallocHost(&(state->kernel_times.block_times),
    size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times), size))) {
    return 0;
  }
  smids_size = block_count * sizeof(uint32_t);
  if (!CheckCUDAError(cudaMallocHost(&(state->kernel_times.block_smids),
    smids_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids), smids_size))) {
    return 0;
  }

  // Allocate the matrices.
  size = width * width * sizeof(float);
  if (!CheckCUDAError(cudaMalloc(&state->d_a, size))) return 0;
  if (!CheckCUDAError(cudaMalloc(&state->d_b, size))) return 0;
  if (!CheckCUDAError(cudaMalloc(&state->d_c, size))) return 0;
  if (!CheckCUDAError(cudaMallocHost(&state->h_a, size))) return 0;
  if (!CheckCUDAError(cudaMallocHost(&state->h_b, size))) return 0;
  if (!CheckCUDAError(cudaMallocHost(&state->h_c, size))) return 0;
  memset(state->h_c, 0, size);

  // Randomly initialize the host's input matrices.
  for (i = 0; i < width; i++) {
    for (j = 0; j < width; j++) {
      state->h_a[i * width + j] = RandomFloat();
      state->h_b[i * width + j] = RandomFloat();
    }
  }

  // Initialize the input matrices on the host.
  if (!CheckCUDAError(cudaMemcpyAsync(state->d_a, state->h_a, size,
    cudaMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(state->d_b, state->h_b, size,
    cudaMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

// Parses the additional_info argument. Returns 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for matrix_multiply.\n");
    return 0;
  }

  // Make sure that matrix_width is present and positive.
  entry = cJSON_GetObjectItem(root, "matrix_width");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid matrix_width setting.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble < 1.0) {
    printf("The matrix_width setting must be at least 1.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble > 1e6) {
    printf("Warning: huge matrix width: %f\n", entry->valuedouble);
  }
  state->matrix_width = (int) entry->valuedouble;

  // If skip_copy is present, make sure it's a boolean. state->skip_copy is
  // initialized to 0 already.
  entry = cJSON_GetObjectItem(root, "skip_copy");
  if (entry) {
    if ((entry->type != cJSON_True) && (entry->type != cJSON_False)) {
      printf("The skip_copy setting must be a boolean.\n");
      cJSON_Delete(root);
      return 0;
    }
    state->skip_copy = entry->type == cJSON_True;
  }

  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  int blocks_wide, blocks_tall, matrix_width;
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;

  state = (PluginState *) calloc(sizeof(*state), 1);
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }

  // We need to know the matrix width before we can allocate memory or
  // determine grid size.
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  matrix_width = state->matrix_width;

  state->block_size.x = params->block_dim[0];
  if (params->block_dim[1] == 1) {
    // Print a warning here; the old behavior was different so this will help
    // catch and update configs expecting the old version. (e.g. we probably
    // want to use 32x32 blocks rather than 1024x1 blocks!)
    printf("Warning! Specified a 1-D block dim for matrix multiply.\n");
    printf("HACK: Defaulting to 32x32.\n");
    state->block_size.x = 32;
    state->block_size.y = 32;
  }
  state->block_size.y = params->block_dim[1];
  state->block_size.z = 1;

  // Compute the grid size from the block size and matrix width.
  blocks_wide = matrix_width / state->block_size.x;
  if ((matrix_width % state->block_size.x) != 0) blocks_wide++;
  blocks_tall = matrix_width / state->block_size.y;
  if ((matrix_width % state->block_size.y) != 0) blocks_tall++;

  state->grid_size.x = blocks_wide;
  state->grid_size.y = blocks_tall;
  state->grid_size.z = 1;

  // Create the stream and fill in boilerplate for reporting to the framework.
  if (!CheckCUDAError(CreateCUDAStreamWithPriorityAndMask(
    params->stream_priority, params->sm_mask, &(state->stream)))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "matrix_multiply";
  state->kernel_times.thread_count = state->block_size.x * state->block_size.y;
  state->kernel_times.block_count = state->grid_size.x * state->grid_size.y;
  state->kernel_times.shared_memory = 0;

  // Allocate the matrices and initialize the input matrices
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// Reset block times and copy input matrices.
static int CopyIn(void *data) {
  PluginState *state = (PluginState *) data;
  int block_count = state->grid_size.x * state->grid_size.y;

  // Reset block times
  size_t size = block_count * 2 * sizeof(uint64_t);
  if (!CheckCUDAError(cudaMemsetAsync(state->device_block_times, 0xff,
    size, state->stream))) {
    return 0;
  }

  // If we're skipping the copies we can return now.
  if (state->skip_copy) {
    if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Copy input matrices.
  size = state->matrix_width * state->matrix_width * sizeof(float);
  if (!CheckCUDAError(cudaMemcpyAsync(state->d_a, state->h_a, size,
    cudaMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(state->d_b, state->h_b, size,
    cudaMemcpyHostToDevice, state->stream))) {
    return 0;
  }

  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

// The GPU kernel for carrying out matrix multiplication. Expects a 2D grid
// with sufficient threads to cover the entire matrix.
__global__ void MatrixMultiplyKernel(float *a, float *b, float *c, int width,
  uint64_t *block_times, uint32_t *block_smids) {
  int row, col, k, block_index;
  float v_a, v_b, v_c;
  uint64_t start_clock = clock64();
  block_index = blockIdx.y * gridDim.x + blockIdx.x;
  if (start_clock < block_times[block_index * 2]) {
    block_times[block_index * 2] = start_clock;
    block_smids[block_index] = GetSMID();
  }

  // The row and column of the element in the output matrix is determined by
  // the thread's position in the 2D grid.
  col = blockIdx.x * blockDim.x + threadIdx.x;
  row = blockIdx.y * blockDim.y + threadIdx.y;
  if ((col >= width) || (row >= width)) {
    // Don't try doing computations if we're outside of the matrix.
    block_times[block_index * 2 + 1] = clock64();
    return;
  }

  // Actually carry out the multiplication for this thread's element.
  v_c = 0;
  for (k = 0; k < width; k++) {
    v_a = a[row * width + k];
    v_b = b[k * width + col];
    v_c += v_a * v_b;
  }
  c[row * width + col] = v_c;
  block_times[block_index * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.cuda_launch_times[0] = CurrentSeconds();

  MatrixMultiplyKernel<<<state->grid_size, state->block_size, 0, state->stream>>>(
    state->d_a, state->d_b, state->d_c, state->matrix_width,
    state->device_block_times, state->device_block_smids);

  state->kernel_times.cuda_launch_times[1] = CurrentSeconds();
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  state->kernel_times.cuda_launch_times[2] = CurrentSeconds();
  return 1;
}

// Copy the block times to the host, along with the result matrix.
static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  int block_count = state->grid_size.x * state->grid_size.y;
  size_t size;


  // First copy the block times to the host.
  size = block_count * 2 * sizeof(uint64_t);
  if (!CheckCUDAError(cudaMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, size, cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(state->kernel_times.block_smids,
    state->device_block_smids, block_count * sizeof(uint32_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }

  // Provide the framework with a pointer to our kernel_times struct.
  times->kernel_count = 1;
  times->kernel_info = &(state->kernel_times);

  // We can return now if we're not copying the result matrix.
  if (state->skip_copy) {
    times->resulting_data_size = 0;
    times->resulting_data = NULL;
    if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Copy the result matrix.
  size = state->matrix_width * state->matrix_width * sizeof(float);
  if (!CheckCUDAError(cudaMemcpyAsync(state->h_c, state->d_c, size,
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  times->resulting_data_size = size;
  times->resulting_data = state->h_c;
  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Matrix Multiply";
}

int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->get_name = GetName;
  functions->cleanup = Cleanup;
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  return 1;
}

