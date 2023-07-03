// This file contains the implementation of the library defined by
// benchmark_gpu_utilities.h.
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "benchmark_gpu_utilities.h"
#ifdef SMCTRL
#include <libsmctrl.h>
#endif

int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
  const char *file, int line) {
  if (result == cudaSuccess) return 1;
  printf("CUDA error %d in %s, line %d (%s)\n", (int) result, file, line, fn);
  return 0;
}

cudaError_t CreateCUDAStreamWithPriorityAndMask(int stream_priority,
    uint64_t sm_mask, cudaStream_t *stream) {
  cudaError_t result;
  int lowest_priority, highest_priority;
  result = cudaDeviceGetStreamPriorityRange(&lowest_priority,
    &highest_priority);
  if (result != cudaSuccess) return result;
  // Low priorities are higher numbers than high priorities.
  if ((stream_priority > lowest_priority) || (stream_priority <
    highest_priority)) {
    result = cudaStreamCreate(stream);
  } else {
    result = cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking,
      stream_priority);
  }
#ifdef SMCTRL
  libsmctrl_set_stream_mask(*stream, sm_mask);
#endif
  return result;
}

double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

int GetSingleBlockAndGridDimensions(InitializationParameters *params,
    int *thread_count, int *block_count) {
  int a, b;
  if ((params->block_dim[1] != 1) || (params->block_dim[2] != 1)) {
    printf("Expected 1-D block dimensions, but got [%d, %d, %d]\n",
      params->block_dim[0], params->block_dim[1], params->block_dim[2]);
    return 0;
  }
  if ((params->grid_dim[1] != 1) || (params->grid_dim[2] != 1)) {
    printf("Expected 1-D grid dimensions, but got [%d, %d, %d]\n",
      params->grid_dim[0], params->grid_dim[1], params->grid_dim[2]);
    return 0;
  }
  a = params->block_dim[0];
  if ((a < 1) || (a > 1024)) {
    printf("Invalid number of threads in a block: %d\n", a);
    return 0;
  }
  b = params->grid_dim[0];
  if (b < 1) {
    printf("Invalid number of blocks: %d\n", b);
  }
  *thread_count = a;
  *block_count = b;
  return 1;
}

int GetSingleBlockDimension(InitializationParameters *params,
    int *thread_count) {
  int x, y, z;
  x = params->block_dim[0];
  y = params->block_dim[1];
  z = params->block_dim[2];
  if ((y != 1) || (z != 1)) {
    printf("Expected 1-D block dimensions, but got [%d, %d, %d]\n", x, y, z);
    return 0;
  }
  *thread_count = x;
  return 1;
}
