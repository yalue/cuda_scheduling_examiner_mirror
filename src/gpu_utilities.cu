// This file contains the implementation of the functions defined in
// gpu_utilities.h--used by runner.c to work with the GPU.
#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "library_interface.h"
#include "gpu_utilities.h"

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

// Returns the value of CUDA's global nanosecond timer.
static __device__ __inline__ uint64_t GlobalTimer64(void) {
  uint64_t to_return;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(to_return));
  return to_return;
}

// A simple kernel which writes the value of the globaltimer64 register to a
// location in device memory.
static __global__ void GetTime(uint64_t *time) {
  *time = GlobalTimer64();
}

// Allocates a private shared memory buffer containing the given number of
// bytes. Can be freed by using FreeSharedBuffer. Returns NULL on error.
// Initializes the buffer to contain 0.
static void* AllocateSharedBuffer(size_t size) {
  void *to_return = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS |
    MAP_SHARED, -1, 0);
  if (to_return == MAP_FAILED) return NULL;
  memset(to_return, 0, size);
  return to_return;
}

// Frees a shared buffer returned by AllocateSharedBuffer.
static void FreeSharedBuffer(void *buffer, size_t size) {
  munmap(buffer, size);
}

// This function should be run in a separate process in order to read the GPU's
// nanosecond counter. Returns 0 on error.
static uint64_t InternalReadGPUNanoseconds(int cuda_device) {
  uint64_t *device_time = NULL;
  uint64_t host_time = 0;
  if (cuda_device != USE_DEFAULT_DEVICE) {
    if (!CheckCUDAError(cudaSetDevice(cuda_device))) return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&device_time, sizeof(*device_time)))) {
    return 0;
  }
  GetTime<<<1, 1>>>(device_time);
  if (!CheckCUDAError(cudaMemcpy(&host_time, device_time, sizeof(host_time),
    cudaMemcpyDeviceToHost))) {
    cudaFree(device_time);
    return 0;
  }
  cudaFree(device_time);
  return host_time;
}

uint64_t GetCurrentGPUNanoseconds(int cuda_device) {
  uint64_t *shared_time = NULL;
  uint64_t to_return = 0;
  int status;
  pid_t pid = -1;
  shared_time = (uint64_t *) AllocateSharedBuffer(sizeof(*shared_time));
  if (!shared_time) {
    printf("Failed allocating shared buffer for IPC.\n");
    return 0;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process to get GPU time: %s\n", strerror(
      errno));
    return 0;
  }
  if (pid == 0) {
    // The following CUDA code is run in the child process
    *shared_time = InternalReadGPUNanoseconds(cuda_device);
    exit(0);
  }
  // The parent will wait for the child to finish, then return the value
  // written to the shared buffer.
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(shared_time, sizeof(*shared_time));
    return 0;
  }
  to_return = *shared_time;
  FreeSharedBuffer(shared_time, sizeof(*shared_time));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  return to_return;
}

// This function should always be run in a separate process.
static int InternalGetMaxResidentThreads(int cuda_device) {
  struct cudaDeviceProp properties;
  int warps_per_sm = 64;
  if (!CheckCUDAError(cudaGetDeviceProperties(&properties, cuda_device))) {
    return 0;
  }
  // Compute capability 2.0 devices have a 48 warps per SM.
  if (properties.major <= 2) warps_per_sm = 48;
  return warps_per_sm * properties.multiProcessorCount * properties.warpSize;
}

int GetMaxResidentThreads(int cuda_device) {
  int to_return, status;
  pid_t pid = -1;
  int *max_thread_count = NULL;
  max_thread_count = (int *) AllocateSharedBuffer(sizeof(*max_thread_count));
  if (!max_thread_count) {
    printf("Failed allocating shared buffer for IPC.\n");
    return 0;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process to get thread count: %s\n",
      strerror(errno));
    return 0;
  }
  if (pid == 0) {
    // The following CUDA code is run in the child process
    *max_thread_count = InternalGetMaxResidentThreads(cuda_device);
    exit(0);
  }
  // The parent will wait for the child to finish, then return the value
  // written to the shared buffer.
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(max_thread_count, sizeof(*max_thread_count));
    return 0;
  }
  to_return = *max_thread_count;
  FreeSharedBuffer(max_thread_count, sizeof(*max_thread_count));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  return to_return;
}
