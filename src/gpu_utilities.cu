// This file contains the implementation of the functions defined in
// gpu_utilities.h--used by runner.c to work with the GPU.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
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

uint64_t GetCurrentGPUNanoseconds(int cuda_device) {
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
    host_time = 0;
  }
  cudaFree(device_time);
  return host_time;
}

static __global__ void GPUSpinKernel(uint64_t spin_duration) {
  uint64_t start_time = GlobalTimer64();
  uint64_t current_elapsed = 0;
  while (current_elapsed < spin_duration) {
    current_elapsed = GlobalTimer64() - start_time;
  }
}

int SpinGPU(int cuda_device, int thread_count, int block_count,
    uint64_t nanoseconds) {
  if (cuda_device != USE_DEFAULT_DEVICE) {
    if (!CheckCUDAError(cudaSetDevice(cuda_device))) return 0;
  }
  GPUSpinKernel<<<block_count, thread_count>>>(nanoseconds);
  return 1;
}

int GetMaxResidentThreads(int cuda_device) {
  struct cudaDeviceProp properties;
  int warps_per_sm = 64;
  if (!CheckCUDAError(cudaGetDeviceProperties(&properties, cuda_device))) {
    return 0;
  }
  // Compute capability 2.0 devices have a 48 warps per SM.
  if (properties.major <= 2) warps_per_sm = 48;
  return warps_per_sm * properties.multiProcessorCount * properties.warpSize;
}
