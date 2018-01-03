// In order to keep task_host.c free from CUDA code, any direct interaction
// between task_host.c and the GPU will go through functions defined in this
// file.
#ifndef TASK_HOST_UTILITIES_H
#define TASK_HOST_UTILITIES_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Returns the closest-correlated CPU and GPU times for the given device. These
// times can then be used to calculate the approximate mapping between CPU and
// GPU time, in combination with GetGPUTimerScale. Returns 0 on error.
int GetHostDeviceTimeOffset(int cuda_device, double *host_seconds,
  uint64_t *gpu_nanoseconds);

// Returns the maximum number of threads that can be sent to the GPU at once.
// This will be equal to the number of warps per SM * the number of SMs * warp
// size. Returns 0 on error.
int GetMaxResidentThreads(int cuda_device);

// Returns the number by which GPU time durations must be multiplied to get
// correct (CPU) time durations. On proper GPUs this should return something
// close to 1. Unfortunately, that's not the case on the TX1... Returns a
// negative value on error. This will block while the GPU is being timed.
double GetGPUTimerScale(int cuda_device);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // TASK_HOST_UTILITIES_H

