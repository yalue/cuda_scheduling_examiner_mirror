// In order to keep runner.c free from CUDA code, any direct interaction
// between runner.c and the GPU will go through functions defined in this file.
#ifndef GPU_UTILITIES_H
#define GPU_UTILITIES_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Returns the current value of the GPU's globaltimer64 register. Of course,
// this will only be a rough value since there will also be overheads for
// allocating and copying memory. Returns 0 on error.
uint64_t GetCurrentGPUNanoseconds(int cuda_device);

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
#endif  // GPU_UTILITIES_H

