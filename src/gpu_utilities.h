// In order to keep runner.c free from CUDA code, any direct interaction
// between runner.c and the GPU will go through functions defined in this file.
#ifndef GPU_UTILITIES_H
#define GPU_UTILITIES_H
#include <stdint.h>
#include "library_interface.h"
#ifdef __cplusplus
extern "C" {
#endif

// Returns the current value of the GPU's globaltimer64 register. Of course,
// this will only be a rough value since there will also be overheads for
// allocating and copying memory. Returns 0 on error. Setting
// USE_DEFAULT_DEVICE (defined in library_interface.h) will cause this to not
// call cudaSetDevice(...).
uint64_t GetCurrentGPUNanoseconds(int cuda_device);

// This invokes a kernel on the GPU with the given grid dimensions that spins
// for the specified number of nanoseconds. This may be used, for example, to
// "release" several other kernels at the same time, by blocking them until a
// spinning kernel has completed. Returns 0 on error and nonzero on success.
// This does *not* block, so an error could actually occur without it being
// detected until the next CUDA call. Once again, cuda_device can be
// USE_DEFAULT_DEVICE.
int SpinGPU(int cuda_device, int thread_count, int block_count,
    uint64_t nanoseconds);

// Returns the maximum number of threads that can be sent to the GPU at once.
// This will be equal to the number of warps per SM * the number of SMs * warp
// size. Returns 0 on error.
int GetMaxResidentThreads(int cuda_device);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // GPU_UTILITIES_H

