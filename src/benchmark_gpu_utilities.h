// This file contains functions that may be shared across different benchmark
// libraries, such as stream creation or error-checking macros. This should not
// be confused with the standard gpu_utilities library, which creates new
// processes to query the GPU. These functions assume a CUDA context has
// already been created!
#ifndef BENCHMARK_GPU_UTILITIES_H
#define BENCHMARK_GPU_UTILITIES_H
#ifdef __cplusplus
extern "C" {
#endif
#include <cuda_runtime.h>
#include <stdint.h>
#include "library_interface.h"

// This macro takes a cudaError_t value. It prints an error message and returns
// 0 if the cudaError_t isn't cudaSuccess. Otherwise, it returns nonzero.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Prints an error message and returns 0 if the given CUDA result is an error.
// This is intended to be used only via the CheckCUDAError macro.
int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
  const char *file, int line);

// This function creates a CUDA stream with the given stream priority, or, if
// the given stream_priority isn't a valid stream, it will create a stream with
// the default priority. This populates the cudaStream_t pointed to by the
// stream parameter. This will return cudaSuccess on success and a different
// value if an error occurs. Always creates a nonblocking stream.
cudaError_t CreateCUDAStreamWithPriority(int stream_priority,
    cudaStream_t *stream);

// This returns the current CPU time, in seconds. This time will correspond
// to the CPU times obtained by runner.c.
double CurrentSeconds(void);

// GlobalTimer64: Get 64-bit counter of current time on GPU
// ***This is duplicated in benchmark_gpu_utilities.h***
#if __CUDA_ARCH__ >= 300 // Kepler+
// Returns the value of CUDA's global nanosecond timer.
// Starting with sm_30, `globaltimer64` was added
// Tuned for sm_5X architectures, but tested to still be efficient on sm_7X
__device__ inline uint64_t GlobalTimer64(void) {
  uint32_t lo_bits, hi_bits, hi_bits_2;
  uint64_t ret;
  // Upper bits may rollover between our 1st and 2nd read
  // (The bug seems constrained to certain old Jetson boards, so this
  // workaround could probably be gated to only those GPUs.)
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits));
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(lo_bits));
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits_2));
  // If upper bits rolled over, lo_bits = 0
  lo_bits = (hi_bits != hi_bits_2) ? 0 : lo_bits;
  // SASS on older architectures (such as sm_52) is natively 32-bit, so the
  // following three lines get optimized out.
  ret = hi_bits_2;
  ret <<= 32;
  ret |= lo_bits;
  return ret;
}
#elif __CUDA_ARCH__ < 200 // Tesla
// On sm_13, we /only/ have `clock` (32 bits)
__device__ inline uint64_t GlobalTimer64(void) {
  uint32_t lo_bits;
  uint64_t ret;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(lo_bits));
  ret = 0;
  ret |= lo_bits;
  return ret;
}
#else
// Could use clock64 for sm_2x (Fermi), but that's untested
#error Fermi-based GPUs (sm_2x) are unsupported!
#endif

// Returns the ID of the SM this is executed on.
static __device__ __inline__ uint32_t GetSMID(void) {
  uint32_t to_return;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(to_return));
  return to_return;
}

// A convenience function that copies the first dimension of 3-dimensional
// block and grid dimensions from the plugin's parameters. In other words, sets
// thread_count to block_dim[0] and block_count to grid_dim[0]. Returns 0 if
// any entry in block_dim or grid_dim other than the first has been set to a
// value other than 1.  Returns 1 on success.
int GetSingleBlockAndGridDimensions(InitializationParameters *params,
    int *thread_count, int *block_count);

// Like GetSingleBlockAndGridDimensions, but only checks and obtains the first
// dimension of params->block_dim.
int GetSingleBlockDimension(InitializationParameters *params,
    int *thread_count);

#ifdef __cplusplus
} // extern "C"
#endif
#endif  // BENCHMARK_GPU_UTILITIES_H

