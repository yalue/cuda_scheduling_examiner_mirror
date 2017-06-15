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

#ifdef __cplusplus
} // extern "C"
#endif
#endif  // BENCHMARK_GPU_UTILITIES_H

