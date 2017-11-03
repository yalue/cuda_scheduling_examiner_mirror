// This file defines types and macros used by the stream_action benchmark. See
// the comment at the top of stream_action.cu for more detail about what the
// benchmark does.
#ifndef STREAM_ACTION_H
#define STREAM_ACTION_H
#include <cuda_runtime.h>
#include <stdint.h>
#include "benchmark_gpu_utilities.h"

// This macro is used to create functions that statically use predefined
// amounts of shared memory. This is used by the GENERATE_KERNEL macro.
#define GENERATE_SHARED_MEMORY_FUNCTION(amount) \
  static __device__ uint32_t UseSharedMemory_##amount(void) { \
    __shared__ uint32_t shared_array[(amount)]; \
    uint32_t elements_per_thread, i; \
    elements_per_thread = (amount) / blockDim.x; \
    for (i = 0; i < elements_per_thread; i++) { \
      shared_array[threadIdx.x * elements_per_thread + i] = threadIdx.x; \
    } \
    return shared_array[threadIdx.x * elements_per_thread]; \
  }

// Generates kernels that use the given amount of shared memory. Kernels have
// names like SharedMemGPUSpin_<amount>, and take the following parameters:
// (int counter, uint64_t duration, uint64_t *block_times,
// uint32_t *block_smids, uint64_t *junk). If the "counter" parameter is
// nonzero, then a constant amount of computation will be carried out rather
// than waiting for a constant amount of time. The "junk" parameter is used to
// prevent optimizations, and must be NULL. Otherwise, this kernel operates
// similarly to the simpler GPUSpin kernel in stream_action.cu. This WILL NOT
// work for 0 bytes of shared memory--that's what the plain GPUSpin in
// stream_action.cu is for.
#define GENERATE_SPIN_KERNEL(amount) \
  /* Produce a function that uses shared memory */ \
  GENERATE_SHARED_MEMORY_FUNCTION(amount) \
  static __global__ void SharedMemGPUSpin_##amount(int use_counter, \
    uint64_t duration, uint64_t *block_times, uint32_t *block_smids, \
    uint64_t *junk) { \
    uint32_t shared_mem_res; \
    uint64_t i, accumulator; \
    uint64_t start_time = GlobalTimer64(); \
    if (threadIdx.x == 0) { \
      block_times[blockIdx.x * 2] = start_time; \
      block_smids[blockIdx.x] = GetSMID(); \
    } \
    __syncthreads(); \
    /* shared_mem_res is our thread index */ \
    shared_mem_res = UseSharedMemory_##amount(); \
    if (use_counter) { \
      for (i = 0; i < duration; i++) { \
        accumulator += i; \
      } \
    } else { \
      while ((GlobalTimer64() - start_time) < duration) { \
        continue; \
      } \
    } \
    if (junk) *junk = accumulator; \
    if (shared_mem_res == 0) { \
      block_times[blockIdx.x * 2 + 1] = GlobalTimer64(); \
    } \
  }

// This holds parameters for the kernel action.
typedef struct {
  // The grid dimensions for this kernel.
  int block_count;
  int thread_count;
  // The amount of shared memory used by this kernel.
  int shared_memory_count;
  // If this is nonzero, the counter_spin kernel will be used, which performs
  // a constant amount of busywork computations. If this is zero, the
  // timer_spin kernel will be used instead, which waits until a certain number
  // of nanoseconds have elapsed.
  int use_counter_spin;
  // The number of either spin iterations or nanoseconds this kernel runs for
  // (depending on whether it is a timer spin or counter spin kernel).
  uint64_t duration;
  // Hold the times needed for a CUDA kernel.
  uint64_t *device_block_times;
  uint64_t *host_block_times;
  uint32_t *device_smids;
  uint32_t *host_smids;
} KernelParameters;

// This holds parameters for the cudaMalloc action.
typedef struct {
  // This is the number of bytes to allocate.
  uint64_t size;
  // If nonzero, call cudaMallocHost rather than cudaMalloc.
  int allocate_host_memory;
} MallocParameters;

// This holds parameters for the cudaFree action.
typedef struct {
  // If nonzero, call cudaFreeHost rather than cudaFree.
  int free_host_memory;
} FreeParameters;

// This holds parameters for the cudaMemset action, which sets bytes to a
// random 8-bit value.
typedef struct {
  // If nonzero, then cudaMemset will be called (associated with no stream),
  // rather than cudaMemsetAsync, which will use the task's specified stream.
  int synchronous;
  // This contains the number of bytes to set.
  uint64_t size;
} MemsetParameters;

// This holds parameters for the cudaMemcpy action, which copies data between
// host and device, or two device buffers.
typedef struct {
  // One of the cudaMemcpyKind values. However, values 0 (host - host) and 4
  // (unspecified) are not supported.
  cudaMemcpyKind direction;
  // If nonzero, then cudaMemcpy will be used. If 0, then cudaMemcpyAsync is
  // used, associated with the task's stream.
  int synchronous;
  // The number of bytes to copy.
  uint64_t size;
} MemcpyParameters;

// This holds parameters for the synchronize action.
typedef struct {
  // If this is nonzero, then cudaDeviceSynchronize will be called. Otherwise,
  // cudaStreamSynchronize is called, associated with the task's stream.
  int sync_device;
} SyncParameters;

// This is used as a tag to identify the parameters and behavior to carry out
// for each action supported by the benchmark.
typedef enum {
  ACTION_UNINITIALIZED = 0,
  ACTION_KERNEL,
  ACTION_MALLOC,
  ACTION_FREE,
  ACTION_MEMSET,
  ACTION_MEMCPY,
  ACTION_SYNC,
} ActionType;

// This defines the behavior and parameters for all potential actions.
typedef struct {
  // The number of seconds to sleep after the current action's completion,
  // before launching this one.
  double delay;
  // The label (typically a kernel name) to give this action.
  char *label;
  ActionType type;
  union {
    KernelParameters kernel;
    MallocParameters malloc;
    FreeParameters free;
    MemsetParameters memset;
    MemcpyParameters memcpy;
    SyncParameters sync;
  } parameters;
} ActionConfig;

// Holds local information for each instantiation of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // The CUDA stream with which copy_out operations will be associated. May
  // differ from the regular stream, because this will never be the NULL
  // stream.
  cudaStream_t copy_out_stream;
  // This will be set to 1 if the stream was created and must be closed during
  // cleanup (it can remain 0 if the NULL stream is used).
  int stream_created;
  // The number of actions to perform per execution.
  int action_count;
  // The list of actions to perform.
  ActionConfig *actions;
  // The number of actions which are kernel launches.
  int kernel_count;
  // Information to provide to the host process about block start and end times
  // for each kernel action.
  KernelTimes *kernel_times;
  // A buffer of host memory for copies and memsets. May be NULL if not needed.
  // Is guaranteed to be the size of the largest copy or memset needed by any
  // action.
  uint8_t *host_copy_buffer;
  // A buffer of device memory for copies and memsets. May be NULL if not
  // needed. This is guaranteed to be the size of the largest copy or memset
  // needed by any action.
  uint8_t *device_copy_buffer;
  // This will be a secondary device buffer, but will only be allocated if a
  // device-to-device memory copy is used.
  uint8_t *device_secondary_buffer;
  // This is a stack of pointers to device memory allocated by cudaMalloc
  // actions.
  uint8_t **device_memory_allocations;
  // Holds the number of pointers in the device_memory_allocations list. This
  // increases with each cudaMalloc action and decreases with each cudaFree.
  int device_memory_allocation_count;
  // This is a stack of pointers to host memory allocated by cudaMallocHost.
  // It works in the same way as device_memory_allocations.
  uint8_t **host_memory_allocations;
  // This is analagous to device_memory_allocation_count, but for host memory
  // allocations.
  int host_memory_allocation_count;
} TaskState;

#endif  // STREAM_ACTION_H
