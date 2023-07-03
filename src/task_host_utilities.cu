// This file contains the implementation of the functions defined in
// task_host_utilities.h--used by task_host_utilities.c to work with the GPU.
#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "task_host_utilities.h"
#include "library_interface.h"

// The number of GPU nanoseconds to spin for GetGPUTimerScale. Increasing this
// will both increase the accuracy and the time the function takes to return.
#define TIMER_SPIN_DURATION (2ull * 1000 * 1000 * 1000)

// This macro takes a cudaError_t value. It prints an error message and returns
// 0 if the cudaError_t isn't cudaSuccess. Otherwise, it returns nonzero.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Prints an error message and returns 0 if the given CUDA result is an error.
static int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
    const char *file, int line) {
  if (result == cudaSuccess) return 1;
  printf("CUDA error %d: %s. In %s, line %d (%s)\n", (int) result,
    cudaGetErrorString(result), file, line, fn);
  return 0;
}

static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

// GlobalTimer64: Get 64-bit counter of current time on GPU
// ***This is duplicated in benchmark_gpu_utilities.h***
#if __CUDA_ARCH__ >= 300 // Kepler+
// Returns the value of CUDA's global nanosecond timer.
// Starting with sm_30, `globaltimer64` was added
// Tuned for sm_5X architectures, but tested to still be efficient on sm_7X
static __device__ inline uint64_t GlobalTimer64(void) {
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
static __device__ inline uint64_t GlobalTimer64(void) {
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

// A simple kernel which writes the value of the globaltimer64 register to a
// location in device memory.
static __global__ void GetTime(uint64_t *time, volatile uint32_t *ready_barrier,
    volatile uint32_t *start_barrier, volatile uint32_t *end_barrier) {
  *ready_barrier = 1;
  while (!*start_barrier)
    continue;
  *time = GlobalTimer64();
  *end_barrier = 1;
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
// nanosecond counter. Sets times to 0 on error.
static void InternalReadGPUNanoseconds(int cuda_device, double *cpu_time,
    uint64_t *gpu_time) {
  uint64_t *device_time = NULL;
  volatile uint32_t *gpu_start_barrier, *start_barrier = NULL;
  volatile uint32_t *gpu_end_barrier, *end_barrier = NULL;
  volatile uint32_t *gpu_ready_barrier, *ready_barrier = NULL;
  volatile double cpu_start, cpu_end;
  double max_error;
  // Times need to be zero in case any of the following logic fails
  *cpu_time = 0;
  *gpu_time = 0;
  if (!CheckCUDAError(cudaSetDevice(cuda_device))) return;
  if (!CheckCUDAError(cudaMalloc(&device_time, sizeof(*device_time)))) return;
  if (!CheckCUDAError(cudaHostAlloc(&start_barrier, sizeof(*start_barrier),
    cudaHostAllocMapped))) goto out;
  if (!CheckCUDAError(cudaHostAlloc(&end_barrier, sizeof(*end_barrier),
    cudaHostAllocMapped))) goto out;
  if (!CheckCUDAError(cudaHostAlloc(&ready_barrier, sizeof(*ready_barrier),
    cudaHostAllocMapped))) goto out;
  // Setup device pointers for all the barriers
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_start_barrier,
    (uint32_t*)start_barrier, 0))) goto out;
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_end_barrier,
    (uint32_t*)end_barrier, 0))) goto out;
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_ready_barrier,
    (uint32_t*)ready_barrier, 0))) goto out;
  /* Clock Synchronization Flow

                            Start Barrier--+
                                           |
    CPU: Launch Kernel......^-->CPU TS 1-->|............^-->CPU TS 2-->Done
                |           |              |            |
    GPU:        +---------->|..............v-->GPU TS-->|-->Terminate
                            |                           |
             Ready Barrier--+              End Barrier--+

    GPU TS ~= CPU TS 1 + (CPU TS 2 - CPU TS 1) / 2.0

    If it takes about as long for a CPU memory write to DRAM to be visible to
    the GPU as it takes for a GPU memory write to DRAM to be visible to the
    CPU.

    If not, the worst possible error is if one of the above legs is instant,
    and the other is extremely slow, in which case the time estimate is off by
    as much as half the CPU interval [(CPU TS 2 - CPU TS 1) / 2.0]. (Depending
    on the platform's CPU cache configuration, an inbalance may be expected.)

    The typical error should be about the difference in the amount of time it
    takes to read the CPU TS counter and the GPU TS counter. On recent
    platforms, the difference shouldn't be more than double-digit nanoseconds.
  */
  // Run the kernel a first time to warm up the GPU.
  GetTime<<<1, 1>>>(device_time, gpu_ready_barrier, gpu_start_barrier,
    gpu_end_barrier);
  *start_barrier = 1;
  if (!CheckCUDAError(cudaDeviceSynchronize())) goto out;
  // Now run the actual time-checking kernel.
  *start_barrier = 0;
  *end_barrier = 0;
  *ready_barrier = 0;
  GetTime<<<1, 1>>>(device_time, gpu_ready_barrier, gpu_start_barrier,
    gpu_end_barrier);
  // Wait for kernel to initialize
  while (!*ready_barrier)
    continue;
  // Immediately record CPU time and tell GPU kernel to record time
  cpu_start = CurrentSeconds();
  *start_barrier = 1;
  // Wait for kernel to finish recording time, and immediately record CPU time again
  while (!*end_barrier)
    continue;
  cpu_end = CurrentSeconds();
  // GPU clock should have been stored about half-way between the CPU reads
  *cpu_time = (cpu_end - cpu_start) / 2.0 + cpu_start;
  if (!CheckCUDAError(cudaMemcpy(gpu_time, device_time, sizeof(device_time),
    cudaMemcpyDeviceToHost))) goto out;
  max_error = (cpu_end - cpu_start) / 2.0;
  fprintf(stderr, "Time synchronized to a maximum error of +/- %f us.\n",
    max_error * (1000.0 * 1000.0));
out:
  cudaFree(device_time);
  cudaFree((void*)start_barrier);
  cudaFree((void*)end_barrier);
  cudaFree((void*)ready_barrier);
}

int GetHostDeviceTimeOffset(int cuda_device, double *host_seconds,
  uint64_t *gpu_nanoseconds) {
  uint64_t *shared_gpu_time = NULL;
  double *shared_cpu_time = NULL;
  int status;
  pid_t pid = -1;
  shared_gpu_time = (uint64_t *) AllocateSharedBuffer(
    sizeof(*shared_gpu_time));
  if (!shared_gpu_time) {
    printf("Failed allocating shared buffer for IPC.\n");
    return 0;
  }
  shared_cpu_time = (double *) AllocateSharedBuffer(sizeof(*shared_cpu_time));
  if (!shared_cpu_time) {
    printf("Failed allocating shared CPU time buffer for IPC.\n");
    FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
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
    InternalReadGPUNanoseconds(cuda_device, shared_cpu_time, shared_gpu_time);
    exit(0);
  }
  // The parent will wait for the child to finish, then return the value
  // written to the shared buffer.
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(shared_cpu_time, sizeof(*shared_cpu_time));
    FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
    return 0;
  }
  *host_seconds = *shared_cpu_time;
  *gpu_nanoseconds = *shared_gpu_time;
  FreeSharedBuffer(shared_cpu_time, sizeof(*shared_cpu_time));
  FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  return 1;
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

#if __CUDA_ARCH__ >= 300 // Kepler+
static __global__ void TimerSpin(uint64_t ns_to_spin) {
  uint64_t start_time = GlobalTimer64();
  while ((GlobalTimer64() - start_time) < ns_to_spin) {
    continue;
  }
}
#elif __CUDA_ARCH__ < 200 // Tesla
// On sm_13, we /only/ have `clock` (32 bits)
static __device__ inline uint32_t Clock32(void) {
  uint32_t lo_bits;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(lo_bits));
  return lo_bits;
}

// 'clock' can easily roll over, so handle that for ancient architectures
static __global__ void TimerSpin(uint64_t ns_to_spin) {
  uint64_t total_time = 0;
  uint32_t last_time = Clock32();
  while (total_time < ns_to_spin) {
    uint32_t time = Clock32();
    if (time < last_time) {
      // Rollover. Compensate...
      total_time += time; // rollover to now
      total_time += UINT_MAX - last_time; // last to rollover
    } else {
      // Step counter
      total_time += time - last_time;
    }
    last_time = time;
  }
}
#else
// Could use clock64 for sm_2x (Fermi), but that's untested
#error Fermi-based GPUs (sm_2x) are unsupported!
#endif

// This function is intended to be run in a child process. Returns -1 on error.
static double InternalGetGPUTimerScale(int cuda_device) {
  struct timespec start, end;
  uint64_t nanoseconds_elapsed;
  if (!CheckCUDAError(cudaSetDevice(cuda_device))) return -1;
  // Run the kernel once to warm up the GPU.
  TimerSpin<<<1, 1>>>(1000);
  if (!CheckCUDAError(cudaDeviceSynchronize())) return -1;
  // After warming up, do the actual timing.
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) != 0) {
    printf("Failed getting start time.\n");
    return -1;
  }
  TimerSpin<<<1, 1>>>(TIMER_SPIN_DURATION);
  if (!CheckCUDAError(cudaDeviceSynchronize())) return -1;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &end) != 0) {
    printf("Failed getting end time.\n");
    return -1;
  }
  nanoseconds_elapsed = end.tv_sec * 1e9 + end.tv_nsec;
  nanoseconds_elapsed -= start.tv_sec * 1e9 + start.tv_nsec;
  return ((double) nanoseconds_elapsed) / ((double) TIMER_SPIN_DURATION);
}

double GetGPUTimerScale(int cuda_device) {
  double to_return;
  double *scale = NULL;
  int status;
  pid_t pid;
  scale = (double *) AllocateSharedBuffer(sizeof(*scale));
  if (!scale) {
    printf("Failed allocating space to hold the GPU time scale.\n");
    return -1;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process.\n");
    FreeSharedBuffer(scale, sizeof(*scale));
    return -1;
  }
  if (pid == 0) {
    // Access the GPU with the child process only.
    *scale = InternalGetGPUTimerScale(cuda_device);
    exit(0);
  }
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(scale, sizeof(*scale));
    return -1;
  }
  to_return = *scale;
  FreeSharedBuffer(scale, sizeof(*scale));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return -1;
  }
  return to_return;
}
