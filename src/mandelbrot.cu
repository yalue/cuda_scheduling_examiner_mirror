// This file defines a CUDA mandelbrot set generator, with configurable
// parameters.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "library_interface.h"

// The maximum number of iterations used when determining if a point escapes
// the mandelbrot set.
#define MANDELBROT_ITERATIONS (10000)

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

// Holds the boundaries and sizes of the fractal, in both pixels and numbers
typedef struct {
  // The width and height of the image in pixels.
  int w;
  int h;
  // The boundaries of the fractal.
  double min_real;
  double min_imag;
  double max_real;
  double max_imag;
  // The distance between pixels in the real and imaginary axes.
  double delta_real;
  double delta_imag;
} FractalDimensions;

// Holds the state of a single instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  // This will be 0 if the stream hasn't been created yet. This value exists
  // in order to avoid calling cudaStreamDestroy when the stream hasn't been
  // created.
  int stream_created;
  // Holds the host and device copies of the mandelbrot set. Each value in the
  // buffers will be either 0 (in the set) or 1 (escaped).
  uint8_t *host_points;
  uint8_t *device_points;
  // The maximum number of iterations used when drawing the set.
  uint32_t max_iterations;
  // The dimensions of the complex plane used when drawing the mandelbrot set.
  FractalDimensions dimensions;
  // Holds 2 64-bit elements: the start and stop times of the kernel, as
  // measured on the device.
  uint64_t *device_kernel_times;
  uint64_t host_kernel_times[2];
  // Holds a start and stop time for each block, as measured on the device.
  uint64_t *device_block_times;
  uint64_t *host_block_times;
  // The grid dimensions for the CUDA program, set during Initialize to a value
  // based on the thread_count specified by the caller. The caller-specified
  // block_count is ignored--instead the number of needed blocks is decided by
  // the data_size field, which determines the size of the image.
  int block_count;
  int thread_count;
} ThreadInformation;

// Implements the Cleanup() function required by the library interface.
static void Cleanup(void *data) {
  ThreadInformation *info = (ThreadInformation *) data;
  if (info->host_points) free(info->host_points);
  if (info->device_points) cudaFree(info->device_points);
  if (info->device_block_times) cudaFree(info->device_block_times);
  if (info->host_block_times) free(info->host_block_times);
  if (info->device_kernel_times) cudaFree(info->device_kernel_times);
  if (info->stream_created) {
    // Call CheckCUDAError here to print a message, even though we won't check
    // the return value.
    CheckCUDAError(cudaStreamDestroy(info->stream));
  }
  memset(info, 0, sizeof(*info));
  free(info);
}

// Allocates GPU and CPU memory. Returns 0 on error, 1 otherwise.
static int AllocateMemory(ThreadInformation *info) {
  uint64_t buffer_size = info->dimensions.w * info->dimensions.h;
  uint64_t block_times_size = info->block_count * sizeof(uint64_t) * 2;
  if (!CheckCUDAError(cudaMalloc(&(info->device_points), buffer_size))) {
    return 0;
  }
  info->host_points = (uint8_t *) malloc(buffer_size);
  if (!info->host_points) return 0;
  if (!CheckCUDAError(cudaMalloc(&(info->device_kernel_times),
    sizeof(info->host_kernel_times)))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(info->device_block_times),
    block_times_size))) {
    return 0;
  }
  info->host_block_times = (uint64_t *) malloc(block_times_size);
  if (!info->host_block_times) return 0;
  return 1;
}

// Implements the Initialize() function required by the library interface.
static void* Initialize(InitializationParameters *params) {
  ThreadInformation *info = NULL;
  FractalDimensions *dimensions = NULL;
  info = (ThreadInformation *) malloc(sizeof(*info));
  if (!info) {
    printf("Failed allocating library state variables.\n");
    return NULL;
  }
  memset(info, 0, sizeof(*info));
  // Set the device if necessary
  if (params->cuda_device != USE_DEFAULT_DEVICE) {
    if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  }
  // Round the thread count up to a value evenly divisble by 32.
  if ((params->thread_count % WARP_SIZE) != 0) {
    params->thread_count += WARP_SIZE - (params->thread_count % WARP_SIZE);
  }
  info->thread_count = params->thread_count;
  // Fill in the dimensions and parameters of the complex plane region we'll
  // draw.
  dimensions = &(info->dimensions);
  dimensions->w = (int) sqrt(params->data_size);
  dimensions->h = dimensions->w;
  dimensions->min_real = -2.0;
  dimensions->max_real = 2.0;
  dimensions->min_imag = -2.0;
  dimensions->max_imag = 2.0;
  dimensions->delta_real = 4.0 / dimensions->w;
  dimensions->delta_imag = 4.0 / dimensions->h;
  // Set the block count based on thread_count and the image dimensions.
  info->block_count = (dimensions->w * dimensions->h) / params->thread_count;
  // In case the image isn't evenly divisible by the thread_count...
  info->block_count++;
  // Allocate both host and device memory.
  if (!AllocateMemory(info)) {
    Cleanup(info);
    return NULL;
  }
  if (!CheckCUDAError(cudaStreamCreate(&(info->stream)))) {
    Cleanup(info);
    return NULL;
  }
  info->stream_created = 1;
  return info;
}

// This function only exists because the idiom for detecting the kernel start
// time requires initializing the start kernel time value on the device to the
// maximum possible 64-bit value.
static int CopyIn(void *data) {
  ThreadInformation *info = (ThreadInformation *) data;
  // Set the start kernel time to the maximum value and the end kernel time to
  // 0.
  memset(info->host_kernel_times, 0, sizeof(info->host_kernel_times));
  info->host_kernel_times[0]--;
  if (!CheckCUDAError(cudaMemcpyAsync(info->device_kernel_times,
    info->host_kernel_times, sizeof(info->host_kernel_times),
    cudaMemcpyHostToDevice, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(info->stream))) return 0;
  return 1;
}

// Returns the value of CUDA's global nanosecond timer.
static __device__ __inline__ uint64_t GlobalTimer64(void) {
  uint64_t to_return;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(to_return));
  return to_return;
}

// A basic mandelbrot set calculator which sets each element in data to 1 if
// the point escapes within the given number of iterations.
static __global__ void BasicMandelbrot(uint8_t *data, int iterations,
    FractalDimensions dimensions, uint64_t *kernel_times,
    uint64_t *block_times) {
  uint64_t start_time = GlobalTimer64();
  // Record kernel and block start times.
  if (kernel_times[0] > start_time) kernel_times[0] = start_time;
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
  }
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = index / dimensions.w;
  int col = index % dimensions.w;
  // This may cause some threads to diverge on the last block only
  if (row >= dimensions.h) {
    kernel_times[1] = GlobalTimer64();
    return;
  }
  __syncthreads();
  double start_real = dimensions.min_real + dimensions.delta_real * col;
  double start_imag = dimensions.min_imag + dimensions.delta_imag * row;
  double current_real = start_real;
  double current_imag = start_imag;
  double magnitude_squared = (start_real * start_real) + (start_imag *
    start_imag);
  uint8_t escaped = 0;
  double tmp;
  int i;
  for (i = 0; i < iterations; i++) {
    if (magnitude_squared < 4) {
      tmp = (current_real * current_real) - (current_imag * current_imag) +
        start_real;
      current_imag = 2 * current_imag * current_real + start_imag;
      current_real = tmp;
      magnitude_squared = (current_real * current_real) + (current_imag *
        current_imag);
    } else {
      escaped = 1;
    }
  }
  data[row * dimensions.w + col] = escaped;
  __syncthreads();
  // Record the block end time.
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
  kernel_times[1] = GlobalTimer64();
}

static int Execute(void *data) {
  ThreadInformation *info = (ThreadInformation *) data;
  BasicMandelbrot<<<info->block_count, info->thread_count, 0, info->stream>>>(
    info->device_points, info->max_iterations, info->dimensions,
    info->device_kernel_times, info->device_block_times);
  if (!CheckCUDAError(cudaStreamSynchronize(info->stream))) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  ThreadInformation *info = (ThreadInformation *) data;
  uint64_t block_times_count = info->block_count * 2;
  uint64_t points_size = info->dimensions.w * info->dimensions.h;
  memset(times, 0, sizeof(*times));
  if (!CheckCUDAError(cudaMemcpyAsync(info->host_kernel_times,
    info->device_kernel_times, sizeof(info->host_kernel_times),
    cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(info->host_block_times,
    info->device_block_times, block_times_count * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(info->host_points, info->device_points,
    points_size, cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaStreamSynchronize(info->stream))) return 0;
  times->kernel_times_count = 2;
  times->kernel_times = info->host_kernel_times;
  times->block_times_count = block_times_count;
  times->block_times = info->host_block_times;
  times->resulting_data_size = points_size;
  times->resulting_data = info->host_points;
  return 1;
}

static const char* GetName(void) {
  return "Mandelbrot Set";
}

void RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  functions->cleanup = Cleanup;
  functions->get_name = GetName;
}
