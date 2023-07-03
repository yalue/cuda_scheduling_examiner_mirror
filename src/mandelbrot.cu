// This file defines a CUDA mandelbrot set generator, with configurable
// parameters.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "benchmark_gpu_utilities.h"
#include "library_interface.h"

// The default number of iterations used when determining if a point escapes
// the mandelbrot set. Optionally, the number of iterations can be specified
// using the additional_info field.
#define DEFAULT_MAX_ITERATIONS (1000)

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
  uint64_t max_iterations;
  // The dimensions of the complex plane used when drawing the mandelbrot set.
  FractalDimensions dimensions;
  // Holds a start and stop time for each block, as measured on the device.
  uint64_t *device_block_times;
  // Holds the ID of the SM for each block, checked once the kernel executes.
  uint32_t *device_block_smids;
  // The grid dimensions for the CUDA program, set during Initialize to a value
  // based on the thread_count specified by the caller. The caller-specified
  // block_count is ignored--instead the number of needed blocks is decided by
  // the data_size field, which determines the size of the image.
  int block_count;
  int thread_count;
  // Holds host-side times that are shared with the calling process.
  KernelTimes mandelbrot_kernel_times;
} TaskState;

// Implements the Cleanup() function required by the library interface.
static void Cleanup(void *data) {
  TaskState *info = (TaskState *) data;
  KernelTimes *host_times = &info->mandelbrot_kernel_times;
  // Device memory
  if (info->device_points) cudaFree(info->device_points);
  if (info->device_block_times) cudaFree(info->device_block_times);
  if (info->device_block_smids) cudaFree(info->device_block_smids);
  // Host memory
  if (info->host_points) cudaFreeHost(info->host_points);
  if (host_times->block_times) cudaFreeHost(host_times->block_times);
  if (host_times->block_smids) cudaFreeHost(host_times->block_smids);
  if (info->stream_created) {
    // Call CheckCUDAError here to print a message, even though we won't check
    // the return value.
    CheckCUDAError(cudaStreamDestroy(info->stream));
  }
  memset(info, 0, sizeof(*info));
  free(info);
}

// Allocates GPU and CPU memory. Returns 0 on error, 1 otherwise.
static int AllocateMemory(TaskState *info) {
  uint64_t buffer_size = info->dimensions.w * info->dimensions.h;
  uint64_t block_times_size = info->block_count * sizeof(uint64_t) * 2;
  uint64_t block_smids_size = info->block_count * sizeof(uint32_t);
  KernelTimes *mandelbrot_kernel_times = &info->mandelbrot_kernel_times;
  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&info->device_points, buffer_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&info->device_block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&info->device_block_smids,
    block_smids_size))) {
    return 0;
  }
  // Allocate host memory
  if (!CheckCUDAError(cudaMallocHost(&info->host_points, buffer_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&mandelbrot_kernel_times->block_times,
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&mandelbrot_kernel_times->block_smids,
    block_smids_size))) {
    return 0;
  }
  return 1;
}

// Checks the additional_info argument to see if it's non-empty and non-NULL,
// in which case it can override the default max iterations if it's parsed into
// a valid base-10 integer.
static int SetMaxIterations(const char *arg, TaskState *info) {
  int64_t parsed_value;
  if (!arg || (strlen(arg) == 0)) {
    info->max_iterations = DEFAULT_MAX_ITERATIONS;
    return 1;
  }
  char *end = NULL;
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid max iterations: %s\n", arg);
    return 0;
  }
  info->max_iterations = (uint64_t) parsed_value;
  return 1;
}

// Implements the Initialize() function required by the library interface.
static void* Initialize(InitializationParameters *params) {
  TaskState *info = NULL;
  FractalDimensions *dimensions = NULL;
  info = (TaskState *) calloc(1, sizeof(*info));
  if (!info) {
    printf("Failed allocating library state variables.\n");
    return NULL;
  }
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  if (!GetSingleBlockDimension(params, &info->thread_count)) {
    Cleanup(info);
    return NULL;
  }
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
  info->block_count = (dimensions->w * dimensions->h) / info->thread_count;
  // In case the image isn't evenly divisible by the thread_count...
  if (((dimensions->w * dimensions->h) % info->thread_count) != 0) {
    info->block_count++;
  }
  if (!SetMaxIterations(params->additional_info, info)) {
    Cleanup(info);
    return NULL;
  }
  // Allocate both host and device memory.
  if (!AllocateMemory(info)) {
    Cleanup(info);
    return NULL;
  }
  if (!CheckCUDAError(CreateCUDAStreamWithPriorityAndMask(
    params->stream_priority, params->sm_mask, &(info->stream)))) {
    Cleanup(info);
    return NULL;
  }
  info->stream_created = 1;
  return info;
}

// Nothing needs to be copied in, so this function does nothing.
static int CopyIn(void *data) {
  return 1;
}

// A basic mandelbrot set calculator which sets each element in data to 1 if
// the point escapes within the given number of iterations.
static __global__ void BasicMandelbrot(uint8_t *data, uint64_t iterations,
    FractalDimensions dimensions, uint64_t *block_times,
    uint32_t *block_smids) {
  uint64_t start_time = GlobalTimer64();
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = index / dimensions.w;
  int col = index % dimensions.w;
  // This may cause some threads to diverge on the last block only
  if (row >= dimensions.h) {
    if (threadIdx.x == 0) {
      block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
    }
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
  uint64_t i;
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
}

static int Execute(void *data) {
  TaskState *info = (TaskState *) data;
  info->mandelbrot_kernel_times.cuda_launch_times[0] = CurrentSeconds();
  BasicMandelbrot<<<info->block_count, info->thread_count, 0, info->stream>>>(
    info->device_points, info->max_iterations, info->dimensions,
    info->device_block_times, info->device_block_smids);
  info->mandelbrot_kernel_times.cuda_launch_times[1] = CurrentSeconds();
  if (!CheckCUDAError(cudaStreamSynchronize(info->stream))) return 0;
  info->mandelbrot_kernel_times.cuda_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  TaskState *info = (TaskState *) data;
  KernelTimes *host_times = &info->mandelbrot_kernel_times;
  uint64_t block_times_count = info->block_count * 2;
  uint64_t block_smids_count = info->block_count;
  uint64_t points_size = info->dimensions.w * info->dimensions.h;
  memset(times, 0, sizeof(*times));
  host_times->block_count = info->block_count;
  host_times->thread_count = info->thread_count;
  host_times->kernel_name = "BasicMandelbrot";
  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_times,
    info->device_block_times, block_times_count * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_smids,
    info->device_block_smids, block_smids_count * sizeof(uint32_t),
    cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMemcpyAsync(info->host_points, info->device_points,
    points_size, cudaMemcpyDeviceToHost, info->stream))) {
    return 0;
  }
  times->kernel_count = 1;
  times->kernel_info = host_times;
  times->resulting_data_size = points_size;
  times->resulting_data = info->host_points;
  if (!CheckCUDAError(cudaStreamSynchronize(info->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Mandelbrot Set";
}

int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  functions->cleanup = Cleanup;
  functions->get_name = GetName;
  return 1;
}
