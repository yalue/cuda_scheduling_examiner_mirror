// This file defines the interface each shared library must export. Library
// functions may print a single error message if they fail. Otherwise, they are
// not expected to output any text.
#ifndef LIBRARY_INTERFACE_H
#define LIBRARY_INTERFACE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This struct is used to pass arguments to the Initialize(..) function. Each
// benchmark isn't required to use every field, but they will always be set by
// the caller regardless. If the specified thread_count, block_count and
// data_size are not exactly supported by the benchmark, the Initialize
// function should modify the fields in this struct so that they reflect the
// actual dimensions used.
typedef struct {
  // The number of threads to be used in each block. This is a 3-dimensional
  // value, with the unused dimensions being set to 1
  int block_dim[3];
  // The number of blocks to create when launching a kernel. Like block_dim,
  // unused dimensions will be set to 1.
  int grid_dim[3];
  // The input size for each benchmark.
  uint64_t data_size;
  // Contains an optional user-specified string which is taken from the
  // additional_info field in the benchmark's user-supplied JSON config.
  char *additional_info;
  // The CUDA device ID to use.
  int cuda_device;
  // The stream priority to use for this instance's streams, if necessary.
  int stream_priority;
  // Which SMs are enabled for this instance's kernels
  uint64_t sm_mask;
} InitializationParameters;

// Holds times and metadata about a single kernel's execution in a benchmark
// iteration.
typedef struct {
  // The name of this kernel. May be NULL, in which case this field should be
  // ignored.
  const char *kernel_name;
  // The total number of threads per block used by this kernel.
  int thread_count;
  // The number of blocks run by this kernel.
  int block_count;
  // The amount of shared memory (in bytes) used by this kernel.
  int shared_memory;
  // This includes the CPU times, in seconds, immediately before and after
  // kernels were launched. The third entry is the CPU time after synchronizing
  // completes when the kernel is finished.
  double cuda_launch_times[3];
  // The start and end times for each individual block. Even-numbered positions
  // contain start times and odd positions end times, in nanoseconds. This will
  // contain block_count * 2 entries.
  uint64_t *block_times;
  // This contains the SM ID on which each block was run. This contains
  // block_count entries.
  uint32_t *block_smids;
} KernelTimes;

// Holds the measurements obtained during a single iteration of the benchmark,
// such as timing information. Any pointers in this struct must not be freed by
// the caller, and remain valid at least until another function in
// BenchmarkLibraryFunctions is called.
typedef struct {
  // The number of kernels run by the benchmark.
  int kernel_count;
  // This will be an array containing one kernel_times entry per kernel.
  KernelTimes *kernel_info;
  // This may be set to the host data buffer containing the results of GPU
  // processing (e.g. an output image). This should be set to NULL if the
  // benchmark doesn't use it. If non-NULL, the benchmark must ensure that this
  // pointer remains valid until any other function in
  // BenchmarkLibraryFunctions is called.
  uint64_t resulting_data_size;
  void *resulting_data;
} TimingInformation;

// Initializes the library/benchmark. This returns a pointer to user-defined
// data that will be passed to all subsequent functions. It is an error for
// this function to return NULL. This should allocate any CPU or GPU memory
// needed for the benchmark. All libraries *must* use a user-defined CUDA
// stream, so at the very least the user-defined data pointer should be a
// pointer to a struct containing a cudaStream_t used by the remaining
// functions.
typedef void* (*InitializeFunction)(InitializationParameters *params);

// Copies data into GPU memory. Receives a pointer to the user data returned
// by Initialize(). This must return 0 on error, or nonzero on success.
typedef int (*CopyInFunction)(void *data);

// All GPU kernels should be executed in this function. Like CopyIn, this
// receives a copy of the user data pointer returned by Initialize and returns
// 0 on error and nonzero on success.
typedef int (*ExecuteFunction)(void *data);

// This function should copy data out from the GPU and fill in the
// TimingInformation struct. Any pointers to buffers in TimingInformation must
// remain valid until any other function in BenchmarkLibraryFunctions is
// called.
typedef int (*CopyOutFunction)(void *data, TimingInformation *times);

// This will be called before exiting, and should be used to clean up any
// internal data.
typedef void (*CleanupFunction)(void *data);

// This function must return a pointer to a constant null-terminated string
// containing the library's name.
typedef const char* (*GetNameFunction)(void);

// The library fills in the members of this struct in the RegisterFunctions()
// function. The only members of this struct which must be set are initialize,
// get_name and copy_out. All other members, if NULL, will be ignored during
// execution.
typedef struct {
  InitializeFunction initialize;
  CopyInFunction copy_in;
  ExecuteFunction execute;
  CopyOutFunction copy_out;
  CleanupFunction cleanup;
  GetNameFunction get_name;
} BenchmarkLibraryFunctions;


// Every library must implement this function, which will be the first thing to
// be called after dlopen(...). Fills in the functions struct and returns 0 on
// error, nonzero on success.
extern int RegisterFunctions(BenchmarkLibraryFunctions *functions);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LIBRARY_INTERFACE_H
