// This file defines a CPU-only benchmark which performs a single-threaded
// random traversal over a buffer of CPU memory. It doesn't use the GPU at
// all. The input_size parameter determines the number of bytes in the buffer
// to traverse. The buffer must be at least 4 bytes.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "library_interface.h"

// The default number of memory reads to perform in each iteration.
#define DEFAULT_MEMORY_ACCESS_COUNT (1000 * 1000)

// Holds local state for one instances of this benchmark.
typedef struct {
  // The buffer to traverse.
  uint64_t *buffer;
  // The number of 32-bit elements in the buffer.
  uint64_t buffer_length;
  // The number of memory reads to perform during the execute phase.
  uint64_t memory_access_count;
  // The sum from the random walk. Probably useless, but will prevent
  // optimization from removing our loop.
  uint64_t accumulator;
} TaskState;

static void Cleanup(void *data) {
  TaskState *state = (TaskState *) data;
  if (state->buffer) free(state->buffer);
  state->buffer = NULL;
  state->buffer_length = 0;
  free(state);
}

// Returns a single random 64-bit value.
static uint64_t Random64(void) {
  int i;
  uint64_t to_return = 0;
  // Get a random number in 16-bit chunks
  for (i = 0; i < 4; i++) {
    to_return = to_return << 16;
    to_return |= rand() & 0xffff;
  }
  return to_return;
}

// Returns a random 64-bit integer in the range [base, limit)
static uint64_t RandomRange(uint64_t base, uint64_t limit) {
  if (limit <= base) return base;
  return (Random64() % (limit - base)) + base;
}

// Shuffles an array of 32-bit values.
static void ShuffleArray(uint64_t *buffer, uint64_t element_count) {
  uint64_t tmp, i, dst;
  for (i = 0; i < element_count; i++) {
    dst = RandomRange(i, element_count);
    tmp = buffer[i];
    buffer[i] = buffer[dst];
    buffer[dst] = tmp;
  }
}

static void* Initialize(InitializationParameters *params) {
  uint64_t buffer_length, i;
  TaskState *state = NULL;
  state = (TaskState *) malloc(sizeof(*state));
  if (!state) return NULL;
  memset(state, 0, sizeof(*state));
  buffer_length = params->data_size / sizeof(uint64_t);
  // The buffer must contain at least one element.
  if (buffer_length == 0) {
    free(state);
    return NULL;
  }
  state->buffer = (uint64_t *) malloc(buffer_length * sizeof(uint64_t));
  if (!state->buffer) {
    free(state);
    return NULL;
  }
  // Initialize the buffer with each entry containing its own index.
  for (i = 0; i < buffer_length; i++) {
    state->buffer[i] = i;
  }
  // Produce a random walk by shuffling the buffer.
  ShuffleArray(state->buffer, buffer_length);
  state->buffer_length = buffer_length;
  state->memory_access_count = DEFAULT_MEMORY_ACCESS_COUNT;
  return state;
}

static int CopyIn(void *data) {
  return 1;
}

static int Execute(void *data) {
  uint64_t i, walk_index, accumulator;
  TaskState *state = (TaskState *) data;
  walk_index = 0;
  accumulator = 0;
  for (i = 0; i < state->memory_access_count; i++) {
    walk_index = state->buffer[walk_index];
    accumulator += walk_index;
  }
  state->accumulator = accumulator;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  TaskState *state = (TaskState *) data;
  times->kernel_count = 0;
  times->kernel_info = NULL;
  times->resulting_data_size = sizeof(state->accumulator);
  times->resulting_data = &(state->accumulator);
  return 1;
}

static const char* GetName(void) {
  return "CPU random walk";
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
