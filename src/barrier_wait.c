// This file implemens the barrier synchronization library with the interface
// defined in barrier_wait.h.

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "barrier_wait.h"

// The internal state of the barrier, which will be held in shared memory.
typedef struct {
  // Maintains a count of the number of remaining processes.
  atomic_int processes_remaining;
  // This will be initialized to 0, and become nonzero if all processes have
  // reached the barrier.
  int done;
} InternalSharedBuffer;

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

int BarrierCreate(ProcessBarrier *b, int process_count) {
  InternalSharedBuffer *internal = NULL;
  internal = (InternalSharedBuffer *) AllocateSharedBuffer(sizeof(*internal));
  if (!internal) return 0;
  internal->done = 0;
  atomic_init(&(internal->processes_remaining), process_count);
  b->internal_buffer = internal;
  return 1;
}

void BarrierDestroy(ProcessBarrier *b) {
  FreeSharedBuffer(b->internal_buffer, sizeof(*b));
  b->internal_buffer = NULL;
}

int BarrierWait(ProcessBarrier *b) {
  volatile InternalSharedBuffer *internal =
    (InternalSharedBuffer *) b->internal_buffer;
  atomic_int value = atomic_fetch_sub(&(internal->processes_remaining), 1);
  if (value == 1) {
    internal->done = 1;
  }
  while (!internal->done) {
    continue;
  }
  return 1;
}

