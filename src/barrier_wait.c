// This file implemens the barrier synchronization library with the interface
// defined in barrier_wait.h.

#include <stdatomic.h>
#include <stdint.h>
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
  // Used in combination with local_sense to prevent multiple barriers from
  // interfering with each other.
  int sense;
  // This will be the number of processes which are sharing the barrier.
  int process_count;
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
  internal->process_count = process_count;
  atomic_init(&(internal->processes_remaining), process_count);
  internal->sense = 0;
  b->internal_buffer = internal;
  return 1;
}

void BarrierDestroy(ProcessBarrier *b) {
  FreeSharedBuffer(b->internal_buffer, sizeof(InternalSharedBuffer));
  b->internal_buffer = NULL;
}

int BarrierWait(ProcessBarrier *b, int *local_sense) {
  volatile InternalSharedBuffer *internal =
    (InternalSharedBuffer *) b->internal_buffer;
  *local_sense = !(*local_sense);
  atomic_int value = atomic_fetch_sub(&(internal->processes_remaining), 1);
  if (value == 1) {
    // We were the last process to call atomic_fetch_sub, so reset the counter
    // and release the other processes by writing internal->sense.
    atomic_store(&(internal->processes_remaining), internal->process_count);
    internal->sense = *local_sense;
    return 1;
  }
  while (internal->sense != *local_sense) {
    continue;
  }
  return 1;
}

