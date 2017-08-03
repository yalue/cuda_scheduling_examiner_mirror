// This file implements an interface to a simple barrier synchronization
// library. Internally, this library simply uses spinning, so it is intended
// for processes or threads that run on multiple CPUs.
#ifndef BARRIER_WAIT_H
#define BARRIER_WAIT_H
#include <stdatomic.h>

// Holds the state of a barrier synchronization object that can be used between
// multiple processes or threads. Members of this struct shouldn't be modified
// directly.
typedef struct {
  // This will be a pointer to a shared memory buffer.
  void *internal_buffer;
} ProcessBarrier;

// Initializes the ProcessBarrier struct so that the given number of processes
// have to wait. Returns nonzero on success and 0 on error.
int BarrierCreate(ProcessBarrier *b, int process_count);

// Cleans up the given initialized process barrier. It is the caller's
// responsibility to ensure this isn't called while processes are still waiting
// on the barrier.
void BarrierDestroy(ProcessBarrier *b);

// Causes the process to wait on the given barrier. Internally, this will
// involve busy-waiting. Returns nonzero on success. local_sense *must* be a
// pointer to a non-shared local integer, initialized to 0, and then unchanged
// by the caller in subsequent calls to BarrierWait.
int BarrierWait(ProcessBarrier *b, int *local_sense);

#endif  // BARRIER_WAIT_H
