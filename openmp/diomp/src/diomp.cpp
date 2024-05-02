/*
 * diomp.cpp
 */
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diomp.hpp"
#include "diompmem.h"
#include "tools.h"
#include <cstddef>
#include <cstdio>
#include <stdexcept>


namespace omp {
namespace diomp {

MemoryManager *MemManger;
gex_TM_t diompTeam;
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_Segment_t diompSeg;

// Initialize DIOMP runtime
void __init_diomp() {
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  size_t SegSize = gasnet_getMaxGlobalSegmentSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, SegSize));
  MemManger = new diomp::MemoryManager(diompTeam);
}

// Get the total number of ranks
int omp_get_num_ranks() { return gex_TM_QuerySize(diompTeam); }

// Get the rank number of the current process
int omp_get_rank_num() { return gex_TM_QueryRank(diompTeam); }

// Get data from a remote nodeßß
void omp_get(void *dest, int node, void *src, size_t nbytes) {
  auto Error = gex_RMA_GetNBI(diompTeam, dest, node, src, nbytes, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP GET Error! Error code is %d", Error);
  }
}

// Put data to a remote node
void omp_put(int node, void *dest, void *src, size_t nbytes) {
  auto Error =
      gex_RMA_PutNBI(diompTeam, node, dest, src, nbytes, GEX_EVENT_DEFER, 0);
  if (Error != 0) {
    THROW_ERROR("OpenMP PUT Error! Error code is %d", Error);
  }
}

// Barrier synchronization across all nodes
void diomp_barrier() { gex_Event_Wait(gex_Coll_BarrierNB(diompTeam, 0)); }

// Wait for completion of all RMA operations
void diomp_waitALLRMA() { gex_NBI_Wait(GEX_EC_ALL, 0); }

// Wait for completion of a specific RMA operation
void diomp_waitRMA(omp_event ev) { gex_NBI_Wait(ev, 0); }

// Get the starting address of theß memory segment on a nodeß
void *omp_get_space(int node) { return MemManger->getSegmentAddr(node); }

// Get the size of the memory segment on a node
uintptr_t omp_get_length_space(int node) {
  return MemManger->getSegmentSpace(node);
}

// Broadcast data from root to all nodes
void omp_bcast(void *data, size_t nbytes, int node) {
  gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, node, data, data, nbytes, 0));
}

// All-reduce operation across all nodes
void omp_allreduce(void *src, void *dst, size_t count, omp_dt dt, omp_op op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, dst, src, dt, sizeof(dt),
                                        count, op, NULL, NULL, 0));
}

// Allocate shared memory accessible by all nodes
void *llvm_omp_distributed_alloc(size_t Size) {
  return MemManger->globalAlloc(Size);
}

// Templated version to allocate typed shared memory
template <typename T> 
void *diomp_alloc(size_t Size) {
  return MemManger->globalAlloc(Size * sizeof(T));
}

} // namespace diomp
} // namespace omp