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



diomp::MemoryManager *MemManger;
gex_TM_t diompTeam;
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_Segment_t diompSeg;
int LockState = 0;
int flag_lock = 0; // 0: initial, 1: locked, 2: occupied

// AM Handlers for DiOMP

void DoNothingAM(gex_Token_t Token, gex_AM_Arg_t Arg0) {
  printf("DoNothingAM\n");
  flag_lock = Arg0;
  return;
}

void LockRequestAM(gex_Token_t Token) {
  if (LockState == 0) {
    LockState = 1;
    gex_AM_ReplyShort1(Token, AM_DO_NOTHING, 0, 1);
  } else{
    gex_AM_ReplyShort1(Token, AM_DO_NOTHING, 0, 2);
  }
}

void LockReleaseAM(gex_Token_t Token) {
  LockState = 0;
}

// AM Handler Table
gex_AM_Entry_t AMTable[] = {{AM_DO_NOTHING, (gex_AM_Fn_t)DoNothingAM,
                            GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQREP, 0},
                           {AM_LOCK_REQ_IDX, (gex_AM_Fn_t)LockRequestAM,
                            GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 1},
                           {AM_LOCK_REL_IDX, (gex_AM_Fn_t)LockReleaseAM,
                            GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 0}};

// Initialize DIOMP runtime
void __init_diomp() {
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  size_t SegSize = gasnet_getMaxGlobalSegmentSize()/2;
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, SegSize));
  gex_EP_RegisterHandlers(diompEp, AMTable, sizeof(AMTable)/sizeof(gex_AM_Entry_t));
  MemManger = new diomp::MemoryManager(diompTeam);
}

// Get the total number of ranks
int omp_get_num_ranks() { return gex_TM_QuerySize(diompTeam); }

// Get the rank number of the current process
int omp_get_rank_num() { return gex_TM_QueryRank(diompTeam); }

// Get the starting address of the memory segment on a node
void *omp_get_space(int node) { return MemManger->getSegmentAddr(node); }

// Get the size of the memory segment on a node
uintptr_t omp_get_length_space(int node) {
  return MemManger->getSegmentSpace(node);
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

// RMA Operations
// Get data from a remote node
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

// End of RMA Operations

// Synchronization Operations
// Barrier synchronization across all nodes
void diomp_barrier() { gex_Event_Wait(gex_Coll_BarrierNB(diompTeam, 0)); }

// Wait for completion of all RMA operations
void diomp_waitALLRMA() { gex_NBI_Wait(GEX_EC_ALL, 0); }

// Wait for completion of a specific RMA operation
void diomp_waitRMA(omp_event_t ev) { gex_NBI_Wait(ev, 0); }

void diomp_lock(int Rank){
  while (1) {
    gex_AM_RequestShort0(diompTeam, Rank, AM_LOCK_REQ_IDX, 0);
    GASNET_BLOCKUNTIL(flag_lock == 1 || flag_lock == 2);
    if (flag_lock == 1)
      break;
  }
}

void diomp_unlock(int Rank) {
    gex_AM_RequestShort0(diompTeam, Rank, AM_LOCK_REL_IDX, 0);
}

// End of Synchronization Operations

// Collective Operations
// Broadcast data from root to all nodes
void omp_bcast(void *data, size_t nbytes, int node) {
  gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, node, data, data, nbytes, 0));
}

// All-reduce operation across all nodes
void omp_allreduce(void *src, void *dst, size_t count, omp_dt_t dt, omp_op_t op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, dst, src, dt, sizeof(dt),
                                        count, op, NULL, NULL, 0));
}

