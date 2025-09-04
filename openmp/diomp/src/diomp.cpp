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
#include "diompcomm.h"
#include "diompmem.h"
#include "tools.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <omp.h>

#ifdef DIOMP_ENABLE_CUDA  
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef DIOMP_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory> // Add this for std::unique_ptr

// GASNet related globals
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_TM_t diompTeam;
gex_Segment_t diompSeg;

// Memory management related
std::unique_ptr<diomp::MemoryManager> MemManager;
std::atomic<size_t> SegSize{0};
 
// Device related
int DevicesNum;
int CMode = 1;

// Communication related
std::unique_ptr<diomp::DiOMPCommunicator> Comm;

// Lock related
std::atomic<int> LockState{0};
std::vector<int> LockQueues;
std::mutex lockQueueMutex;
std::atomic<int> flag_lock{0}; // 0: initial, 1: locked, 2: occupied

size_t convertToBytes(const std::string &sizeStr) {
  if (sizeStr.empty()) {
    throw std::invalid_argument("Input string cannot be empty.");
  }

  size_t lastDigitIndex = sizeStr.find_last_of("0123456789");
  if (lastDigitIndex == std::string::npos) {
    throw std::invalid_argument("No numeric part found in the input string.");
  }

  std::string numberPart = sizeStr.substr(0, lastDigitIndex + 1);
  std::string unitPart = sizeStr.substr(lastDigitIndex + 1);

  size_t number = std::stoul(numberPart);

  std::transform(unitPart.begin(), unitPart.end(), unitPart.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  size_t Bytes = 0;
  if (unitPart == "kb") {
    if (number > std::numeric_limits<size_t>::max() / 1024) {
      throw std::overflow_error(
          "Value too large to be represented in Bytes as size_t.");
    }
    Bytes = number * 1024;
  } else if (unitPart == "mb") {
    if (number > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
      throw std::overflow_error(
          "Value too large to be represented in Bytes as size_t.");
    }
    Bytes = number * 1024 * 1024;
  } else if (unitPart == "gb") {
    if (number > std::numeric_limits<size_t>::max() / (1024 * 1024 * 1024)) {
      throw std::overflow_error(
          "Value too large to be represented in Bytes as size_t.");
    }
    Bytes = number * 1024 * 1024 * 1024;
  } else {
    Bytes = number;
  }
  return Bytes;
}

size_t getOMPDistributedSize() {
  const char *envValue = std::getenv("OMP_DISTRIBUTED_MEM_SIZE");
  if (envValue == nullptr) {
    size_t Bytes = 16ULL * 1024 * 1024 * 1024; // Default: 16GB
    return Bytes;
  }
  std::string valueStr(envValue);
  size_t Bytes = convertToBytes(valueStr);
  return Bytes;
}

// AM Handlers for DiOMP
void AcknowledgeLockAM(gex_Token_t Token, gex_AM_Arg_t Arg0) {
  flag_lock = Arg0;
  return;
}

void LockRequestAM(gex_Token_t Token) {
  gex_Token_Info_t info;
  int sourceRank = 0;
  gex_TI_t result = gex_Token_Info(Token, &info, GEX_TI_SRCRANK);
  if (result & GEX_TI_SRCRANK) {
    sourceRank = (int)info.gex_srcrank;
  }
  int expected = 0;
  if (LockState.compare_exchange_strong(expected, 1) && LockQueues.empty()) {
    gex_AM_ReplyShort1(Token, AM_ACK_LOCK, 0, 1);
  } else {
    std::lock_guard<std::mutex> lock(lockQueueMutex);
    LockQueues.push_back(sourceRank);
  }
}

void LockReleaseAM(gex_Token_t Token) {
  LockState.store(0);
  int nextRank = -1;
  {
    std::lock_guard<std::mutex> lock(lockQueueMutex);
    if (!LockQueues.empty()) {
      nextRank = LockQueues.front();
      LockQueues.erase(LockQueues.begin());
    }
  }
  if (nextRank != -1) {
    gex_AM_RequestShort1(diompTeam, nextRank, AM_ACK_LOCK, 0, 1);
    LockState.store(1);
  }
}

// AM Handler Table
gex_AM_Entry_t AMTable[] = {{AM_ACK_LOCK, (gex_AM_Fn_t)AcknowledgeLockAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQREP, 0},
                            {AM_LOCK_REQ_IDX, (gex_AM_Fn_t)LockRequestAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 1},
                            {AM_LOCK_REL_IDX, (gex_AM_Fn_t)LockReleaseAM,
                             GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST, 0}};

// Initialize DIOMP runtime
void __init_diomp() {
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  if (SegSize == 0)
    SegSize = getOMPDistributedSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, SegSize));
  gex_EP_RegisterHandlers(diompEp, AMTable,
                          sizeof(AMTable) / sizeof(gex_AM_Entry_t));
  MemManager = std::make_unique<diomp::MemoryManager>(diompTeam);

  // Create base communicator by default
  Comm = std::make_unique<diomp::DiOMPCommunicator>();
}

void __init_diomp_target(int Mode = 1) {
  CMode = Mode;
  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);
  if (SegSize == 0)
    SegSize = getOMPDistributedSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, 1024*1024));
  gex_EP_RegisterHandlers(diompEp, AMTable,
                          sizeof(AMTable) / sizeof(gex_AM_Entry_t));

  MemManager = std::make_unique<diomp::MemoryManager>(diompTeam, Mode);

#ifdef DIOMP_ENABLE_CUDA
  MemManager = std::make_unique<diomp::CUDAMemoryManager>(diompTeam, Mode, SegSize);
  auto cudaComm = std::make_unique<diomp::DiOMPCUDACommunicator>(Mode);
  cudaComm->initNCCL();
  Comm = std::move(cudaComm);
#endif

#ifdef DIOMP_ENABLE_HIP
  MemManager = std::make_unique<diomp::HIPMemoryManager>(diompTeam, Mode, SegSize);
  auto hipComm = std::make_unique<diomp::DiOMPHIPCommunicator>(Mode);
  hipComm->initRCCL();
  Comm = std::move(hipComm);
#endif
  
}

void *diomp_device_alloc(size_t Size, int DeviceId) {
  void *MemAddr = nullptr;
  MemAddr = MemManager->deviceAlloc(Size, DeviceId);
  return MemAddr;
}

void diomp_device_dealloc() {
  MemManager->deviceDealloc();
  return;
}

void omp_set_distributed_size(size_t Size) { SegSize = Size; }

// Get the total number of ranks
int omp_get_num_ranks() { return gex_TM_QuerySize(diompTeam); }

// Get the rank number of the current process
int omp_get_rank_num() { return gex_TM_QueryRank(diompTeam); }

// Get the starting address of the memory segment on a Rank
void *omp_get_space(int Rank) { return MemManager->getSegmentAddr(Rank); }

// Get the size of the memory segment on a Rank
uintptr_t omp_get_length_space(int Rank) {
  return MemManager->getSegmentSpace(Rank);
}

// Allocate shared memory accessible by all Ranks
void *llvm_omp_distributed_alloc(size_t Size) {
  return MemManager->globalAlloc(Size);
}

// Templated version to allocate typed shared memory
template <typename T> void *diomp_alloc(size_t Size) {
  return MemManager->globalAlloc(Size * sizeof(T));
}

// RMA Operations
// Get data from a remote Rank
void ompx_get(void *Dst, int Rank, void *Src, size_t Size) {
  Comm->get(Dst, Rank, Src, Size);
}

// Put data to a remote Rank
void ompx_put(int Rank, void *Dst, void *Src, size_t Size) {
  Comm->put(Rank, Dst, Src, Size);
}

#ifdef OPENMP_ENABLE_DIOMP_DEVICE


// Overloaded ompx_get equivalent to ompx_dget
// void ompx_get(void *Dst, int Rank, void *Src, size_t Size, int DstID, int SrcID) {
//   Comm->dget(Dst, Rank, Src, Size, DstID, SrcID);
// }

// // Overloaded ompx_put equivalent to ompx_dput
// void ompx_put(void *Dst, int Rank, void *Src, size_t Size, int DstID, int SrcID) {
//   Comm->dput(Dst, Rank, Src, Size, DstID, SrcID);
// }

void ompx_dget(void *Dst, int Rank, void *Src, size_t Size, int DstID,
               int SrcID) {
  void *DevDst = Dst;
  void *DevSrc = Src;
  // Rule: if the pointer is not present. It is device pointer.
  // It is UB to use host pointer which is not mapped to device pointer.
  if (omp_target_is_present(Dst, DstID)) {
    DevDst = omp_get_mapped_ptr(Dst, DstID);
  }
  if (omp_target_is_present(Src, SrcID)) {
    DevSrc = omp_get_mapped_ptr(Src, SrcID);
  }
  Comm->dget(DevDst, Rank, DevSrc, Size, DstID, SrcID);
}

void ompx_dput(void *Dst, int Rank, void *Src, size_t Size, int DstID,
               int SrcID) {
  void *DevDst = Dst;
  void *DevSrc = Src;
  // Rule: if the pointer is not present. It is device pointer.
  // It is UB to use host pointer which is not mapped to device pointer.
  if (omp_target_is_present(Dst, DstID)) {
    DevDst = omp_get_mapped_ptr(Dst, DstID);
  }
  if (omp_target_is_present(Src, SrcID)) {
    DevSrc = omp_get_mapped_ptr(Src, SrcID);
  }
  Comm->dput(DevDst, Rank, DevSrc, Size, DstID, SrcID);
}

#endif

// End of RMA Operations

// Synchronization Operations
// Barrier synchronization across all Ranks or within group
void diomp_barrier(ompx_group_t *group) { 
  if (group && group->rank != -1) {
    printf("barrier in group\n");
    gex_Event_Wait(gex_Coll_BarrierNB(group->team, 0));
  } else {
    Comm->barrier(); 
  }
}

// ompx_barrier - equivalent to diomp_barrier
void ompx_barrier(ompx_group_t *group) { 
  if (group && group->rank != -1) {
    printf("barrier in group\n");
    gex_Event_Wait(gex_Coll_BarrierNB(group->team, 0));
  } else {
    Comm->barrier(); 
  }
}

// Wait for completion of all RMA operations
void diomp_waitALLRMA() { Comm->waitAllRMA(); }

// Fence is equivalent to waitAllRMA
void ompx_fence() { Comm->waitAllRMA(); }

// Wait for completion of a specific RMA operation
void diomp_waitRMA(omp_event_t Ev) { Comm->waitRMA(Ev); }

void diomp_lock(int Rank) { Comm->lock(Rank); }

void diomp_unlock(int Rank) { Comm->unlock(Rank); }

// End of Synchronization Operations

// Collective Operations
// Broadcast data from root to all Ranks or within group
void omp_bcast(void *Data, size_t Size, int Rank, ompx_group_t *group) {
  if (group && group->rank != -1) {
    // Convert group-local root to global rank if needed
    gex_Rank_t gex_root = (Rank < group->size) ? (gex_Rank_t)group->ranks[Rank] : (gex_Rank_t)Rank;
    gex_Event_Wait(gex_Coll_BroadcastNB(group->team, gex_root, Data, Data, Size, 0));
  } else {
    gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, Rank, Data, Data, Size, 0));
  }
}

// All-reduce operation across all Ranks or within group
void omp_allreduce(void *Src, void *Dst, size_t Size, omp_dt_t Dt,
                   omp_op_t Op, ompx_group_t *group) {
  if (group && group->rank != -1) {
    gex_Event_Wait(gex_Coll_ReduceToAllNB(group->team, Dst, Src, Dt, sizeof(Dt),
                                          Size, Op, NULL, NULL, 0));
  } else {
    gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, Dst, Src, Dt, sizeof(Dt),
                                          Size, Op, NULL, NULL, 0));
  }
}

void omp_reduce(void *Src, void *Dst, size_t Size, omp_dt_t Dt, omp_op_t Op,
                int Root, ompx_group_t *group) {
  if (group && group->rank != -1) {
    gex_Rank_t gex_root = (Root < group->size) ? (gex_Rank_t)group->ranks[Root] : (gex_Rank_t)Root;
    gex_Event_Wait(gex_Coll_ReduceToOneNB(group->team, gex_root, Dst, Src, Dt,
                                          sizeof(Dt), Size, Op, NULL, NULL, 0));
  } else {
    gex_Event_Wait(gex_Coll_ReduceToOneNB(diompTeam, Root, Dst, Src, Dt,
                                          sizeof(Dt), Size, Op, NULL, NULL, 0));
  }
}

#ifdef OPENMP_ENABLE_DIOMP_DEVICE

void ompx_dbcast(void *Data, size_t Size, omp_device_dt_t Dt, int Rank,
                 int DstID, ompx_group_t *group) {
  Comm->dbcast(Data, Size, Dt, Rank, DstID, group);
}

void ompx_dallreduce(void *Src, void *Dst, size_t Size, omp_device_dt_t Dt,
                     omp_red_op_t Op, int DstID, ompx_group_t *group) {
  Comm->dallreduce(Src, Dst, Size, Dt, Op, DstID, group);
}

void ompx_dreduce(void *Src, void *Dst, size_t Size, omp_device_dt_t Dt,
                  omp_red_op_t Op, int Root, int DstID, ompx_group_t *group) {
  Comm->dreduce(Src, Dst, Size, Dt, Op, Root, DstID, group);
}

#endif
