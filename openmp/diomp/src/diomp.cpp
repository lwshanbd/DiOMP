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

#include "diomp.h"
#include "diompmem.h"
#include "tools.h"
#include <cstddef>
#include <cstdio>

diomp::MemoryManager *MemManger;

void my_gasnet_handler(gasnet_token_t token, void *buf, size_t nbytes) {
  // Do Nothing
}

gex_TM_t diompTeam;
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_Segment_t diompSeg;

void __init_diomp() {

  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", nullptr, nullptr,
                  0);

  gasnet_handlerentry_t handlers[1];
  handlers[0].index = 128;
  handlers[0].fnptr = (void (*)())my_gasnet_handler;
  size_t segsize = gasnet_getMaxGlobalSegmentSize();
  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, segsize));
  MemManger = new diomp::MemoryManager(diompTeam);
}

int omp_get_num_ranks() { return gex_TM_QuerySize(diompTeam); }

int omp_get_rank_num() { return gex_TM_QueryRank(diompTeam); }

void omp_get(void *dest, gasnet_node_t node, void *src, size_t nbytes) {
  if (gex_RMA_GetNBI(diompTeam, dest, node, src, nbytes, 0) != 0) {
    printf("Boom!\n");
  }
}

void omp_put(gasnet_node_t node, void *dest, void *src, size_t nbytes) {
  if (gex_RMA_PutNBI(diompTeam, node, dest, src, nbytes, GEX_EVENT_DEFER, 0) !=
      0) {
    printf("Boom!\n");
  }
}

void diomp_barrier() { gex_Event_Wait(gex_Coll_BarrierNB(diompTeam, 0)); }

void diomp_waitRMA() { gex_NBI_Wait(GEX_EC_PUT, 0); }

void *omp_get_space(int node) { return MemManger->getSegmentAddr(node); }

uintptr_t omp_get_length_space(int node) {
  return MemManger->getSegmentSpace(node);
}

void omp_bcast(void *data, size_t nbytes, gasnet_node_t node) {
  gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, node, data, data, nbytes, 0));
}

void omp_allreduce(void *src, void *dst, size_t count, int op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, dst, src, GEX_DT_I32,
                                        sizeof(GEX_DT_I32), count, (gex_OP_t)op,
                                        NULL, NULL, 0));
}

void *omp_ralloc(size_t Size) { return MemManger->allocate(Size); }
