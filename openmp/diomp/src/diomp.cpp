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

diomp::MemoryManager *MemManger;

void my_gasnet_handler(gasnet_token_t token, void *buf, size_t nbytes) {
  printf("XXXXX\n");
}

void diomp_init(int argc, char *argv[]) {

  gex_Client_Init(&diompClient, &diompEp, &diompTeam, "diomp", &argc, &argv, 0);
  // Handler init

  gasnet_handlerentry_t handlers[1];
  handlers[0].index = 128;
  handlers[0].fnptr = (void (*)())my_gasnet_handler;
  size_t page_size = GASNET_PAGESIZE;
  size_t segsize = 10000000000;

  GASNET_Safe(gex_Segment_Attach(&diompSeg, diompTeam, segsize));
  MemManger = new diomp::MemoryManager();
}

int omp_get_num_ranks() { return gasnet_nodes(); }

int omp_get_rank_num() { return gasnet_mynode(); }

void omp_get(void *dest, gasnet_node_t node, void *src, size_t nbytes) {
  gex_RMA_GetNBI(diompTeam, dest, node, src, nbytes, 0);
}

void omp_put(gasnet_node_t node, void *dest, void *src, size_t nbytes) {
  gex_RMA_PutNBI(diompTeam, node, dest, src, nbytes, GEX_EVENT_DEFER, 0);
}

void diomp_barrier() { gex_NBI_Wait(GEX_EC_ALL, 0); }

void *omp_get_space(int node) { return MemManger->getSegmentAddr(node); }

uintptr_t omp_get_length_space(int node) {
  return MemManger->getSegmentSpace(node);
}

void omp_bcast(void *data, size_t nbytes, gasnet_node_t node) {
  gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, node, data, data, nbytes, 0));
}

void omp_allreduce(void *src, void *dst, size_t count, omp_op op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(diompTeam, dst, src, 
                        GEX_DT_I32, sizeof(GEX_DT_I32), count,
                        (gex_OP_t)op, NULL, NULL, 0));
}
