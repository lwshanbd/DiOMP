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

  // Handler init

  gasnet_handlerentry_t handlers[1];
  handlers[0].index = 128;
  handlers[0].fnptr = (void (*)())my_gasnet_handler;
  size_t page_size = GASNET_PAGESIZE;
  size_t segsize = 1024 * 1024 * 1024;

  gasnet_init(NULL, NULL);
  GASNET_Safe(gasnet_attach(
      handlers, sizeof(handlers) / sizeof(gasnet_handlerentry_t), segsize, 0));

  MemManger = new diomp::MemoryManager();
}

int omp_get_num_ranks() { return gasnet_nodes(); }

int omp_get_rank_num() { return gasnet_mynode(); }

void omp_get(void *dest, gasnet_node_t node, void *src, size_t nbytes) {
  gasnet_get_bulk(dest, node, src, nbytes);
}

void omp_put(gasnet_node_t node, void *dest, void *src, size_t nbytes) {
  gasnet_put_bulk(node, dest, src, nbytes);
}

void *omp_get_space(int node) { return MemManger->getSegmentAddr(node); }

uintptr_t omp_get_length_space(int node) {
  return MemManger->getSegmentSpace(node);
}
