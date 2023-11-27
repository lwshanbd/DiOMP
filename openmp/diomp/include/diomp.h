/*
 * diomp.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_H
#define DIOMP_H
#define GASNET_PAR
#include <cstdio>

#include <gasnet.h>
#include <gasnet_tools.h>
#include <gasnetex.h>

void diomp_init(int argc, char *argv[]);

int omp_get_num_ranks();
int omp_get_rank_num();

#define EVERYTHING_SEG_HANDLERS()

#define PAGESZ GASNETT_PAGESIZE

void omp_get(void *dest, gasnet_node_t node, void *src, size_t nbytes);
void omp_put(gasnet_node_t node, void *dest, void *src, size_t nbytes);
void *omp_get_space(int node);
uintptr_t omp_get_length_space(int node);

void diomp_barrier() {
  gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
  gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
}

// void gasnet_get ( void *dest , gasnet_node_t node , void *src , size t nbytes
// ) void gasnet_put ( gasnet node t node , void *dest , void *src , size t
// nbytes )

/* ------------------------------------------------------------------------------------
 */
/* segment management */
#ifndef TEST_SEGSZ_PADFACTOR
#define TEST_SEGSZ_PADFACTOR 1
#endif

#define TEST_SEGSZ_REQUEST (TEST_SEGSZ_PADFACTOR * TEST_SEGSZ)

#define TEST_MINHEAPOFFSET alignup(128 * 4096, PAGESZ)

#endif
