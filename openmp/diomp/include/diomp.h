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
#include <gasnet_coll.h>
#include <gasnet_tools.h>
#include <gasnetex.h>

extern "C" void __init_diomp();

int omp_get_num_ranks();
int omp_get_rank_num();

#define EVERYTHING_SEG_HANDLERS()

#define PAGESZ GASNETT_PAGESIZE

enum class omp_op : gex_OP_t {

  // accessors
  load = GEX_OP_GET,
  store = GEX_OP_SET,
  compare_exchange = GEX_OP_FCAS,

  // arithmetic
  add = GEX_OP_ADD,
  fetch_add = GEX_OP_FADD,
  sub = GEX_OP_SUB,
  fetch_sub = GEX_OP_FSUB,
  inc = GEX_OP_INC,
  fetch_inc = GEX_OP_FINC,
  dec = GEX_OP_DEC,
  fetch_dec = GEX_OP_FDEC,
  mul = GEX_OP_MULT,
  fetch_mul = GEX_OP_FMULT,
  min = GEX_OP_MIN,
  fetch_min = GEX_OP_FMIN,
  max = GEX_OP_MAX,
  fetch_max = GEX_OP_FMAX,

  // bitwise operations
  bit_and = GEX_OP_AND,
  fetch_bit_and = GEX_OP_FAND,
  bit_or = GEX_OP_OR,
  fetch_bit_or = GEX_OP_FOR,
  bit_xor = GEX_OP_XOR,
  fetch_bit_xor = GEX_OP_FXOR,
};

static gex_TM_t diompTeam;
gex_Client_t diompClient;
gex_EP_t diompEp;
gex_Segment_t diompSeg;

void omp_get(void *dst, gasnet_node_t node, void *src, size_t nbytes);
void omp_put(gasnet_node_t node, void *dst, void *src, size_t nbytes);

void *omp_get_space(int node);
uintptr_t omp_get_length_space(int node);

void diomp_barrier();

/* Collective Operation */
// Broadcast
void omp_bcast(void *data, size_t nbytes, gasnet_node_t node);

void omp_allreduce(void *src, void *dst, size_t count, omp_op op);

// Reduction to one

/* ------------------------------------------------------------------------------------
 */
/* segment management */
#ifndef TEST_SEGSZ_PADFACTOR
#define TEST_SEGSZ_PADFACTOR 1
#endif

#define TEST_SEGSZ_REQUEST (TEST_SEGSZ_PADFACTOR * TEST_SEGSZ)

#define TEST_MINHEAPOFFSET alignup(128 * 4096, PAGESZ)

#endif
