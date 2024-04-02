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


#include <gasnet.h>
#include <gasnet_coll.h>
#include <gasnet_tools.h>
#include <gasnetex.h>



typedef enum omp_op {
  // accessors
  omp_op_load = GEX_OP_GET,
  omp_op_store = GEX_OP_SET,
  omp_op_compare_exchange = GEX_OP_FCAS,

  // arithmetic
  omp_op_add = GEX_OP_ADD,
  omp_op_fetch_add = GEX_OP_FADD,
  omp_op_sub = GEX_OP_SUB,
  omp_op_fetch_sub = GEX_OP_FSUB,
  omp_op_inc = GEX_OP_INC,
  omp_op_fetch_inc = GEX_OP_FINC,
  omp_op_dec = GEX_OP_DEC,
  omp_op_fetch_dec = GEX_OP_FDEC,
  omp_op_mul = GEX_OP_MULT,
  omp_op_fetch_mul = GEX_OP_FMULT,
  omp_op_min = GEX_OP_MIN,
  omp_op_fetch_min = GEX_OP_FMIN,
  omp_op_max = GEX_OP_MAX,
  omp_op_fetch_max = GEX_OP_FMAX,

  // bitwise operations
  omp_op_bit_and = GEX_OP_AND,
  omp_op_fetch_bit_and = GEX_OP_FAND,
  omp_op_bit_or = GEX_OP_OR,
  omp_op_fetch_bit_or = GEX_OP_FOR,
  omp_op_bit_xor = GEX_OP_XOR,
  omp_op_fetch_bit_xor = GEX_OP_FXOR,
} omp_op_t;

#ifdef __cplusplus
extern "C" {
#endif

extern gex_TM_t diompTeam;
extern gex_Client_t diompClient;
extern gex_EP_t diompEp;
extern gex_Segment_t diompSeg;

void __init_diomp();

int omp_get_num_ranks();
int omp_get_rank_num();

void omp_get(void *dst, gasnet_node_t node, void *src, size_t nbytes);
void omp_put(gasnet_node_t node, void *dst, void *src, size_t nbytes);

void *omp_get_space(int node);
uintptr_t omp_get_length_space(int node);

void diomp_barrier();
void diomp_waitRMA();


void omp_bcast(void *data, size_t nbytes, gasnet_node_t node);

// Experimental. Only for benchmark
void omp_allreduce(void *src, void *dst, size_t count, int op);

void* omp_ralloc(size_t Size);

#ifdef __cplusplus
}
#endif

#endif
