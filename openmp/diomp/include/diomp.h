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
#include <gasnetex.h>
#include <gasnet_tools.h>

void diomp_init();

int omp_get_num_ranks();
int omp_get_rank_num();


#define EVERYTHING_SEG_HANDLERS()

#define PAGESZ GASNETT_PAGESIZE


#define GASNET_Safe(fncall) do {                                     \
    int _retval;                                                     \
    if ((_retval = fncall) != GASNET_OK) {                           \
      fprintf(stderr, "ERROR calling: %s\n"                          \
                   " at: %s:%i\n"                                    \
                   " error: %s (%s)\n",                              \
              #fncall, __FILE__, __LINE__,                           \
              gasnet_ErrorName(_retval), gasnet_ErrorDesc(_retval)); \
      fflush(stderr);                                                \
      gasnet_exit(_retval);                                          \
    }                                                                \
  } while(0)


void omp_get ( void *dest , gasnet_node_t node , void *src , size_t nbytes );
void omp_put ( gasnet_node_t node , void *dest , void *src , size_t nbytes );
void* omp_get_space(int node);
long long omp_get_length_space(int node);

void diomp_barrier()
{
    gasnet_barrier_notify(0, GASNET_BARRIERFLAG_ANONYMOUS);
    gasnet_barrier_wait(0, GASNET_BARRIERFLAG_ANONYMOUS);
}

//void gasnet_get ( void *dest , gasnet_node_t node , void *src , size t nbytes ) 
//void gasnet_put ( gasnet node t node , void *dest , void *src , size t nbytes )

/* ------------------------------------------------------------------------------------ */
/* segment management */
#ifndef TEST_SEGSZ_PADFACTOR
  #define TEST_SEGSZ_PADFACTOR 1
#endif

#define TEST_SEGSZ_REQUEST (TEST_SEGSZ_PADFACTOR*TEST_SEGSZ)

#define TEST_MINHEAPOFFSET  alignup(128*4096,PAGESZ)


#endif
