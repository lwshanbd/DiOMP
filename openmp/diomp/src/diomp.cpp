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

void my_gasnet_handler(gasnet_token_t token, void* buf, size_t nbytes) {
    printf("XXXXX\n");
}


void diomp_init(){

    printf("Init successful!\n");
    
    // Handler init

    gasnet_handlerentry_t handlers[1];
    handlers[0].index = 128;
    handlers[0].fnptr = (void(*)())my_gasnet_handler;
    size_t page_size = GASNET_PAGESIZE;
    size_t segsize = 1024 * 1024 * 1024;

    gasnet_init(NULL, NULL);
    GASNET_Safe(gasnet_attach(handlers, sizeof(handlers)/sizeof(gasnet_handlerentry_t), 
                            segsize, 0));

    int numOfNodes = gasnet_nodes();
    printf("number of nodes %d\n", numOfNodes);
}

int omp_get_num_ranks(){
    return gasnet_nodes();
}

int omp_get_rank_num(){
    return gasnet_mynode();
}



void omp_get( void *dest , gasnet_node_t node , void *src , size_t nbytes ){
    gasnet_get_bulk(dest, node, src, nbytes);
}

void omp_put( gasnet_node_t node , void *dest , void *src , size_t nbytes ){
    gasnet_put_bulk(node, dest, src, nbytes);
}

void* omp_get_space(int node){
    void *addr = NULL;
    unsigned long len = NULL;
    gex_Flags_t imm = GEX_FLAG_IMMEDIATE;
    static gex_TM_t _test_tm0;
    gex_Event_t ev =  gex_EP_QueryBoundSegmentNB(_test_tm0, node, &addr, NULL, &len, imm);
    printf("size of Seg %d\n", len);
    return addr;
}

long long omp_get_length_space(int node){

    // gasnet_seginfo_t* seginfo_table;
    // seginfo_table = new gasnet_seginfo_t[gasnet_nodes()+1];
    // gasnet_getSegmentInfo(seginfo_table, gasnet_nodes());

    void *addr = NULL;
    uintptr_t len;
    gex_Flags_t imm = GEX_FLAG_IMMEDIATE;
    static gex_TM_t _test_tm0;
    printf("XXXXX value%lu\n", len);
    gex_Event_t ev =  gex_EP_QueryBoundSegmentNB(_test_tm0, node, &addr, NULL, &len, imm);
    printf("XXXXX value%lu\n", len);
    //printf("addres of len %p\n", seginfo_table[0].size);
    return 0;
}


