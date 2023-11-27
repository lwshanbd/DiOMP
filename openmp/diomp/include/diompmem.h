/*
 * diompmem.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_MEM_H
#define DIOMP_MEM_H

#ifndef GASNET_PAR
#define GASNET_PAR
#endif

#include <gasnet.h>
#include <gasnetex.h>
#include <gasnet_tools.h>

#include "tools.h"


namespace diomp{
class MemoryManager {
  public:
    MemoryManager();
    ~MemoryManager();
    uintptr_t getSegmentSpace(int node);
    void* getSegmentAddr(int node);

  private:
    gasnet_seginfo_t *SegInfo = 0;
    int NodesNum;
    int NodeID;

    
};

} // namespace diomp

#endif