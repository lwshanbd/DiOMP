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

#include <cstddef>
#ifndef GASNET_PAR
#define GASNET_PAR
#endif

#include <gasnet.h>
#include <gasnetex.h>
#include <gasnet_tools.h>

#include <vector>

#include "tools.h"

extern gex_TM_t diompTeam;
extern gex_Client_t diompClient;
extern gex_EP_t diompEp;
extern gex_Segment_t diompSeg;

struct gex_Seginfo_t{
  void *SegStart;
  void *SegRemain;
  size_t SegSize;
};

struct MemoryBlock{
  void *Ptr;
  size_t Size;
};

namespace diomp{

class MemoryManager {
  private:
    int NodesNum;
    int NodeID;
    std::vector<MemoryBlock> MemBlocks;
    std::vector<gex_Seginfo_t> SegInfo;

    // Local Segment Information
    void *LocalSegStart;
    void *LocalSegRemain;
    size_t LocalSegSize;

  public:
    MemoryManager(gex_TM_t gexTeam);
    ~MemoryManager(){};

    size_t getSegmentSpace(int Rank);
    void *getSegmentAddr(int Rank);

    void *globalAlloc(size_t Size);

    size_t getAvailableSize() const;
    size_t getOffset(void *Ptr); // Compute ptr on local memory
    size_t getOffset(void *Ptr, int Rank); // Compute ptr on remote memory

    bool validGlobalAddr(void *Ptr, int Rank);
    void *syncGlobalfromLocalAddr(void *Ptr, int Rank);

};

} // namespace diomp

#endif // DIOMP_MEM_H
