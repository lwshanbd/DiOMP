#include "diompmem.h"
#include <cstddef>

namespace diomp {

MemoryManager::MemoryManager(gex_TM_t gexTeam) {
  NodesNum = gex_TM_QuerySize(gexTeam);
  NodeID = gex_TM_QueryRank(gexTeam);
  SegInfo.resize(NodesNum);
  for (auto &Seg : SegInfo) {
    void *SegBase = 0;
    size_t SegSize = 0;
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(diompTeam, &Seg - &SegInfo[0],
                                              &SegBase, nullptr, &SegSize, 0));
    Seg.SegStart = SegBase;
    Seg.SegSize = SegSize;
    Seg.SegRemain = SegBase;
  }
  LocalSegStart = SegInfo[NodeID].SegStart;
  LocalSegRemain = SegInfo[NodeID].SegRemain;
  LocalSegSize = SegInfo[NodeID].SegSize;
}

uintptr_t MemoryManager::getSegmentSpace(int Rank) {
  return SegInfo[Rank].SegSize;
}

void *MemoryManager::getSegmentAddr(int Rank) { return SegInfo[Rank].SegStart; }

void *MemoryManager::globalAlloc(size_t Size) {
  if (Size > getAvailableSize()) {
    return nullptr;
  }

  void *Ptr = LocalSegRemain;
  LocalSegRemain = reinterpret_cast<char *>(LocalSegRemain) + Size;
  MemBlocks.push_back({Ptr, Size});

  return Ptr;
}

size_t MemoryManager::getAvailableSize() const {
  uintptr_t start = reinterpret_cast<uintptr_t>(LocalSegStart);
  uintptr_t end = start + LocalSegSize;

  if (end < start) {
    return 0; // Handle overflow
  }

  return static_cast<size_t>(end - start);
}

size_t MemoryManager::getOffset(void *Ptr) {
  uintptr_t start = reinterpret_cast<uintptr_t>(LocalSegStart);
  uintptr_t ptr = reinterpret_cast<uintptr_t>(Ptr);

  if (ptr < start || ptr > start + LocalSegSize) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(ptr - start);
}

size_t MemoryManager::getOffset(void *Ptr, int Rank) {
  uintptr_t start = reinterpret_cast<uintptr_t>(SegInfo[Rank].SegStart);
  uintptr_t ptr = reinterpret_cast<uintptr_t>(Ptr);

  if (ptr < start || ptr > start + SegInfo[Rank].SegSize) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(ptr - start);
}

bool MemoryManager::validGlobalAddr(void *Ptr, int Rank) {
  if (!Ptr) {
    return false;
  }

  size_t Offset = reinterpret_cast<char *>(Ptr) -
                  reinterpret_cast<char *>(SegInfo[Rank].SegStart);
  return Offset <= SegInfo[Rank].SegSize;
}

void *MemoryManager::syncGlobalfromLocalAddr(void *Ptr, int Rank) {
  void *GlobalPtr = SegInfo[Rank].SegStart;
  if (!validGlobalAddr(GlobalPtr, Rank)) {
    return nullptr;
  }
  return GlobalPtr;
}

} // namespace diomp
