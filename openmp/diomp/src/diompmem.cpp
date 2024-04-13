#include "diompmem.h"
#include <cstddef>

namespace diomp {

MemoryManager::MemoryManager(gex_TM_t gexTeam) {
  NodesNum = gex_TM_QuerySize(gexTeam);
  NodeID = gex_TM_QueryRank(gexTeam);
  SegInfo = new gex_Seginfo_t[NodesNum];
  for (int r = 0; r < NodesNum; r++) {
    void *segment_base = 0;
    size_t segment_size = 0;
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(diompTeam, r, &segment_base,
                                              nullptr, &segment_size, 0));
    SegInfo[r].SegStart = segment_base;
    SegInfo[r].SegSize = segment_size;
    SegInfo[r].SegRemain = segment_base;
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
  LocalSegRemain = (char *)LocalSegRemain + Size;
  MemBlocks.push_back({Ptr, Size});
  return Ptr;
}

size_t MemoryManager::getAvailableSize() const {
  return (char *)LocalSegStart + LocalSegSize - (char *)LocalSegStart;
}

size_t MemoryManager::getOffset(void *Ptr) {
  size_t Offset = (char *)Ptr - (char *)LocalSegStart;
  if (Offset < 0 || Offset > LocalSegSize)
    return -1;
  return Offset;
}

size_t MemoryManager::getOffset(void *Ptr, int Rank) {
  size_t Offset = (char *)Ptr - (char *)SegInfo[Rank].SegStart;
  if (Offset < 0 || Offset > SegInfo[Rank].SegSize)
    return -1;
  return Offset;
}

bool MemoryManager::validGlobalAddr(void *Ptr, int Rank){
  if(Ptr == nullptr){
    return false;
  }

  size_t Offset = (char *)Ptr - (char *)SegInfo[Rank].SegStart;
  if (Offset < 0 || Offset > SegInfo[Rank].SegSize)
    return false;
  return true;
}

void *MemoryManager::syncGlobalfromLocalAddr(void *Ptr, int Rank){
  size_t offset = getOffset(Ptr);
  void *GlobalPtr = SegInfo[Rank].SegStart;
  if(!validGlobalAddr(GlobalPtr, Rank)){
    return nullptr;
  }
  return GlobalPtr;
}


} // namespace diomp

// template <typename T> T *shmem_alloc(size_t count) {
//   size_t size = count * sizeof(T);
//   void *ptr = memory_manager.allocate(size);
//   static_assert(ptr == nullptr, "Allocat global memory failed\n");
//   return static_cast<T *>(ptr);
// }
