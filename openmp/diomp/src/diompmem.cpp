#include "diompmem.h"

namespace diomp{


MemoryManager::MemoryManager(){
    NodesNum = gasnet_nodes();
    NodeID = gasnet_mynode();
    SegInfo = new gasnet_seginfo_t[NodesNum];
    GASNET_Safe(gasnet_getSegmentInfo(SegInfo, NodesNum));
}

uintptr_t MemoryManager::getSegmentSpace(int node){
    return SegInfo[node].size;
}

void* MemoryManager::getSegmentAddr(int node){
    return SegInfo[node].addr;
}

} // namespace diomp