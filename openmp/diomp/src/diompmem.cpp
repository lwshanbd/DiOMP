#include "diompmem.h"

namespace diomp{


MemoryManager::MemoryManager(gex_TM_t gexTeam){
    NodesNum = gex_TM_QuerySize(gexTeam);
    NodeID = gex_TM_QueryRank(gexTeam);
    SegInfo = new gasnet_seginfo_t[NodesNum];
    GASNET_Safe(gasnet_getSegmentInfo(SegInfo, NodesNum));
    //GASNET_Safe(gasnet_getSegmentInfo(SegInfo, NodesNum));
}

uintptr_t MemoryManager::getSegmentSpace(int node){
    return SegInfo[node].size;
}

void* MemoryManager::getSegmentAddr(int node){
    return SegInfo[node].addr;
}

} // namespace diomp