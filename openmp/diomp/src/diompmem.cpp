#include "diompmem.h"
#include "tools.h"
#include <cstddef>

namespace diomp {

// Base MemoryManager implementation
MemoryManager::MemoryManager(gex_TM_t GexTeam, int Mode) {
  RanksNum = gex_TM_QuerySize(GexTeam);
  MyRank = gex_TM_QueryRank(GexTeam);
  this->Mode = Mode;

  // Initialize segment info
  SegInfo.resize(RanksNum);
  for (auto &Seg : SegInfo) {
    void *SegBase = 0;
    size_t SegSize = 0;
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(diompTeam, &Seg - &SegInfo[0],
                                              &SegBase, nullptr, &SegSize, 0));
    Seg.SegStart = SegBase;
    Seg.SegSize = SegSize;
    Seg.SegRemain = SegBase;
  }

  LocalSegStart = SegInfo[MyRank].SegStart;
  LocalSegRemain = SegInfo[MyRank].SegRemain;
  LocalSegSize = SegInfo[MyRank].SegSize;
}

void *MemoryManager::globalAlloc(size_t Size) {
  if (Size > getAvailableSize()) {
    return nullptr;
  }

  void *Ptr = LocalSegRemain;
  LocalSegRemain = reinterpret_cast<char *>(LocalSegRemain) + Size;
  MemBlocks.push_back({Ptr, Size});
  return Ptr;
}

size_t MemoryManager::getSegmentSpace(int Rank) {
  return SegInfo[Rank].SegSize;
}

void *MemoryManager::getSegmentAddr(int Rank) { return SegInfo[Rank].SegStart; }

size_t MemoryManager::getAvailableSize() {
  uintptr_t Start = reinterpret_cast<uintptr_t>(LocalSegStart);
  uintptr_t End = Start + LocalSegSize;
  if (End < Start) {
    return 0; // Handle overflow
  }
  return static_cast<size_t>(End - Start);
}

size_t MemoryManager::getOffset(void *Ptr, int Rank) {
  uintptr_t Start = reinterpret_cast<uintptr_t>(SegInfo[Rank].SegStart);
  uintptr_t ptr = reinterpret_cast<uintptr_t>(Ptr);

  if (ptr < Start || ptr > Start + SegInfo[Rank].SegSize) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(ptr - Start);
}

void *MemoryManager::convertRemotetoLocalAddr(void *Ptr, int Rank) {
  uintptr_t RemoteOffset = reinterpret_cast<uintptr_t>(Ptr) -
                           reinterpret_cast<uintptr_t>(getSegmentAddr(Rank));
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(LocalSegStart) +
                                  RemoteOffset);
}

#ifdef DIOMP_ENABLE_CUDA

// CUDA Memory Manager implementation
CUDAMemoryManager::CUDAMemoryManager(gex_TM_t GexTeam, int Mode,
                                     size_t DeviceSegSize)
    : MemoryManager(GexTeam, Mode) {
  this->DeviceSegSize = DeviceSegSize;
  int TargetDevicesNum = omp_get_num_devices();
  if (Mode != 1) {
    LocalRank = MyRank % TargetDevicesNum;
    TargetDevicesNum = 1;
  } else {
    LocalRank = 0;
  }

  DeviceEPs.resize(TargetDevicesNum);
  gex_MK_Create_args_t Args;
  Args.gex_flags = 0;
  Args.gex_class = GEX_MK_CLASS_CUDA_UVA;
  //  args.gex_class = GEX_MK_CLASS_HIP;
  // args.gex_args.gex_class_hip.gex_hipDevice = 0;
  std::vector<gex_MK_t> MkArray(TargetDevicesNum);
  void *LocalPtr = nullptr;
  // Create and bind local segments for each device
  for (int DeviceID = 0; DeviceID < TargetDevicesNum; DeviceID++) {
    gex_EP_t DeviceEP;
    Args.gex_args.gex_class_cuda_uva.gex_CUdevice = DeviceID + LocalRank;
    GASNET_Safe(gex_MK_Create(&MkArray[DeviceID], diompClient, &Args, 0));

    void *DeviceSegAddr = omp_target_alloc(DeviceSegSize, DeviceID + LocalRank);
    LocalPtr = DeviceSegAddr;
    gex_Segment_t DeviceSeg = GEX_SEGMENT_INVALID;
    GASNET_Safe(gex_Segment_Create(&DeviceSeg, diompClient, DeviceSegAddr,
                                   DeviceSegSize, MkArray[DeviceID], 0));

    GASNET_Safe(
        gex_EP_Create(&DeviceEP, diompClient, GEX_EP_CAPABILITY_RMA, 0));
    GASNET_Safe(gex_EP_BindSegment(DeviceEP, DeviceSeg, 0));
    GASNET_Safe(gex_EP_PublishBoundSegment(diompTeam, &DeviceEP, 1, 0));
    DeviceEPs[DeviceID] = DeviceEP;
  }

  // Initialize device segment info
  DeviceSegInfo.resize(RanksNum,
                       std::vector<gex_DeviceSeginfo_t>(TargetDevicesNum));
  for (int Rank = 0; Rank < RanksNum; Rank++) {
    for (int DeviceID = 0; DeviceID < TargetDevicesNum; DeviceID++) {
      DeviceSegInfo[Rank][DeviceID].SegStart = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegRemain = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegSize = 0;
    }
  }

  // Setup CUDA IPC
  if (Mode != 1) {
    CUDACHECK(cudaSetDevice(LocalRank));
    for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
      if (DeviceID == LocalRank)
        continue;
      CUDACHECK(cudaDeviceEnablePeerAccess(DeviceID, 0));
    }

    cudaIpcMemHandle_t IpcHandle;
    CUDACHECK(cudaIpcGetMemHandle(&IpcHandle, LocalPtr));
    IpcHandles.resize(RanksNum);
    for (int i = 0; i < RanksNum; i++) {
      cudaIpcMemHandle_t Handle;
      gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, i, &Handle, &IpcHandle,
                                          sizeof(cudaIpcMemHandle_t), 0));
      IpcHandles[i] = Handle;
    }
    for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
      if (DeviceID == LocalRank)
      {
        PeerPtrs.push_back(LocalPtr);
        continue;
      }
      void *PeerPtr = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(&PeerPtr,
                                     IpcHandles[MyRank - LocalRank + DeviceID],
                                     cudaIpcMemLazyEnablePeerAccess));
      PeerPtrs.push_back(PeerPtr);
    }
    printf("PeerPtrs size = %zu\n", PeerPtrs.size());
  }
}

void *CUDAMemoryManager::deviceAlloc(size_t Size, int DeviceId) {
  if (Mode != 1) {
    DeviceId = 0;
  }

  // Assuming DeviceRemain is a class member variable, initialized to 0
  static const size_t ALIGNMENT = 16; // Example: 16-byte alignment
  static uintptr_t MaxAddr = 0;       // Tracks the maximum allowed address

  // Initialize DeviceRemain if it is uninitialized
  if (DeviceRemain == 0) {
    void *Res = getDeviceSegmentAddr(MyRank, DeviceId);
    if (!Res) {
      THROW_ERROR("Failed to get device segment address");
    }
    DeviceRemain = reinterpret_cast<uintptr_t>(Res);
    MaxAddr = DeviceRemain + DeviceSegSize; // Assume this gets the segment size
  }

  // Ensure the address is properly aligned
  uintptr_t Res = (DeviceRemain + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
  if (Res + Size > MaxAddr) {
    throw std::bad_alloc(); // Allocation exceeds the available memory
  }

  DeviceRemain = Res + Size;
  return reinterpret_cast<void *>(Res);
}

void CUDAMemoryManager::deviceDealloc() {
  DeviceRemain = reinterpret_cast<uintptr_t>(nullptr);
}

void *CUDAMemoryManager::getDeviceSegmentAddr(int Rank, int DeviceID) {
  if (Mode != 1) {
    DeviceID = 0;
  }
  if (DeviceSegInfo[Rank][DeviceID].SegStart == nullptr) {
    gex_EP_Index_t TargetEPIdx = gex_EP_QueryIndex(DeviceEPs[DeviceID]);
    gex_TM_t TargetTM = gex_TM_Pair(DeviceEPs[0], TargetEPIdx);
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(
        TargetTM, Rank, &DeviceSegInfo[Rank][DeviceID].SegStart, nullptr,
        &DeviceSegInfo[Rank][DeviceID].SegSize, 0));
    DeviceSegInfo[Rank][DeviceID].SegRemain =
        DeviceSegInfo[Rank][DeviceID].SegStart;
  }
  return DeviceSegInfo[Rank][DeviceID].SegStart;
}

size_t CUDAMemoryManager::getDeviceOffset(void *Ptr) {
  uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, 0));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return (size_t)LocalOffset;
}

void *CUDAMemoryManager::convertLocaltoRemoteAddr(void *Ptr, int Rank,
                                                  int DeviceId) {
  uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, DeviceId));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId)) +
      LocalOffset);
}

void *CUDAMemoryManager::convertRemotetoLocalAddr(void *Ptr, int Rank,
                                                  int DeviceId) {
  uintptr_t RemoteBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t RemoteOffset = reinterpret_cast<uintptr_t>(Ptr) - RemoteBase;
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(LocalSegStart) +
                                  RemoteOffset);
}

size_t CUDAMemoryManager::getOffset(void *Ptr, int Rank, int DeviceId) {
  // Get segment base address and size
  uintptr_t SegmentBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t TargetPtr = reinterpret_cast<uintptr_t>(Ptr);

  // Check for null pointer
  if (!Ptr || !SegmentBase) {
    return static_cast<size_t>(-1);
  }

  // Check if pointer is within valid segment range
  uintptr_t SegmentEnd = SegmentBase + DeviceSegSize;
  if (SegmentEnd < SegmentBase) { // Check for overflow
    return static_cast<size_t>(-1);
  }

  if (TargetPtr < SegmentBase || TargetPtr >= SegmentEnd) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(TargetPtr - SegmentBase);
}

gex_EP_t CUDAMemoryManager::getEP(int DeviceId) { return DeviceEPs[DeviceId]; }

cudaIpcMemHandle_t CUDAMemoryManager::getIpcHandle(int Rank) {
  return IpcHandles[Rank];
}

#endif // DIOMP_ENABLE_CUDA

#ifdef DIOMP_ENABLE_HIP

// HIP Memory Manager implementation
HIPMemoryManager::HIPMemoryManager(gex_TM_t GexTeam, int Mode,
                                   size_t DeviceSegSize)
    : MemoryManager(GexTeam, Mode) {
  this->DeviceSegSize = DeviceSegSize;
  int TargetDevicesNum = omp_get_num_devices();
  if (Mode != 1) {
    LocalRank = MyRank % TargetDevicesNum;
    TargetDevicesNum = 1;
  } else {
    LocalRank = 0;
  }

  DeviceEPs.resize(TargetDevicesNum);
  gex_MK_Create_args_t Args;
  Args.gex_flags = 0;
  Args.gex_class = GEX_MK_CLASS_HIP;
  // Args.gex_args.gex_class_hip.gex_hipDevice = LocalRank;
  std::vector<gex_MK_t> MkArray(TargetDevicesNum);
  void *LocalPtr = nullptr;
  for (int DeviceID = 0; DeviceID < TargetDevicesNum; DeviceID++) {
    gex_EP_t DeviceEP;
    Args.gex_args.gex_class_hip.gex_hipDevice = DeviceID + LocalRank;
    GASNET_Safe(gex_MK_Create(&MkArray[DeviceID], diompClient, &Args, 0));
    void *DeviceSegAddr = nullptr;
    // omp_target_alloc(DeviceSegSize, DeviceID + LocalRank);

    gex_Segment_t DeviceSeg = GEX_SEGMENT_INVALID;
    GASNET_Safe(gex_Segment_Create(&DeviceSeg, diompClient, DeviceSegAddr,
                                   DeviceSegSize, MkArray[DeviceID], 0));
    LocalPtr = gex_Segment_QueryAddr(DeviceSeg);

    
    GASNET_Safe(gex_EP_Create(&DeviceEP, diompClient, GEX_EP_CAPABILITY_RMA, 0));
    GASNET_Safe(gex_EP_BindSegment(DeviceEP, DeviceSeg, 0));
    GASNET_Safe(gex_EP_PublishBoundSegment(diompTeam, &DeviceEP, 1, 0));
    DeviceEPs[DeviceID] = DeviceEP;
  }

  // Initialize device segment info
  DeviceSegInfo.resize(RanksNum,
                       std::vector<gex_DeviceSeginfo_t>(TargetDevicesNum));
  for (int Rank = 0; Rank < RanksNum; Rank++) {
    for (int DeviceID = 0; DeviceID < TargetDevicesNum; DeviceID++) {
      DeviceSegInfo[Rank][DeviceID].SegStart = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegRemain = nullptr;
      DeviceSegInfo[Rank][DeviceID].SegSize = 0;
    }
  }

  // Setup HIP IPC
  if (Mode != 1) {
    HIPCHECK(hipSetDevice(LocalRank));  
    for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
      if (DeviceID == LocalRank)
        continue;
      HIPCHECK(hipDeviceEnablePeerAccess(DeviceID, 0));
    }
    
    hipIpcMemHandle_t IpcHandle;
    HIPCHECK(hipIpcGetMemHandle(&IpcHandle, LocalPtr));
    IpcHandles.resize(RanksNum);
    for (int i = 0; i < RanksNum; i++) {
      hipIpcMemHandle_t Handle;
      gex_Event_Wait(gex_Coll_BroadcastNB(diompTeam, i, &Handle, &IpcHandle,
                                          sizeof(hipIpcMemHandle_t), 0));
      IpcHandles[i] = Handle;
    }
    for (int DeviceID = 0; DeviceID < fmin(omp_get_num_devices(), RanksNum); DeviceID++) {
      if (DeviceID == LocalRank)
      {
        PeerPtrs.push_back(LocalPtr);
        continue;
      }
      void *PeerPtr = nullptr;
      HIPCHECK(hipIpcOpenMemHandle(&PeerPtr, IpcHandles[MyRank - LocalRank + DeviceID], hipIpcMemLazyEnablePeerAccess));
      PeerPtrs.push_back(PeerPtr);
    }
  }
}

void *HIPMemoryManager::deviceAlloc(size_t Size, int DeviceId) {
  if (Mode != 1) {
    DeviceId = 0;
  }

  // Assuming DeviceRemain is a class member variable, initialized to 0
  static const size_t ALIGNMENT = 16; // Example: 16-byte alignment
  static uintptr_t MaxAddr = 0;       // Tracks the maximum allowed address

  // Initialize DeviceRemain if it is uninitialized
  if (DeviceRemain == 0) {
    void *Res = getDeviceSegmentAddr(MyRank, DeviceId);
    if (!Res) {
      THROW_ERROR("Failed to get device segment address");
    }
    DeviceRemain = reinterpret_cast<uintptr_t>(Res);
    MaxAddr = DeviceRemain + DeviceSegSize; // Assume this gets the segment size
  }

  // Ensure the address is properly aligned
  uintptr_t Res = (DeviceRemain + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
  if (Res + Size > MaxAddr) {
    throw std::bad_alloc(); // Allocation exceeds the available memory
  }

  DeviceRemain = Res + Size;
  return reinterpret_cast<void *>(Res);
}

void HIPMemoryManager::deviceDealloc() {
  DeviceRemain = reinterpret_cast<uintptr_t>(nullptr);
}

void *HIPMemoryManager::getDeviceSegmentAddr(int Rank, int DeviceId) {
  if (Mode != 1) {
    DeviceId = 0;
  }
  if (DeviceSegInfo[Rank][DeviceId].SegStart == nullptr) {
    gex_EP_Index_t TargetEPIdx = gex_EP_QueryIndex(DeviceEPs[DeviceId]);
    gex_TM_t TargetTM = gex_TM_Pair(DeviceEPs[0], TargetEPIdx);
    gex_Event_Wait(gex_EP_QueryBoundSegmentNB(
        TargetTM, Rank, &DeviceSegInfo[Rank][DeviceId].SegStart, nullptr,
        &DeviceSegInfo[Rank][DeviceId].SegSize, 0));
    DeviceSegInfo[Rank][DeviceId].SegRemain =
        DeviceSegInfo[Rank][DeviceId].SegStart;
  }
  return DeviceSegInfo[Rank][DeviceId].SegStart;
}

size_t HIPMemoryManager::getDeviceOffset(void *Ptr) {
  uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, 0));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return (size_t)LocalOffset;
}

void *HIPMemoryManager::convertLocaltoRemoteAddr(void *Ptr, int Rank,
                                                  int DeviceId) {
  uintptr_t LocalBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(MyRank, DeviceId));
  uintptr_t LocalOffset = reinterpret_cast<uintptr_t>(Ptr) - LocalBase;
  return reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId)) +
      LocalOffset);
}

void *HIPMemoryManager::convertRemotetoLocalAddr(void *Ptr, int Rank,
                                                  int DeviceId) {
  uintptr_t RemoteBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t RemoteOffset = reinterpret_cast<uintptr_t>(Ptr) - RemoteBase;
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(LocalSegStart) +
                                  RemoteOffset);
}

size_t HIPMemoryManager::getOffset(void *Ptr, int Rank, int DeviceId) {
  // Get segment base address and size
  uintptr_t SegmentBase =
      reinterpret_cast<uintptr_t>(getDeviceSegmentAddr(Rank, DeviceId));
  uintptr_t TargetPtr = reinterpret_cast<uintptr_t>(Ptr);

  // Check for null pointer
  if (!Ptr || !SegmentBase) {
    return static_cast<size_t>(-1);
  }

  // Check if pointer is within valid segment range 
  uintptr_t SegmentEnd = SegmentBase + DeviceSegSize;
  if (SegmentEnd < SegmentBase) { // Check for overflow
    return static_cast<size_t>(-1);
  }

  if (TargetPtr < SegmentBase || TargetPtr >= SegmentEnd) {
    return static_cast<size_t>(-1);
  }

  return static_cast<size_t>(TargetPtr - SegmentBase);
}

gex_EP_t HIPMemoryManager::getEP(int DeviceId) { return DeviceEPs[DeviceId]; }

hipIpcMemHandle_t HIPMemoryManager::getIpcHandle(int Rank) {
  return IpcHandles[Rank];
}


#endif // DIOMP_ENABLE_HIP

} // namespace diomp