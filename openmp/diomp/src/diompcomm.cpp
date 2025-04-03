/*
 * diompcomm.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diompcomm.h"

namespace diomp {

extern "C" void omp_target_setup_diompallocator(int DeviceId, void *Alloc, void *Dealloc);
extern "C" void omp_target_setup_default_allocator(int DeviceNum);

// Base communicator implementations
DiOMPCommunicator::DiOMPCommunicator()
    : Team(diompTeam), DevicesNum(1), Mem(MemManager.get()) {}

void DiOMPCommunicator::bcast(void *Data, size_t Size, int Root) {
  gex_Event_Wait(gex_Coll_BroadcastNB(Team, Root, Data, Data, Size, 0));
}

void DiOMPCommunicator::allreduce(void *Src, void *Dst, size_t Size,
                                  omp_dt_t Dt, omp_op_t Op) {
  gex_Event_Wait(gex_Coll_ReduceToAllNB(Team, Dst, Src, Dt, sizeof(Dt), Size,
                                        Op, NULL, NULL, 0));
}

void DiOMPCommunicator::reduce(void *Src, void *Dst, size_t Size, omp_dt_t Dt,
                               omp_op_t Op, int Root) {
  gex_Event_Wait(gex_Coll_ReduceToOneNB(Team, Root, Dst, Src, Dt, sizeof(Dt),
                                        Size, Op, NULL, NULL, 0));
}

void DiOMPCommunicator::barrier() {
  gex_Event_Wait(gex_Coll_BarrierNB(Team, 0));
}

void DiOMPCommunicator::waitAllRMA() { gex_NBI_Wait(GEX_EC_ALL, 0); }

void DiOMPCommunicator::waitRMA(omp_event_t Ev) { gex_NBI_Wait(Ev, 0); }

void DiOMPCommunicator::lock(int Rank) { return; }

void DiOMPCommunicator::unlock(int Rank) { return; }

void DiOMPCommunicator::get(void *Dest, int Node, void *Src, size_t Nbytes) {
  auto Error = gex_RMA_GetNBI(Team, Dest, Node, Src, Nbytes, GEX_FLAG_NONE);
  if (Error) {
    THROW_ERROR("OpenMP GET Error! Error code is %d", Error);
  }
}

void DiOMPCommunicator::put(int Node, void *Dest, void *Src, size_t Nbytes) {
  auto Error =
      gex_RMA_PutNBI(Team, Node, Dest, Src, Nbytes, GEX_EVENT_DEFER, GEX_FLAG_NONE);
  if (Error != 0) {
    THROW_ERROR("OpenMP PUT Error! Error code is %d", Error);
  }
}

#ifdef OPENMP_ENABLE_DIOMP_DEVICE
// Device communicator implementations
DiOMPDeviceCommunicator::DiOMPDeviceCommunicator(int Mode) {}

DiOMPDeviceCommunicator::~DiOMPDeviceCommunicator() = default;

#ifdef DIOMP_ENABLE_CUDA
// CUDA communicator implementations
CUDAMemoryManager *DiOMPCUDACommunicator::StaticCudaMem = nullptr;

DiOMPCUDACommunicator::DiOMPCUDACommunicator(int Mode) {
  CudaMem = dynamic_cast<CUDAMemoryManager *>(Mem);
  StaticCudaMem = CudaMem;
  this->Mode = Mode;
  if (Mode == 1) {
    DevicesNum = omp_get_num_devices();
    LocalRank = 0;
    for (int DeviceID = 0; DeviceID < DevicesNum; DeviceID++) {
      omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                      (void *)diomp_device_dealloc);
    }
  } else {
    LocalRank = omp_get_rank_num() % omp_get_num_devices();
    DevicesNum = 1;
    for (int DeviceID = 0; DeviceID < omp_get_num_devices(); DeviceID++) {
      omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                      (void *)diomp_device_dealloc);
    }
    omp_set_default_device(LocalRank);
  }

  if (!CudaMem) {
    THROW_ERROR("Memory manager is not a CUDA memory manager");
  }
  if (DevicesNum > 1) {
    NcclStreams = new cudaStream_t[DevicesNum];
    NcclComms = new ncclComm_t[DevicesNum];
  }
}

DiOMPCUDACommunicator::~DiOMPCUDACommunicator() {
  if (DevicesNum > 1) {
    for (int i = 0; i < DevicesNum; i++) {
      if (NcclComms[i] != nullptr) {
        ncclCommDestroy(NcclComms[i]);
      }
      if (NcclStreams[i] != nullptr) {
        cudaStreamDestroy(NcclStreams[i]);
      }
    }
    delete[] NcclStreams;
    delete[] NcclComms;
  } else {
    if (NcclComm != nullptr) {
      ncclCommDestroy(NcclComm);
    }
    if (NcclStream != nullptr) {
      cudaStreamDestroy(NcclStream);
    }
  }
}

void DiOMPCUDACommunicator::initNCCL() {
  ncclUniqueId NcclId;
  if (omp_get_rank_num() == 0) {
    ncclGetUniqueId(&NcclId);
  }

  gex_Event_Wait(
      gex_Coll_BroadcastNB(Team, 0, &NcclId, &NcclId, sizeof(NcclId), 0));

  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(LocalRank));
    NCCLCHECK(ncclCommInitRank(&NcclComm, omp_get_num_ranks(), NcclId,
                               omp_get_rank_num()));
    CUDACHECK(cudaStreamCreate(&NcclStream));
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int DeviceId = 0; DeviceId < DevicesNum; DeviceId++) {
      CUDACHECK(cudaSetDevice(DeviceId));
      NCCLCHECK(ncclCommInitRank(&NcclComms[DeviceId],
                                 omp_get_num_ranks() * DevicesNum, NcclId,
                                 omp_get_rank_num() * DevicesNum + DeviceId));
      CUDACHECK(cudaStreamCreate(&NcclStreams[DeviceId]));
    }
    NCCLCHECK(ncclGroupEnd());
  }
}

void DiOMPCUDACommunicator::waitAllRMA() {
  StreamManager.synchronizeAll();
  StreamManager.clearStreams();
  gex_NBI_Wait(GEX_EC_ALL, 0);
}

void DiOMPCUDACommunicator::dget(void *Dest, int Node, void *Src, size_t Size,
                                 int DstId, int SrcId) {
  if (Mode != 1) {
    int TotalDevices = omp_get_num_devices();

    gex_EP_t LocalEP = CudaMem->getEP(0);
    gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(LocalEP);
    gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);
    void *SrcR = CudaMem->convertLocaltoRemoteAddr(Src, Node, 0);
    if (omp_get_rank_num() / TotalDevices == Node / TotalDevices) {
      int SrcDevice = Node % TotalDevices;
      int DstDevice = LocalRank;
      void *DevicePtr = nullptr;
      cudaStream_t Stream = StreamManager.createStream();
      // cudaIpcMemHandle_t IpcHandle = CudaMem->getIpcHandle(Node);
      // CUDACHECK(cudaIpcOpenMemHandle(&DevicePtr, IpcHandle,
      //                                cudaIpcMemLazyEnablePeerAccess));
      DevicePtr = CudaMem->getPeerPtr(SrcDevice);
      size_t Offset = CudaMem->getOffset(SrcR, Node, 0);
      char *RemotePtr = static_cast<char *>(DevicePtr) + Offset;
      CUDACHECK(cudaMemcpyPeerAsync(Dest, DstDevice, RemotePtr,
                                    SrcDevice, Size, Stream));
      // printf("cudaMemcpyPeerAsync\n");
      return;
    }
    auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, GEX_FLAG_NONE);
    if (Error != 0) {
      THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
    }
    return;
  }
  gex_EP_t LocalEP = CudaMem->getEP(DstId);
  gex_EP_t RemoteEP = CudaMem->getEP(SrcId);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *SrcR = CudaMem->convertLocaltoRemoteAddr(Src, Node, SrcId);
  auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, GEX_FLAG_NONE);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
  }

  return;
}

void DiOMPCUDACommunicator::dput(void *Dst, int Node, void *Src, size_t Size,
                                 int DstId, int SrcId) {
  if (Mode != 1) {
    int TotalDevices = omp_get_num_devices();

    gex_EP_t LocalEP = CudaMem->getEP(0);
    gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(LocalEP);
    gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

    void *SrcR = CudaMem->convertLocaltoRemoteAddr(Src, Node, 0);
    if (omp_get_rank_num() / TotalDevices == Node / TotalDevices) {
      int SrcDevice = Node % TotalDevices;
      int DstDevice = LocalRank;

      cudaStream_t Stream = StreamManager.createStream();

      cudaIpcMemHandle_t IpcHandle = CudaMem->getIpcHandle(Node);
      void *DevicePtr = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(&DevicePtr, IpcHandle,
                                     cudaIpcMemLazyEnablePeerAccess));
    }
  }

  gex_EP_t LocalEP = CudaMem->getEP(SrcId);
  gex_EP_t RemoteEP = CudaMem->getEP(DstId);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *DstR = CudaMem->convertLocaltoRemoteAddr(Dst, Node, DstId);
  auto Error =
      gex_RMA_PutNBI(CommTM, Node, DstR, Src, Size, GEX_EVENT_DEFER, GEX_FLAG_NONE);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device Put Error! Error code is %d", Error);
  }

  return;
}

void DiOMPCUDACommunicator::dbcast(void *Data, size_t Size, omp_device_dt_t Dt,
                                   int Node, int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(DstId));
    NCCLCHECK(
        ncclBcast(Data, Size, (ncclDataType_t)Dt, Node, NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int I = 0; I < DevicesNum; I++) {
    void *RemoteData =
        CudaMem->convertLocaltoRemoteAddr(Data, omp_get_rank_num(), DstId);
    NCCLCHECK(ncclBcast(RemoteData, Size, (ncclDataType_t)Dt,
                        Node * DevicesNum + DstId, NcclComms[I],
                        NcclStreams[I]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int I = 0; I < DevicesNum; I++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[I]));
  }
}

void DiOMPCUDACommunicator::dallreduce(void *Src, void *Dst, size_t Size,
                                       omp_device_dt_t Dt, omp_red_op_t Op,
                                       int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(LocalRank));
    NCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                            NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    NCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                            NcclComms[i], NcclStreams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[i]));
  }
}

void DiOMPCUDACommunicator::dreduce(void *Src, void *Dst, size_t Size,
                                    omp_device_dt_t Dt, omp_red_op_t Op,
                                    int Root, int DstId) {
  if (DevicesNum == 1) {
    CUDACHECK(cudaSetDevice(DstId));
    NCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                         Root, NcclComm, NcclStream));
    CUDACHECK(cudaStreamSynchronize(NcclStream));
    return;
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    NCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                         Root * DevicesNum + DstId, NcclComms[i],
                         NcclStreams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    CUDACHECK(cudaStreamSynchronize(NcclStreams[i]));
  }
}

void *diomp_device_alloc(size_t Size, int DeviceId) {
  return DiOMPCUDACommunicator::cuda_device_alloc(Size, DeviceId);
}

void diomp_device_dealloc() {
  DiOMPCUDACommunicator::cuda_device_dealloc();
  return;
}

#endif // DIOMP_ENABLE_CUDA

#ifdef DIOMP_ENABLE_HIP
// HIP communicator implementations

HIPMemoryManager *DiOMPHIPCommunicator::StaticHIPMem = nullptr;



DiOMPHIPCommunicator::DiOMPHIPCommunicator(int Mode) :streamTracker(streamPool) {
  HipMem = dynamic_cast<HIPMemoryManager *>(Mem);
  StaticHIPMem = HipMem;
  this->Mode = Mode;
  if (Mode == 1) {
    DevicesNum = omp_get_num_devices();
    LocalRank = 0;
    for (int DeviceID = 0; DeviceID < DevicesNum; DeviceID++) {
      omp_target_setup_diompallocator(DeviceID, (void *)diomp_device_alloc,
                                      (void *)diomp_device_dealloc);
    }
  } else {
    LocalRank = omp_get_rank_num() % omp_get_num_devices();
    DevicesNum = 1;
    omp_target_setup_diompallocator(LocalRank, (void *)diomp_device_alloc,
                                    (void *)diomp_device_dealloc);
    omp_set_default_device(LocalRank);
  }

  if (!HipMem) {
    THROW_ERROR("Memory manager is not a HIP memory manager");
  }
  if (DevicesNum > 1) {
    RcclStreams = new hipStream_t[DevicesNum];
    RcclComms = new ncclComm_t[DevicesNum];
  }

}

DiOMPHIPCommunicator::~DiOMPHIPCommunicator() {
  
  if (DevicesNum > 1) {
    for (int i = 0; i < DevicesNum; i++) {
      omp_target_setup_default_allocator(i);
      if (RcclComms[i] != nullptr) {
        ncclCommDestroy(RcclComms[i]);
      }
      if (RcclStreams[i] != nullptr) {
        hipStreamDestroy(RcclStreams[i]);
      }
    }
    delete[] RcclStreams;
    delete[] RcclComms;
  } else {
    omp_target_setup_default_allocator(LocalRank);
    if (RcclComm != nullptr) {
      ncclCommDestroy(RcclComm);
    }
    if (RcclStream != nullptr) {
      hipStreamDestroy(RcclStream);
    }
  }
}

void DiOMPHIPCommunicator::initRCCL() {
  ncclUniqueId RcclId;
  if (omp_get_rank_num() == 0) {
    ncclGetUniqueId(&RcclId);
  }

  gex_Event_Wait(
      gex_Coll_BroadcastNB(Team, 0, &RcclId, &RcclId, sizeof(RcclId), 0));

  if (DevicesNum == 1) {
    HIPCHECK(hipSetDevice(LocalRank));
    RCCLCHECK(ncclCommInitRank(&RcclComm, omp_get_num_ranks(), RcclId,
                               omp_get_rank_num()));
    HIPCHECK(hipStreamCreate(&RcclStream));
  } else {
    RCCLCHECK(ncclGroupStart());
    for (int DeviceId = 0; DeviceId < DevicesNum; DeviceId++) {
      HIPCHECK(hipSetDevice(DeviceId));
      RCCLCHECK(ncclCommInitRank(&RcclComms[DeviceId],
                                 omp_get_num_ranks() * DevicesNum, RcclId,
                                 omp_get_rank_num() * DevicesNum + DeviceId));
      HIPCHECK(hipStreamCreate(&RcclStreams[DeviceId]));
    }
    RCCLCHECK(ncclGroupEnd());
  }
}

void DiOMPHIPCommunicator::waitAllRMA() {
  streamTracker.syncAll();
  gex_NBI_Wait(GEX_EC_ALL, 0);
}

void DiOMPHIPCommunicator::dget(void *Dest, int Node, void *Src, size_t Size,
                                 int DstId, int SrcId) {

  if (Mode != 1) {
    int TotalDevices = omp_get_num_devices();
    if (omp_get_rank_num() / TotalDevices == Node / TotalDevices) {
      int SrcDevice = Node % TotalDevices;
      int DstDevice = LocalRank;
      void *DevicePtr = nullptr;
      DevicePtr = HipMem->getPeerPtr(SrcDevice);
      size_t Offset = HipMem->getDeviceOffset(Src);
      char *RemotePtr = static_cast<char *>(DevicePtr) + Offset;

      hipStream_t stream = streamPool.getStream();
      streamPool.returnStream(stream);

      HIPCHECK(hipMemcpyAsync(Dest, RemotePtr, Size, hipMemcpyDeviceToDevice, stream));
      streamTracker.addStream(stream);
      return;
    }
    void *SrcR = HipMem->convertLocaltoRemoteAddr(Src, Node, 0);
    gex_EP_t LocalEP = HipMem->getEP(0);
    gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(LocalEP);
    gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);
    auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, GEX_FLAG_NONE);
    if (Error != 0) {
      THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
    }
    return;
  }
  gex_EP_t LocalEP = HipMem->getEP(SrcId);
  gex_EP_t RemoteEP = HipMem->getEP(DstId);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  void *SrcR = HipMem->convertLocaltoRemoteAddr(Src, Node, SrcId);
  auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, GEX_FLAG_NONE);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
  }

  return;
}

void DiOMPHIPCommunicator::dput(void *Dest, int Node, void *Src, size_t Size,
                                 int DstId, int SrcId) {
  // if (Mode != 1) {
  //   int TotalDevices = omp_get_num_devices();

  //   gex_EP_t LocalEP = HipMem->getEP(0);  
  //   gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(LocalEP);
  //   gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);

  //   void *SrcR = HipMem->convertLocaltoRemoteAddr(Src, Node, 0);
  //   if (omp_get_rank_num() / TotalDevices == Node / TotalDevices) {
  //     int SrcDevice = Node % TotalDevices;
  //     int DstDevice = LocalRank;

  //     // hipStream_t Stream = StreamManager.createStream();
  //     hipStream_t Stream = Streams[SrcDevice];
  //     hipIpcMemHandle_t IpcHandle = HipMem->getIpcHandle(Node);
  //     void *DevicePtr = nullptr;
  //     HIPCHECK(hipIpcOpenMemHandle(&DevicePtr, IpcHandle,
  //                                   hipIpcMemLazyEnablePeerAccess));
  //     size_t Offset = HipMem->getOffset(SrcR, Node, 0);
  //     char *RemotePtr = static_cast<char *>(DevicePtr) + Offset;
  //     HIPCHECK(hipMemcpyPeerAsync(RemotePtr, DstDevice, Src,
  //                                   SrcDevice, Size, Stream));
  //     return;
  //   } 
  //   auto Error = gex_RMA_GetNBI(CommTM, Dest, Node, SrcR, Size, GEX_FLAG_NONE);
  //   if (Error != 0) {
  //     THROW_ERROR("OpenMP Device Get Error! Error code is %d", Error);
  //   }
  //   return;
  // } 
  gex_EP_t LocalEP = HipMem->getEP(SrcId);
  gex_EP_t RemoteEP = HipMem->getEP(DstId);

  gex_EP_Index_t RemoteIdx = gex_EP_QueryIndex(RemoteEP);
  gex_TM_t CommTM = gex_TM_Pair(LocalEP, RemoteIdx);    

  void *DstR = HipMem->convertLocaltoRemoteAddr(Dest, Node, DstId);
  auto Error =
      gex_RMA_PutNBI(CommTM, Node, DstR, Src, Size, GEX_EVENT_DEFER, GEX_FLAG_NONE);
  if (Error != 0) {
    THROW_ERROR("OpenMP Device Put Error! Error code is %d", Error);
  } 

  return;
}

void DiOMPHIPCommunicator::dbcast(void *Data, size_t Size, omp_device_dt_t Dt,
                                   int Node, int DstId) {
  if (DevicesNum == 1) {
    HIPCHECK(hipSetDevice(DstId));
    RCCLCHECK(ncclBcast(Data, Size, (ncclDataType_t)Dt, Node, RcclComm, RcclStream));
    HIPCHECK(hipStreamSynchronize(RcclStream));
    return;
  }

  RCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    void *RemoteData =
        HipMem->convertLocaltoRemoteAddr(Data, omp_get_rank_num(), DstId);
    RCCLCHECK(ncclBcast(RemoteData, Size, (ncclDataType_t)Dt,
                        Node * DevicesNum + DstId, RcclComms[i],
                        RcclStreams[i]));
  } 
  RCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    HIPCHECK(hipStreamSynchronize(RcclStreams[i]));
  }
}

void DiOMPHIPCommunicator::dallreduce(void *Src, void *Dst, size_t Size,
                                       omp_device_dt_t Dt, omp_red_op_t Op,
                                       int DstId) {
  if (DevicesNum == 1) {
    HIPCHECK(hipSetDevice(DstId)); 
    RCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op,
                            RcclComm, RcclStream));
    HIPCHECK(hipStreamSynchronize(RcclStream));
    return;
  }

  RCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {
    RCCLCHECK(ncclAllReduce(Src, Dst, Size, (ncclDataType_t)Dt,
                            (ncclRedOp_t)Op, RcclComms[i], RcclStreams[i]));
  } 
  RCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    HIPCHECK(hipStreamSynchronize(RcclStreams[i]));
  }
}

void DiOMPHIPCommunicator::dreduce(void *Src, void *Dst, size_t Size,
                                    omp_device_dt_t Dt, omp_red_op_t Op,
                                    int Root, int DstId) {
  if (DevicesNum == 1) {
    HIPCHECK(hipSetDevice(DstId));
    RCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt, (ncclRedOp_t)Op, 
                         Root, RcclComm, RcclStream));
    HIPCHECK(hipStreamSynchronize(RcclStream));
    return;
  }

  RCCLCHECK(ncclGroupStart());
  for (int i = 0; i < DevicesNum; i++) {  
    RCCLCHECK(ncclReduce(Src, Dst, Size, (ncclDataType_t)Dt,
                         (ncclRedOp_t)Op, Root * DevicesNum + DstId,
                         RcclComms[i], RcclStreams[i]));
  } 
  RCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < DevicesNum; i++) {
    HIPCHECK(hipStreamSynchronize(RcclStreams[i]));
  }
}

void *diomp_device_alloc(size_t Size, int DeviceId) {
  return DiOMPHIPCommunicator::hip_device_alloc(Size, DeviceId);
}

void diomp_device_dealloc() {
  DiOMPHIPCommunicator::hip_device_dealloc();
  return;
}





#endif // DIOMP_ENABLE_HIP

#endif // OPENMP_ENABLE_DIOMP_DEVICE

} // namespace diomp