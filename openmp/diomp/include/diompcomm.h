/*
 * diompcomm.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIOMP_COMM_H
#define DIOMP_COMM_H

#include <cstdint>
#ifndef GASNET_PAR
#define GASNET_PAR
#endif

#define GEX_FLAG_NONE ((gex_Flags_t)0)

#include "diomp.h"
#include "diompmem.h"
#include "tools.h"
#include "omp.h"
#include <cstddef>
#include <gasnet.h>
#include <gasnet_mk.h>
#include <gasnet_tools.h>
#include <gasnetex.h>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>

extern std::unique_ptr<diomp::MemoryManager> MemManager;

namespace diomp {

#ifdef DIOMP_ENABLE_CUDA\

class CUDAStreamManager {
private:
  std::vector<cudaStream_t> Streams;
  std::mutex StreamMutex;

public:
  cudaStream_t createStream() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    cudaStream_t Stream;
    cudaStreamCreate(&Stream);
    Streams.push_back(Stream);
    return Stream;
  }

  void synchronizeAll() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    for (auto Stream : Streams) {
      cudaStreamSynchronize(Stream);
    }
  }

  void clearStreams() {
    std::lock_guard<std::mutex> lock(StreamMutex);
    for (auto Stream : Streams) {
      cudaStreamDestroy(Stream);
    }
    Streams.clear();
  }

  ~CUDAStreamManager() {
    for (auto Stream : Streams) {
      cudaStreamDestroy(Stream);
    }
  }
};

#endif

#ifdef DIOMP_ENABLE_HIP

// class HIPStreamManager {
// private:
//   std::vector<hipStream_t> Streams;
//   std::mutex StreamMutex;


// public:
//   hipStream_t createStream() {
//     std::lock_guard<std::mutex> lock(StreamMutex);
//     hipStream_t Stream;
//     hipStreamCreate(&Stream);
//     Streams.push_back(Stream);
//     return Stream;
//   }

//   void synchronizeAll() {
//     std::lock_guard<std::mutex> lock(StreamMutex);
//     for (auto Stream : Streams) {
//       hipStreamSynchronize(Stream);
//     }
//   }

//   void clearStreams() {
//     std::lock_guard<std::mutex> lock(StreamMutex);
//     for (auto Stream : Streams) {
//       hipStreamDestroy(Stream);
//     }
//     Streams.clear();
//   }

//   ~HIPStreamManager() {
//     for (auto Stream : Streams) {
//       hipStreamDestroy(Stream);
//     }
//   }
// };

#endif

// Base communicator class
class DiOMPCommunicator {
public:
  DiOMPCommunicator();
  virtual ~DiOMPCommunicator() = default;

  // Basic communication operations
  virtual void bcast(void *Data, size_t Size, int Root);
  virtual void allreduce(void *Src, void *Dst, size_t Size, 
                        omp_dt_t Dt, omp_op_t Op);
  virtual void reduce(void *Src, void *Dst, size_t Size,
                     omp_dt_t Dt, omp_op_t Op, int Root);
  virtual void barrier();

  // RMA operations
  virtual void get(void *Dest, int Node, void *Src, size_t Size);
  virtual void put(int Node, void *Dest, void *Src, size_t Size);

  // Device collective operations
  virtual void dbcast(void *Data, size_t Size, omp_device_dt_t Dt, 
                     int Node, int DstId) {
    THROW_ERROR("Device broadcast not supported in base communicator");
  }
  virtual void dallreduce(void *Src, void *Dst, size_t Size,
                         omp_device_dt_t Dt, omp_red_op_t Op, int DstId) {
    THROW_ERROR("Device allreduce not supported in base communicator");
  }
  virtual void dreduce(void *Src, void *Dst, size_t Size,
                      omp_device_dt_t Dt, omp_red_op_t Op, 
                      int Root, int DstId) {
    THROW_ERROR("Device reduce not supported in base communicator");
  }

  // RMA operations for device
  virtual void dget(void *Dest, int Node, void *Src, size_t Size, 
                   int DstId, int SrcId) {
    THROW_ERROR("Device get not supported in base communicator");
  }
  virtual void dput(void *Dest, int Node, void *Src, size_t Size,
                   int DstId, int SrcId) {
    THROW_ERROR("Device put not supported in base communicator");
  }

  // Synchronization operations
  virtual void waitAllRMA();
  virtual void waitRMA(omp_event_t Ev);
  virtual void lock(int Rank);
  virtual void unlock(int Rank);

protected:
  gex_TM_t Team;
  int DevicesNum;
  MemoryManager* Mem;
};

#ifdef OPENMP_ENABLE_DIOMP_DEVICE
// Base device communicator class
class DiOMPDeviceCommunicator : public DiOMPCommunicator {
public:
  DiOMPDeviceCommunicator(int Mode = 1);
  virtual ~DiOMPDeviceCommunicator();

  // Generic device communication interfaces
  virtual void dget(void *Dest, int Node, void *Src, size_t Size, 
                   int DstId, int SrcId) override = 0;
  virtual void dput(void *Dest, int Node, void *Src, size_t Size,
                   int DstId, int SrcId) override = 0;

protected:
  int DevicesNum;
  int Mode;
};

#ifdef DIOMP_ENABLE_CUDA
// CUDA device communicator class
class DiOMPCUDACommunicator : public DiOMPDeviceCommunicator {
public:
  DiOMPCUDACommunicator(int Mode = 1);
  ~DiOMPCUDACommunicator();

  // CUDA specific implementations
  void dget(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;
  void dput(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;

  void waitAllRMA() override;

  // CUDA collective operations
  void dbcast(void *Data, size_t Size, omp_device_dt_t Dt, 
              int Node, int DstId) override;
  void dallreduce(void *Src, void *Dst, size_t Size,
                  omp_device_dt_t Dt, omp_red_op_t Op, int DstId) override;
  void dreduce(void *Src, void *Dst, size_t Size,
               omp_device_dt_t Dt, omp_red_op_t Op, 
               int Root, int DstId) override;

  static void *cuda_device_alloc(size_t Size, int DeviceId){
    return StaticCudaMem->deviceAlloc(Size, DeviceId);
  }
  static void cuda_device_dealloc(){
    StaticCudaMem->deviceDealloc();
  }
  // static void *diomp_device_alloc(size_t Size, int DeviceId);
  // static void diomp_device_dealloc();

  void initNCCL();

private:

  int LocalRank = 0;

  CUDAStreamManager StreamManager;
  CUDAMemoryManager* CudaMem;
  static CUDAMemoryManager* StaticCudaMem;

  ncclComm_t NcclComm;
  cudaStream_t NcclStream;
  // Per process multiple devices
  cudaStream_t *NcclStreams;
  ncclComm_t *NcclComms;
};
#endif

#ifdef DIOMP_ENABLE_HIP

class StreamPool {
private:
    std::vector<hipStream_t> streams;
    std::queue<hipStream_t> availableStreams;
    std::mutex mutex;

public:
    hipStream_t getStream() {
        std::lock_guard<std::mutex> lock(mutex);
        if (availableStreams.empty()) {
            hipStream_t newStream;
            HIPCHECK(hipStreamCreate(&newStream));
            streams.push_back(newStream);
            return newStream;
        }
        hipStream_t stream = availableStreams.front();
        availableStreams.pop();
        return stream;
    }

    void returnStream(hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        availableStreams.push(stream);
    }

    ~StreamPool() {
        for (auto stream : streams) {
            hipStreamDestroy(stream);
        }
    }
};


class ActiveStreamTracker {
private:
    static constexpr int MAX_ACTIVE_STREAMS = 16;
    hipStream_t activeStreams[MAX_ACTIVE_STREAMS];
    int streamCount = 0;
    std::mutex mutex;
    StreamPool& streamPool; 

public:
    ActiveStreamTracker(StreamPool& pool) : streamPool(pool) {}

    void addStream(hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        if (streamCount < MAX_ACTIVE_STREAMS) {
            activeStreams[streamCount++] = stream;
        } else {
            syncBatch(MAX_ACTIVE_STREAMS/2);
            activeStreams[streamCount++] = stream;
        }
    }

    void syncBatch(int count) {
        int syncCount = std::min(count, streamCount);
        for (int i = 0; i < syncCount; i++) {
            HIPCHECK(hipStreamSynchronize(activeStreams[i]));
            streamPool.returnStream(activeStreams[i]);
        }
        for (int i = 0; i < streamCount - syncCount; i++) {
            activeStreams[i] = activeStreams[i + syncCount];
        }
        streamCount -= syncCount;
    }

    void syncAll() {
        syncBatch(streamCount);
    }
};



// HIP device communicator class
class DiOMPHIPCommunicator : public DiOMPDeviceCommunicator {
public:
  DiOMPHIPCommunicator(int Mode = 1);
  ~DiOMPHIPCommunicator();

  // HIP specific implementations
  void dget(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;
  void dput(void *Dest, int Node, void *Src, size_t Size,
            int DstId, int SrcId) override;

  void waitAllRMA() override;

  // HIP collective operations
  void dbcast(void *Data, size_t Size, omp_device_dt_t Dt, 
              int Node, int DstId) override;
  void dallreduce(void *Src, void *Dst, size_t Size,
                  omp_device_dt_t Dt, omp_red_op_t Op, int DstId) override;
  void dreduce(void *Src, void *Dst, size_t Size,
               omp_device_dt_t Dt, omp_red_op_t Op, 
               int Root, int DstId) override;

  static void *hip_device_alloc(size_t Size, int DeviceId){
    return StaticHIPMem->deviceAlloc(Size, DeviceId);
  }
  static void hip_device_dealloc(){
    StaticHIPMem->deviceDealloc();
  }

  void initRCCL();


private:

  int LocalRank = 0;
  // // HIP specific members
  StreamPool streamPool;
  ActiveStreamTracker streamTracker;
  HIPMemoryManager* HipMem;
  static HIPMemoryManager* StaticHIPMem;

  ncclComm_t RcclComm;
  hipStream_t RcclStream;
  // Per process multiple devices
  hipStream_t *RcclStreams;
  ncclComm_t *RcclComms;
};
#endif

#endif // OPENMP_ENABLE_DIOMP_DEVICE
} // namespace diomp

#endif // DIOMP_COMM_H