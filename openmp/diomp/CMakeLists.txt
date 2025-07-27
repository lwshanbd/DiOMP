# CMakeLists.txt for DiOMP

if(OPENMP_DIOMP_GASNET_ROOT)
  set(GASNET_ROOT "${OPENMP_DIOMP_GASNET_ROOT}")
else()
  set(GASNET_ROOT "/home/l1065028/gasnet-arm-cuda")
endif()
set(GASNET_LIB "${GASNET_ROOT}/lib")

if(DEFINED OPENMP_GASNET_API)
  if(OPENMP_GASNET_API STREQUAL "IBV")
    set(GASNET_CONDUIT_IBV ON)
  elseif(OPENMP_GASNET_API STREQUAL "OFI")
    set(GASNET_CONDUIT_OFI ON)
  else()
    message("Invalid GASNET_API value: ${OPENMP_GASNET_API}. Defaulting to IBV.")
    set(GASNET_CONDUIT_IBV ON)
  endif()
else()
  message("OPENMP_GASNET_API not defined. Defaulting to IBV.")
  set(GASNET_CONDUIT_IBV ON)
endif()

if(OPENMP_OFI_ROOT)
  set(OFI_ROOT "${OPENMP_OFI_ROOT}")
else()
  set(OFI_ROOT "/opt/cray/libfabric/2.1")
endif()

if(OPENMP_PMI_ROOT)
  set(PMI_ROOT "${OPENMP_PMI_ROOT}")
else()
  set(PMI_ROOT "/opt/cray/pe/pmi/6.1.15.6")
endif()

if(OPENMP_MPI_ROOT)
  set(MPI_ROOT "${OPENMP_MPI_ROOT}")
else()
  set(MPI_ROOT "/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3")
endif()

if(GASNET_CONDUIT_IBV)
  set(GASNET_CORE_INCLUDE "${GASNET_ROOT}/include/")
  set(GASNET_INCLUDE "${GASNET_ROOT}/include/ibv-conduit/")
  message("Using IBV")
elseif(GASNET_CONDUIT_OFI)
  set(GASNET_CORE_INCLUDE "${GASNET_ROOT}/include/")
  set(GASNET_INCLUDE "${GASNET_ROOT}/include/ofi-conduit/")
  set(GASNET_LIB "${GASNET_ROOT}/lib")
  set(LIBFABRIC_LIB "${OFI_ROOT}/lib64")
  set(LIBPMI_LIB "${PMI_ROOT}/lib")
  set(LIBMPI_LIB "${MPI_ROOT}/lib")
  set(LIBMPI_LIB_GTL "${MPI_ROOT}/../../gtl/lib")
  link_directories(${LIBFABRIC_LIB} ${LIBPMI_LIB} ${LIBMPI_LIB} ${LIBMPI_LIB_GTL} ${GASNET_LIB} /opt/cray/pe/mpich/8.1.31/ofi/crayclang/18.0/lib /opt/cray/pe/mpich/8.1.31/gtl/lib LLVMSupport)
  message("Using OFI")
endif()
include_directories(${GASNET_CORE_INCLUDE} ${GASNET_INCLUDE} ${MPI_INCLUDE})

if(DEFINED OPENMP_HWLOC_LIB)
  set(HWLOC_LIB "${OPENMP_HWLOC_LIB}")
else()
  set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
endif()

if(DEFINED OPENMP_LIBFABRIC_LIB)
  set(LIBFABRIC_LIB "${OPENMP_LIBFABRIC_LIB}")
else()
  set(LIBFABRIC_LIB "/opt/cray/libfabric/1.20.1/lib64")
endif()

# Device support options
option(OPENMP_ENABLE_DIOMP_DEVICE "Enable device support in DiOMP" OFF)
if(OPENMP_ENABLE_DIOMP_DEVICE)
  message(STATUS "Building DiOMP with device support")
  add_definitions(-DOPENMP_ENABLE_DIOMP_DEVICE)
endif()

# CUDA support
option(OPENMP_DIOMP_ENABLE_CUDA "Enable CUDA support in DiOMP" OFF)
if(OPENMP_DIOMP_ENABLE_CUDA)
  set(OPENMP_ENABLE_DIOMP_DEVICE ON)

  message(STATUS "Building DiOMP with CUDA support")
  add_definitions(-DDIOMP_ENABLE_CUDA)
  
  # Find CUDA and NCCL
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME $ENV{CUDA_HOME})
  else()
    set(CUDA_HOME "/hrtc/apps/cuda/12.4.131/aarch64/rocky9")
  endif()
  
  if(DEFINED ENV{NCCL_HOME})
    set(NCCL_HOME $ENV{NCCL_HOME})
  else()
    set(NCCL_HOME "/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/comm_libs/nccl/")
  endif()
  
  include_directories(${CUDA_HOME}/include ${NCCL_HOME}/include)
  link_directories(${CUDA_HOME}/lib64 ${NCCL_HOME}/lib)
endif()

# HIP support 
option(OPENMP_DIOMP_ENABLE_HIP "Enable HIP support in DiOMP" OFF)
if(OPENMP_DIOMP_ENABLE_HIP)
  set(OPENMP_ENABLE_DIOMP_DEVICE ON)

  message(STATUS "Building DiOMP with HIP support")
  add_definitions(-DDIOMP_ENABLE_HIP)
  
  # Define HIP platform (AMD by default)
  option(OPENMP_DIOMP_HIP_PLATFORM_NVIDIA "Use NVIDIA as HIP platform instead of AMD" OFF)
  if(OPENMP_DIOMP_HIP_PLATFORM_NVIDIA)
    add_definitions(-D__HIP_PLATFORM_NVIDIA__)
    message(STATUS "Using NVIDIA as HIP platform")
  else()
    add_definitions(-D__HIP_PLATFORM_AMD__)
    message(STATUS "Using AMD as HIP platform")
  endif()

  if(DEFINED ENV{ROCM_HOME})
    set(ROCM_HOME $ENV{ROCM_HOME})
  elseif(DEFINED ENV{ROCM_PATH})
    set(ROCM_HOME $ENV{ROCM_PATH})
  else()
    set(ROCM_HOME "/hrtc/apps/rocm/5.7.0/aarch64/rocky9")
  endif()

  include_directories(${ROCM_HOME}/include)
  link_directories(${ROCM_HOME}/lib)
  
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../offload/include)

add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
  src/diompcomm.cpp
)

link_directories(${GASNET_LIB})

set(DIOMP_LIBS
  # Base libraries
  -lmpi
  -fopenmp
  -lrt
  -lhwloc
  -pthread
  -lLLVMSupport
  
  # GASNet IBV libraries
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:-lgasnet-ibv-par>
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:-libverbs>
  $<$<BOOL:${GASNET_CONDUIT_IBV}>:--offload-arch=gfx942>

  # GASNet OFI libraries
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:${GASNET_LIB}/libgasnet-ofi-par.a>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lgasnet-ofi-par>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lhugetlbfs>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lfabric>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-L${LIBFABRIC_LIB}>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:-lpmi>
  $<$<BOOL:${GASNET_CONDUIT_OFI}>:--offload-arch=sm_80>
  # CUDA libraries
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lcudart>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lcuda>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_CUDA}>:-lnccl>
  # HIP libraries
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_HIP}>:-lamdhip64>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_HIP}>:-lrccl>
  $<$<BOOL:${OPENMP_DIOMP_ENABLE_HIP}>:-L/p/lustre2/shan4/softwares/diomp/lib/x86_64-unknown-linux-gnu>

  
)
add_definitions(-DGASNETT_USE_GCC_ATTRIBUTE_NOINLINE=1)

target_link_libraries(diomp PUBLIC ${DIOMP_LIBS})

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES 
  include/diomp.h 
  include/diomp.hpp
  include/diompmem.h
  include/diompcomm.h
  DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")

# install(PROGRAMS 
#  src/diompcxx.py
#  src/diompcc.py
#  DESTINATION "${CMAKE_INSTALL_BINDIR}"
#  RENAME diompcxx
# )