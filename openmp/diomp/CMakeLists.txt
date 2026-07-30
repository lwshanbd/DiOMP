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
  -lrt
  -lhwloc
  -pthread
  
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
  # Local adaptation for Delta AI: dropped --offload-arch=sm_80 here. This is
  # the *link* library list of a host-only shared object, so an offload arch is
  # meaningless (and sm_80 is not this machine's sm_90 anyway). Its real effect
  # is to send the in-tree clang down an offload link path that the partially
  # built runtimes tree cannot yet service: "posix_spawn failed". The ROCm
  # equivalent happens to be satisfiable in-tree, which is why Tioga does not
  # hit this. The IBV/gfx942 line above is left untouched.
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

# -fopenmp was removed from DIOMP_LIBS above: on the link line it makes the
# driver request a bare -lomp, and in a runtimes build libomp is produced in
# this same pass and is not installed yet, so that is unresolvable. Keep it as
# a compile flag and let the in-tree omp target supply the library, which also
# orders diomp after omp.
target_compile_options(diomp PRIVATE -fopenmp)
if(TARGET omp)
  target_link_libraries(diomp PUBLIC omp)
endif()
if(TARGET omptarget)
  target_link_libraries(diomp PUBLIC omptarget)
else()
  # openmp is configured before offload in the runtimes build, so the omptarget
  # target is not visible here. omp_target_alloc and
  # omp_target_setup_diompallocator come from libomptarget.so, which is always
  # loaded in a process that uses DiOMP, so let them resolve at load time
  # instead (LLVM puts -Wl,-z,defs on shared libraries by default).
  target_link_options(diomp PRIVATE "-Wl,-z,undefs")
endif()

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