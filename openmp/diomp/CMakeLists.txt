# CMakeLists.txt for diomp

IF(NOT GASNET_ROOT)
    MESSAGE(STATUS "GASNET_LIB is not set.")
    set(GASNET_ROOT "/home/l1065028/gasnet-cuda")
ENDIF()

IF(NOT HWLOC_LIB)
    MESSAGE(STATUS "HWLOC_LIB is not set.")
    set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
ENDIF()

set(GASNET_IBV_INCLUDE "${GASNET_ROOT}/include/ibv-conduit/" "/hrtc/apps/cuda/11.2.152/x86_64/lib")
set(GASNET_LIB "${GASNET_ROOT}/lib")
set(CUDA_HOME "/hrtc/apps/cuda/11.2.152/x86_64")
include_directories(${GASNET_IBV_INCLUDE})
link_directories(${GASNET_LIB} ${HWLOC_LIB})


add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
)
target_link_libraries(diomp -lcuda -lcudart -lmpi
-lrt -lgasnet-ibv-par -lhwloc -pthread -libverbs)

target_include_directories(diomp
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES include/diomp.h include/diomp.hpp DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")
install(PROGRAMS src/diompcxx.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME diompcxx)
  install(PROGRAMS src/diompcc.py
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  RENAME diompcc)  



# clang++ test.cpp -ldiomp -isystem  /home/l1065028/llvm-install/include -I $HOME/gasnet-install/include/ibv-conduit/ -I $MPI_HOME/include -L $MPI_HOME/lib -lmpi  -lgasnet-ibv-par -lhwloc -libverbs -lrt -pthread -fopenmp