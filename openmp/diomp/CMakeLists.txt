# CMakeLists.txt for diomp

IF(NOT GASNET_LIB)
    MESSAGE(STATUS "GASNET_LIB is not set.")
    set(GASNET_LIB "$HOME/gasnet-install")
ENDIF()

IF(NOT HWLOC_LIB)
    MESSAGE(STATUS "HWLOC_LIB is not set.")
    set(HWLOC_LIB "/lustre/software/hwloc/2.8.0/lib")
ENDIF()

set(GASNET_IBV_INCLUDE "${GASNET_LIB}/include/ibv-conduit/")

include_directories(${GASNET_IBV_INCLUDE})
link_directories(${GASNET_LIB} ${HWLOC_LIB})



add_library(diomp SHARED
  src/diomp.cpp
  src/diompmem.cpp
)
target_link_libraries(diomp hwloc ibverbs rt)

target_include_directories(diomp
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(TARGETS diomp LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
install(FILES include/diomp.h DESTINATION "${LIBOMP_HEADERS_INSTALL_PATH}")


# clang++ test.cpp -ldiomp -isystem  /home/l1065028/llvm-install/include -I $HOME/gasnet-install/include/ibv-conduit/ -I $MPI_HOME/include -L $MPI_HOME/lib -lmpi  -lgasnet-ibv-par -lhwloc -libverbs -lrt -pthread -fopenmp