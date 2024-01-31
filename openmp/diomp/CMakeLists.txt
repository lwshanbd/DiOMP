# CMakeLists.txt for diomp

set(GASNET "/gpfs/projects/ChapmanGroup/bshan/gasnet-install")
set(GASNET_IBV_INCLUDE "${GASNET}/include/ibv-conduit/")
set(GASNET_LIB "${GASNET}/lib")
set(HWLOC_LIB "/gpfs/software/hwloc/xeonmax/2.9.2/lib")

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