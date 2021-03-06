cmake_minimum_required(VERSION 2.8)
project(matrixRep)

find_package(CUDA REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)

if (NOT DEFINED CUDA_cusparse_LIBRARY OR NOT ${CUDA_cusparse_LIBRARY})
    find_library(CUDA_cusparse_LIBRARY NAMES cusparse)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=gnu99")

set(LIB_FILES
        lib/src/CSR.c
        lib/src/CSR_io.c
        lib/src/list.c
        lib/src/UBCSR.c
        lib/src/FBCSR.c
        lib/src/VBR.c
        lib/src/vector.c
        lib/src/timing.c)

set(LIB_CUDA
        lib/cuda/UBCSR.cu
        lib/cuda/CSR.cu
        lib/cuda/vector.cu
        lib/cuda/cuHelper.cu
        lib/cuda/FBCSR.cu
        lib/cuda/FBCSR_krnl.cu)

set(TESTLIB_FILES
        testlib/src/fix_csr.c
        testlib/src/vector_gen.c
        testlib/src/fbcsr_gen.c)

include_directories(lib)
cuda_add_library(cudalib ${LIB_CUDA} ${LIB_FILES})

cuda_add_executable(testcases testcases.c ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(testcases cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(ubcsrCUDA ubcsrCUDA.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(ubcsrCUDA cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(csrCUDA csrCUDA.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(csrCUDA cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(hybCUDA hybCUDA.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(hybCUDA cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(fbcsrHybCUDA fbcsrHybCUDA.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(fbcsrHybCUDA cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(fbcsrShapes fbcsrShapes.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(fbcsrShapes cudalib ${CUDA_cusparse_LIBRARY})

cuda_add_executable(fbcsrCUDA fbcsrCUDA.cu ${TESTLIB_FILES} ${LIB_FILES})
target_link_libraries(fbcsrCUDA cudalib ${CUDA_cusparse_LIBRARY})
