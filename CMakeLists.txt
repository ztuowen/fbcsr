cmake_minimum_required(VERSION 2.8)
project(matrixRep)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")

set(LIB_FILES
        lib/src/CSR.c
        lib/src/CSR_io.c
        lib/src/list.c
        lib/src/UBCSR.c
        lib/src/VBR.c
        lib/src/vector.c)

set(TESTLIB_FILES testlib/src/vector_gen.c)

include_directories(lib)

add_executable(testcases testcases.c ${LIB_FILES} ${TESTLIB_FILES})

cuda_add_executable(cudaSample cudaSample.cu)
cuda_add_cublas_to_target( cudaSample )
