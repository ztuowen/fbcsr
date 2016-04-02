cmake_minimum_required(VERSION 2.8)
project(matrixRep)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -std=c99")

set(SOURCE_FILES
        lib/CSR.c
        lib/CSR_io.c
        lib/list.c
        lib/UBCSR.c
        lib/VBR.c
        lib/vector.c)

set(TEST_FILES testlib/vector_gen.c)

include_directories(lib)

add_executable(testcases testcases.c ${SOURCE_FILES} ${TEST_FILES})
cuda_add_executable(cudaSample cudaSample.cu)
CUDA_ADD_CUBLAS_TO_TARGET( cudaSample )
