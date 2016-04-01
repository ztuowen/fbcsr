cmake_minimum_required(VERSION 3.5)
project(matrixRep)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

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
