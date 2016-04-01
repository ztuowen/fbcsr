cmake_minimum_required(VERSION 3.5)
project(matrixRep)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        lib/CSR.c
        lib/CSR.h
        lib/list.c
        lib/list.h
        lib/prefix.h
        lib/UBCSR.c
        lib/UBCSR.h
        lib/VBR.c
        lib/VBR.h
        lib/vector.c
        lib/vector.h)

add_executable(matrixRep ${SOURCE_FILES})