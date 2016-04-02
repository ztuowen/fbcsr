# To compile

1. clone this repository using git.
2. build using cmake
    1. `cd` to source directory
    1. `mkdir Release`
    2. `cd Release`
    3. `cmake -DCMAKE_BUILD_TYPE=Release ..`
    4. `make`

**Can also change Release for Debug**

# File structure

* lib
    * prefix.h `global definitions and helpers`
    * list.c/.h `list structure for c` thinking there might be a standard library somewhere...
    * CSR.c/.h `own representation of CSR`
    * VBR.c/.h `VBR and converters between CSR format`
    * UBCSR.c/.h `UBSCR and converters between CSR format`
    * vector.c/.h `some vector functions for easier implementation`
* testlib `libs related with testing`
    * test_prefix.h `global definition for tests`
    * vector_gen.c/.h `Random vector generator`
* CMakeLists.txt `CMake makefile`
* testcases.c `Some coherency tests`
* readme.md `This file`
* cudaSample.cu `A simple device query CUDA sample to showcase the build system for CUDA`

# Using CUDA

Check `cudaSample.cu` & `CMakeLists.txt`
