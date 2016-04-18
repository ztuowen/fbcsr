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

* **lib/** libs for different matrix representations
    * **cuda/** cuda related implementations
    * **src/** implementations
    * **prefix.h** global definitions and helpers
    * **list.h** list structure for c
    * **CSR.h** own representation of CSR
    * **VBR.h** VBR and converters between CSR format
    * **UBCSR.h** UBSCR and converters between CSR format
    * **vector.h** some vector functions for easier implementation
* **testlib/** libs related with testing
    * **src/** implementations
    * **test_prefix.h** global definition for tests
    * **vector_gen.h** Random vector generator
* **CMakeLists.txt** CMake makefile
* **testcases.c** Some coherency tests
* **csrCUDA.cu** cuSPARSE CSR
* **hybCUDA.cu** cuSPARSE HYB
* **fbcsrCUDA.cu**
* **fbcsrHybCUDA.cu** FBCSR with leftover using HYB
* **ubcsrCUDA.cu**
* **readme.md** This file
* **kernel.cu** How to implement a custom kernel

## Guidelines

**To implement a new public interface**: 

1. add the function declaration in respective .h file
2. implementation writen in `src/` folder and quote the .h file using `#include"../XXXX.h"`

**To write a new test**:

1. write the test function in `testcases.c` as a `void func(void)`
2. add the test description to `tNames` and the function to `tFuncs`

**To use debug print**:

* `DEBUG_PRINT(...)` works the same way as `fprintf(stderr,...)`
    * will be disabled `#ifndef DEBUG`
    * will be disabled when building as Release

**To implement a custom kernel**

**SEE kernel.md**

