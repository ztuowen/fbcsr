# Work in progress

In currently trying to build the code for VBR, as UBSCR seemingly depends on this representation

This might take longer than expected

# File structure

Only a whole bunch of files floating around

* /
    * prefix.h `global definitions and helpers`
    * list.c/.h `list structure for c` thinking there might be a standard library somewhere...
    * CSR.c/.h `own representation of CSR`
    * VBR.c/.h `VBR and converters between CSR format`
    * UBCSR.c/.h `UBSCR and converters between CSR format`
    * vector.c/.h `some vector functions for easier implementation`

**Needs**

* Debuging & Test -- a lots of it
* Main file

# To compile

1. clone this repository using git.
2. build using cmake
    1. `cd` to source directory
    1. `mkdir Release`
    2. `cd Release`
    3. `cmake -DCMAKE_BUILD_TYPE=Release ..`
    4. `make`

**Can also change Release for Debug**
