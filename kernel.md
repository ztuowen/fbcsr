# How to implement a new kernel

It consists of three steps:

1. Implement a new block shape.
2. Write a custom kernel in CUDA.
3. Enable it.

## Implement a new block shape

* Function signature: `coo func(int indx,int nelem)`
* existing functions: `lib/src/FBCSR.c`
* coo:
    * .r: row offset
    * .c: column offset

## Write a custom kernel in CUDA

* Function signature: `void func(fbcsr *f, vec *v, vec *r)`
* existing functions: `lib/cuda/FBCSR_krnl.cu`
* r += f\*v
* kernel is called by this entry function`

## Enable it

* append this representation to the respective location of the list(list is built as a stack)
    * CPU representation
    * GPU representation
* Note that GPU representation will require a custom kernel to execute
* See `fbcsrCUDA.cu`
