/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublas.h>

/* Matrix size */
// #define N  (275)
#define N  (512)

#define BLOCKSZ 32
#define THDWORK 8
#define BLSZ 32
#define BLOCK


/* Host implementation of a simple version of strsm */
static void simple_strsm(int n, float alpha, const float *A, const float *B, float *B_out)
{

    int i;
    int j;
    int k;

    for (j = 0; j < n; ++j)
      for (i = 0; i < n; ++i) {
        if (alpha == 0.0f) B_out[j*n+i] = 0;
        else B_out[j*n+i] = B[j*n+i];
    }

      for (j = 0; j < n; ++j) {
        for (k = 0; k < n; ++k) {
	  if (B_out[j*n+k] != 0.0f) { 
             for (i = k+1; i < n; ++i) {
                B_out[j*n+i] -= A[k * n + i] * B_out[j * n + k];
            }
          }
        }
    }
}
/*
for (k = 0; k < n; ++k) {
for (j = 0; j < n; ++j) {
   for (i = k+1; i < n; ++i) {
                B_out[j*n+i] -= A[k * n + i] * B_out[j * n + k];
            }
          }
        }
    }
*/
__global__ void strsm_set(int n, float alpha, float *B)
{
    int x = BLOCKSZ*THDWORK*blockIdx.x+threadIdx.x;
    int y = (BLOCKSZ*blockIdx.y+threadIdx.y)*n;
    int i;
    if (y>=n)
        return;
    for (i=0;i<THDWORK && x<n;++i,x+=BLOCKSZ)
        B[y+x] = 0;
}

__global__ void strsm_krnl_blocked(int n,float alpha,const float*A,float *B)
{
    // The real x (c + tid.x), y
    int x,y = (blockIdx.x*BLSZ+threadIdx.y);
    // The "real" k (k - c)
    int k;
    // BCache, ACache
    __shared__ float bc[BLSZ][BLSZ];
    __shared__ float ac[BLSZ][BLSZ];
    // The starting point of a warp work
    int c;
    float yk;
    const float* An = A;
    int bnd;
    B=B+y*n;
    for (c=0;(c<n && (x=c+threadIdx.x)<n);c+=BLSZ)
    {
        bnd=min(BLSZ,n-c);
        if (threadIdx.y<bnd)
            An=A+(c+threadIdx.y)*n;
        // This copies everything into cache
        ac[threadIdx.y][threadIdx.x] = An[x];
        __syncthreads(); // Protect ACache 0 {
        if (y<n) {
            bc[threadIdx.y][threadIdx.x] = B[x]; // Don't have to sync in a warp
            {
                int x = threadIdx.x,y = threadIdx.y; // This is only a shortcut, don't ask why I'm lazy
                for (k=0;k<bnd && x>k;++k)
                    bc[y][x] = bc[y][x] - ac[k][x]*bc[y][k];
            }
            // copies everything back
            B[x] = bc[threadIdx.y][threadIdx.x];
        }
        __syncthreads(); // } Protect ACache 0
        // Fixes all points after
        x = x+BLSZ;
        for (;x<n;x+=BLSZ) {
            // Load the next workset into ACache
            ac[threadIdx.y][threadIdx.x] = An[x];
            __syncthreads(); // Protect ACache 1 {
            if (y<n) {
                yk=B[x]; // copy into register
                for (k=0;k<bnd;++k)
                    yk = yk - ac[k][threadIdx.x]*bc[threadIdx.y][k];
                B[x]=yk; // From register
            }
            __syncthreads(); // } 1 Protect ACache
        }
    }
}

void device_strsm(int n,float alpha,const float*A,float *B)
{
    if (alpha==0.0f)
    {
        int tilex = BLOCKSZ*THDWORK, tiley=BLOCKSZ;
        dim3 grid((n+tilex-1)/tilex,(n+tiley-1)/tiley),block(BLOCKSZ,BLOCKSZ);
        strsm_set<<<grid,block>>>(n,alpha,B);
        return;
    }
#ifdef BLOCK
    // This one is the best performing one, ~350x speedup
    {
        dim3 grid((n+BLSZ-1)/BLSZ),block(BLSZ,BLSZ);
        strsm_krnl_blocked<<<grid,block>>>(n,alpha,A,B);
    }
#endif
}

/* Main */
int main(int argc, char** argv)
{    
    cublasStatus status;
    float* h_A;
    float* h_B;
    float* h_B_out;
    float* h_B_ref;
    float* d_A = 0;
    float* d_B = 0;
    float alpha = 1.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
    cudaEvent_t start_event, stop_event;
    float elapsed_time_seq, elapsed_time_cublas, elapsed_time_gpu;

    /* Initialize CUBLAS */
    printf("simpleSTRSM test running..\n");

    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for the matrices */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    h_B = (float*)malloc(n2 * sizeof(h_B[0]));
    if (h_B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    h_B_out = (float*)malloc(n2 * sizeof(h_B_out[0]));
    if (h_B_out == 0) {
        fprintf (stderr, "!!!! host memory allocation error (B_out)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++) {
        h_A[i] = rand() / (float)RAND_MAX; h_A[i] = 0.5 * h_A[i] - 0.25f;
        h_B[i] = rand() / (float)RAND_MAX; h_B[i] = 0.5 * h_B[i] - 0.25f;
    }

    /* Allocate device memory for the matrices */
    status = cublasAlloc(n2, sizeof(d_A[0]), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    status = cublasAlloc(n2, sizeof(d_B[0]), (void**)&d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    // time sequential execution time
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);  

    simple_strsm(N, alpha, h_A, h_B, h_B_out);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_seq,start_event, stop_event );

      h_B_ref = h_B_out;  // pointer to output

    // time CUBLAS call
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);   

    cublasStrsm('l' /* left operator */, 'l' /* lower triangular */, 'N' /* not transposed */, 'u' /* unit triangular? */, N, N, alpha, d_A, N, d_B, N);

    cudaThreadSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_cublas,start_event, stop_event);
    
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);   
    // gpustrsm<<<>>>
    device_strsm(N,alpha,d_A,d_B);

    cudaThreadSynchronize();
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time_gpu,start_event, stop_event );


    /* Allocate host memory for reading back the result from device memory */
    h_B = (float*)malloc(n2 * sizeof(h_B[0]));
    if (h_B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_B[0]), d_B, 1, h_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device access error (read B)\n");
        return EXIT_FAILURE;
    }

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;
    for (i = 0; i < n2; i++) {
        diff = h_B_ref[i] - h_B[i];
        error_norm += diff * diff;
        ref_norm += h_B_ref[i] * h_B_ref[i];
	//	fprintf(stderr, "diff %d = %f\n", i, diff); 
    } 
    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);
    //	fprintf(stderr, "error_norm = %f, ref_norm = %f\n", error_norm, ref_norm); 
    if (fabs(ref_norm) < 1e-7) {
        fprintf (stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }
   printf( "STRSM Test %s\n", (error_norm / ref_norm < 1e-4f) ? "PASSED" : "FAILED");

   // Print results
   printf("Sequential Time: %.2f msec\nCUBLAS Time: %.2f msec\nYour Time: %.2f msec\nCUBLAS Speedup = %.2f\nYour Speedup = %2f\n", elapsed_time_seq, 
	  elapsed_time_cublas, elapsed_time_gpu, 
          elapsed_time_seq/elapsed_time_cublas,
	  elapsed_time_seq/elapsed_time_gpu);

    /* Memory clean up */
    free(h_A);
    free(h_B);
    status = cublasFree(d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }
    status = cublasFree(d_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        if (!strcmp(argv[1], "-noprompt") ||
            !strcmp(argv[1], "-qatest") ) 
        {
            return EXIT_SUCCESS;
        }
    } 
    else
    {
        printf("\nPress ENTER to exit...\n");
        getchar();
    }

    return EXIT_SUCCESS;
}
