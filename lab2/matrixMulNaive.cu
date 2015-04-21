/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define MATRIX_DIM 256

#define POS(i, j) (((i) * MATRIX_DIM) + j)

// CUDA API error checking macro
static void handleError(cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line );
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void matrixMulCUDA(float *C, float *A, float *B)
{
    int posx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int posy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    float res = 0.0;
    for (int i = 0; i < MATRIX_DIM; ++i) {
        res += A[POS(posx, i)] * B[POS(i, posy)];
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[POS(posx, posy)] = res;
}

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = ((double)rand()/(double)RAND_MAX);
    }
}

double computeMatrixMulPos(int row, int column, float* A, float* B) {
  double sum = 0.0;
  for (int i = 0; i < MATRIX_DIM; i++) {
    sum += A[POS(row, i)] * B[POS(i, column)];
  }
  return sum;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int main()
{
    // Allocate host memory for matrices A, B and C
    size_t matrix_mem_size = sizeof(float) * MATRIX_DIM * MATRIX_DIM;
    float *h_A = (float *)malloc(matrix_mem_size);
    float *h_B = (float *)malloc(matrix_mem_size);
    float *h_C = (float *) malloc(matrix_mem_size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    randomInit(h_A, MATRIX_DIM * MATRIX_DIM);
    randomInit(h_B, MATRIX_DIM * MATRIX_DIM);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaCheck(cudaMalloc((void **) &d_A, matrix_mem_size));
    cudaCheck(cudaMalloc((void **) &d_B, matrix_mem_size));
    cudaCheck(cudaMalloc((void **) &d_C, matrix_mem_size));

    // copy host memory to device
    cudaCheck(cudaMemcpy(d_A, h_A, matrix_mem_size, cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(d_B, h_B, matrix_mem_size, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_DIM / threads.x, MATRIX_DIM / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    matrixMulCUDA<<< grid, threads >>>(d_C, d_A, d_B);
    
    cudaCheck(cudaPeekAtLastError());
    

    // Copy result from device to host
    cudaCheck(cudaMemcpy(h_C, d_C, matrix_mem_size, cudaMemcpyDeviceToHost));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-3; // machine zero

    for (int row = 0; row < MATRIX_DIM; row++)
        for (int column = 0; column < MATRIX_DIM; column++) {
            double expected = computeMatrixMulPos(row, column, h_A, h_B);
            if (abs(h_C[POS(row,column)] - expected) > eps) {
                printf("ERROR: position (%d, %d) %f != %f\n", row, 
                    column, h_C[POS(row, column)], expected);
                correct = false;
            }
        }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
