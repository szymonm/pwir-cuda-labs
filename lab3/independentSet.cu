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

#define NUM_VERTICES 256

#define SEQUENTIAL 1

#define POS(i, j) (((j) * NUM_VERTICES) + i)

#define PRINT_EDGES 1

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
 * Independent Set algorithm
 */
__global__ void independentSetCUDA(int *d_adjacencyMatrix, int *d_independentSet, int* d_degrees,
    int* d_marked)
{
    // Fill kernel code
}

int sampleBinomialDistribution(double succesProbability) {
    return (int)((double)rand() / (double)RAND_MAX < succesProbability);
}

void printIndependentSet(int *independentSet) {
    printf("IndependentSet(");
    for (int i = 0; i < NUM_VERTICES; i++) {
        if (independentSet[i] == 1) {
            printf("%d, ", i);
        }
    }
    printf(")\n");
}

void printEdges(int *adjacencyMatrix) {
    printf("Edges: \n");
    for (int i = 0; i < NUM_VERTICES; i++) {
        for (int j = 0; j < NUM_VERTICES; j++) {
            if (adjacencyMatrix[POS(i, j)] == 1) {
                printf("\t(%d, %d)\n", i, j);
            }
        }
    }
}

// Skewed random graph
void randomGraph(int *adjacencyMatrix)
{
    memset(adjacencyMatrix, 0, NUM_VERTICES * NUM_VERTICES);
    int edge;
    for (int i = 0; i < NUM_VERTICES; i++) {
        for (int j = i + 1; j < NUM_VERTICES; j++) {
            double probability = (((double) i) + j) / (4 * NUM_VERTICES);
            edge = sampleBinomialDistribution(probability);
            adjacencyMatrix[POS(i, j)] = edge;
            adjacencyMatrix[POS(j, i)] = edge;
        }
    }
}

bool verifyMaximalIndependentSet(int *adjacencyMatrix, int* independentSet) {
    for (int i = 0; i < NUM_VERTICES; i++) {
        int neighborsInV = 0;
        for (int j = 0; j < NUM_VERTICES; j++) {
            if (i != j && adjacencyMatrix[POS(i, j)] == 1 && independentSet[j] == 1) {
                if (independentSet[i]) {
                    printf("Set is not independent as it contains neighbors %d and %d", i, j);
                    return false;
                }
              neighborsInV++;
            }
        }
        if (independentSet[i] == 0 && neighborsInV == 0) {
            printf("Set is not maximal as %d (not in set) has no neighbors in set\n", i);
            return false;
        }
    }
    return true;
}

void lfIndependentSet(int *adjacencyMatrix, int* independentSet) {
    memset(independentSet, 0, NUM_VERTICES);

    for (int i = 0; i < NUM_VERTICES; i++) {
        bool hasNeighborsInV = false;
        for (int j = 0; j < i; j++) {
            if (adjacencyMatrix[POS(i, j)] == 1 && independentSet[j] == 1) {
                hasNeighborsInV = true;
            }
        }
        if (!hasNeighborsInV) {
            independentSet[i] = 1;
        }
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int main()
{
    srand(1);

    // Allocate host memory for matrices A, B and C
    size_t adjacencyMatrix_mem_size = sizeof(int) * NUM_VERTICES * NUM_VERTICES;
    int *adjacencyMatrix = (int *) malloc(adjacencyMatrix_mem_size);
    int *independentSet = (int *) malloc(NUM_VERTICES * sizeof(int));

    if (adjacencyMatrix == NULL || independentSet == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    randomGraph(adjacencyMatrix);

    // Allocate device memory
    int *d_adjacencyMatrix, *d_degrees, *d_independentSet, *d_marked;

    cudaCheck(cudaMalloc((void **) &d_adjacencyMatrix, adjacencyMatrix_mem_size));
    cudaCheck(cudaMalloc((void **) &d_degrees, sizeof(int) * NUM_VERTICES));
    cudaCheck(cudaMalloc((void **) &d_independentSet, sizeof(int) * NUM_VERTICES));
    cudaCheck(cudaMalloc((void **) &d_marked, sizeof(int) * NUM_VERTICES));

    // copy host memory to device
    cudaCheck(cudaMemcpy(d_adjacencyMatrix, adjacencyMatrix, 
        adjacencyMatrix_mem_size, cudaMemcpyHostToDevice));

    dim3 threads = 1; // fill settings
    dim3 grid = 1; //fill settings

    if (SEQUENTIAL) {
        lfIndependentSet(adjacencyMatrix, independentSet);
    } else {
        printf("Computing result using CUDA Kernel...\n");

        independentSetCUDA<<< grid, threads >>>(d_adjacencyMatrix, d_independentSet, d_degrees, d_marked);

        cudaCheck(cudaPeekAtLastError());
        
        // Copy result from device to host
        cudaCheck(cudaMemcpy(independentSet, d_independentSet, NUM_VERTICES * sizeof(int), 
            cudaMemcpyDeviceToHost));
    }


    printf("Checking computed result for correctness: ");
    
    if (PRINT_EDGES) printEdges(adjacencyMatrix);
    printIndependentSet(independentSet);

    bool correct = verifyMaximalIndependentSet(adjacencyMatrix, independentSet);

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(adjacencyMatrix);
    free(independentSet);
    cudaFree(d_adjacencyMatrix);
    cudaFree(d_degrees);
    cudaFree(d_marked);
    cudaFree(d_independentSet);

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
