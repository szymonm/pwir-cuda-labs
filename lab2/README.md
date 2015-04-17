# Lab 2 - matrix multiplication case study

## Naive version

## Using CUDA profiler

## Tiled matrix multiplication (2 points)

It's easy to observe that adjacent elements of result matrix C depend on adjacent elements
of matrices A and B, i.e., `C[1, 1] = A[1,1] * B[1,1] + A[1, 2] * B[2, 1] + ...` and 
`C[1, 2] = A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2] + ...`. Similarly to the Stencil case from
the previous lab, we can benefit from using shared memory to reduce the number of fetching
data from the DRAM.

Tiled matrix multiplication is an algorithm for faster multiplication using shared memory.

Algorithm:

1. Divide matrices A and B into tiles of sizes `groupDim.x * groupDim.y`.
2. For each tile that has elements needed to compute matrix product:
  1. Load that tile to the shared memmory (each thread loads 1 element)
  2. Each thread computes part of the sum from the loaded tile

You can assume that matrix dimensions are multiplications of tiles' dimensions.