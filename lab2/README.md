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

**Implement the algorithm using CUDA platform.**

You can assume that matrix dimensions are multiplications of tiles' dimensions.

### Example
When `A = B`, values of `C` in red tile `(2, 2)` depend only on tiles in grey (dark blue element depends on green elements).
![Observation](https://raw.githubusercontent.com/szymonm/pwir-cuda-labs/master/lab2/graphics/tiledMM0.png)

**Step 1** Download tiles `(0, 2)` and `(2, 0)` to shared memory and compute parts of sums in `(2, 2)` from this tiles.

![Step 1](https://raw.githubusercontent.com/szymonm/pwir-cuda-labs/master/lab2/graphics/tiledMM1.png "")

**Step 2** Do the same with tiles `(1, 2)` and `(2, 1)`.

![Step 2](https://raw.githubusercontent.com/szymonm/pwir-cuda-labs/master/lab2/graphics/tiledMM2.png)

**Steps 3 and 4** are analogous.

![Step 3](https://raw.githubusercontent.com/szymonm/pwir-cuda-labs/master/lab2/graphics/tiledMM3.png)
![Step 4](https://raw.githubusercontent.com/szymonm/pwir-cuda-labs/master/lab2/graphics/tiledMM4.png)
