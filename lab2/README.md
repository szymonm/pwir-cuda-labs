# Lab 2 - matrix multiplication case study

## Naive version

In naive implementation every thread computes one element of resulting matrix (see `matrixMulNaive.cu`).

```cuda
// A, B - square matrices of size MATRIX_DIM * MATRIX_DIM
#define MATRIX_DIM 256

#define POS(i, j) (((i) * MATRIX_DIM) + j)

/**
 * Matrix multiplication: C = A * B
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
```

## CUDA profiler

CUDA Visual Profiler is a tool to profile CUDA applications written in any language.

We can use it to profile `matrixMulNaive` program.

Run command line profiler on the CUDA server using `nvprof ./matrixMulNaive` to obtain summary of the run. 

We can also use it to generate a trace for detailed analysis:
```
nvprof --analysis-metrics -o  mm-analysis.nvprof ./matrixMulNaive
```
Now, download `mm-analysis.nvprof` file and analyse it using visual profiler (`nvvp`).

Can you tell what is the problem with the naive version of matrix multiplication?

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

**Implement the algorithm for square matrices using CUDA platform.**

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

## Extensions

