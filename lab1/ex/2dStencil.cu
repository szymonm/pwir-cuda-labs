#include <stdio.h>

// We assume that size are divisible
#define RADIUS        2
#define BLOCK_SIZE_X    16 
#define BLOCK_SIZE_Y    16 
#define MATRIX_WIDTH (128)
#define MATRIX_HEIGHT (128)
#define NUM_ELEMENTS (MATRIX_HEIGHT * MATRIX_WIDTH)

// CUDA API error checking macro
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
  }

__global__ void stencil_2d(int *in, int *out) 
{
    /*

      Fill kernel code!
    
    */
}

int main()
{
  unsigned int i, j;
  int h_in[NUM_ELEMENTS], h_out[NUM_ELEMENTS];
  int *d_in, *d_out;

  // To access element (i, j) of the matrix, use h_in[i + j * MATRIX_WIDTH]
  // Initialize host data
  for (i = 0; i < (NUM_ELEMENTS); ++i)
    h_in[i] = 1;

  // Allocate space on the device
  cudaCheck( cudaMalloc( &d_in, NUM_ELEMENTS * sizeof(int)) );
  cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

  // Copy input data to device
  cudaCheck( cudaMemcpy( d_in, h_in, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice) );

  dim3 blocks = dim3(MATRIX_WIDTH / BLOCK_SIZE_X, MATRIX_HEIGHT / BLOCK_SIZE_Y, 1);
  dim3 threads = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

  stencil_2d<<< blocks, threads >>> (d_in, d_out);

  cudaCheck(cudaPeekAtLastError());

  cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );

  bool correct = true;
  // Verify results (inclusion-exclusion principle)
  for (j = 0; j < MATRIX_WIDTH; ++j) {
    for (i = 0; i < MATRIX_HEIGHT; ++i) {
      int expected = (2 * RADIUS + 1) * (2 * RADIUS + 1) - 
        (j < RADIUS ? (RADIUS - j) * (2 * RADIUS + 1) : 0) -
        (i < RADIUS ? (RADIUS - i) * (2 * RADIUS + 1) : 0) +
        ((j < RADIUS && i < RADIUS) ? (RADIUS - j) * (RADIUS - i) : 0) -
        (j > MATRIX_WIDTH - RADIUS - 1 ? (j + RADIUS + 1 - MATRIX_WIDTH) * (2 * RADIUS + 1) : 0) -
        (i > MATRIX_HEIGHT - RADIUS - 1 ? (i + RADIUS + 1 - MATRIX_HEIGHT) * (2 * RADIUS + 1) : 0) +
        (j > MATRIX_WIDTH - RADIUS - 1 && i > MATRIX_HEIGHT - RADIUS - 1 ?
         (j + RADIUS + 1 - MATRIX_WIDTH) * (i + RADIUS + 1 - MATRIX_HEIGHT) : 0) + 
        (j < RADIUS && i > MATRIX_HEIGHT - RADIUS - 1 ? 
         (RADIUS - j) * (i + RADIUS + 1 - MATRIX_HEIGHT) : 0) + 
        (i < RADIUS && j > MATRIX_WIDTH - RADIUS - 1 ?
         (RADIUS - i) *  (j + RADIUS + 1 - MATRIX_WIDTH) : 0);
      if (h_out[j + i * MATRIX_WIDTH] != expected) {
        printf("Element h_out[%d + %d * MATRIX_WIDTH] == %d != %d\n", j, i, h_out[j + i * MATRIX_WIDTH], expected);
        correct = false;
      }
    }
  }

  if (correct)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}

