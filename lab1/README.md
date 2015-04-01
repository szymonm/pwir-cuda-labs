# Lab 1

## Introduction

**Compute Unified Device Architecture (CUDA)** is a parallel computing plaform and programming model created by NVIDIA to allow use of GPUs for general purpose processing (not exclusively graphics).

Unlike traditional CPUs, GPUs are processors specialized for compute-intense, high parallel computation. More specifically, GPU is particullary efective for problems that can be solved by data-parallel computations, i.e., programs that run the same code in parallel on different pieces of data. On the other hand, GPUs are not the best choice for realization of compuations that pose sophistacated flow control or operate on large amounts of data. To this end, CUDA platform may be viewed as a realization of PRAM model (but the differences we cover later).

An alternative to CUDA is OpenCL - a standard supported by many computer companies (Apple, Intel, AMD, Qualcomm). It was designed as a framework for writing code on various heterogenous platforms. Although it is far more general than CUDA (suited only for NVIDIA devices), it is also more 

### Applications

There are following ways of using CUDA supported GPU:

1. Middleware libraries for popular languages (examples include cuBLAS, cuFFT, Thrust, CULA) that hide details of using GPU behind normal function calls (C++, Java, Python).

2. OpenACC compiler directives that are injected in normal code to indicate fragments of the program that should be parallelized and run on the GPU.

3. Direct GPU programming using C/C++ based language that is executed on the GPU processors.

In the labs we cover only the last possibility.

### Programming model

CUDA program consists of serial code performed on the host (CPU and computer memory) and parallel code performed by GPU’s processors called kernel. Communication between host and device (GPU) occurs via data transfer between computer memory and device main memory. In the simplest case program copies data from host memory to device’s, runs kernel code on the device and then copies results from device memory to main program memory.

### Terminology

**Host**- the CPU and its memory

**Device**- the GPU and its memory

**Kernel** - code that is performed by a single GPU processor

**Multiprocessor (Streaming Multiprocessor)**- a group of GPU processors that can synchronize between each other. They usually share some part of the memory and share cache of global device memory. Current GPUs have few SMs.

**Thread**- a single instance of executing kernel

**Work-group (block)** - a logical group of threads working on a single multiprocessor in parallel.

## Programming using CUDA - the basics

### Memory model

GPU has following memory types:

* **global device memory** (read/write for all threads, cached, high latency (200 cycles compared to 4 cycles of aritmetic operation), relatively large (GTX 470 - 1G))

* **shared block memory** (read/write for threads in the block, fast but small - GTX 470 - 48 KB)

* **thread private memory** (read/write for the thread, very fast but small - like processor registers)

* **constants memory** (read for all threads, optimized for cuncurrent read, size: 64 KB)

* **textures memory** (read only, offers different addressing, filtering for specific data)

Figure below shows a rough comparison of CPU and GPU memory structures.

![image alt text](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-devotes-more-transistors-to-data-processing.png)

### Kernel

Kernels are written in CUDA C a subset of C. When called, kernels are executed multiple times by many threads. Let’s analyze simple example below that demonstrates a kernel and calling it from the host code. The function computes sum of two vectors.

```cuda
// Kernel definition - we prepend it with keyword __global__
__global__ void VecAdd(float* A, float* B, float* C) { 
  int i = threadIdx.x; // we can extract thread Id
  C[i] = A[i] + B[i];  // single thread sums only position equal to its thread Id
} 
int main() { 
  ...  
  // Kernel invocation with N threads 
  VecAdd<<<1, N>>>(A, B, C); // A, B, C are addresses of vectors in device memory
  ... 
}
```

### Thread hierarchy

Inside a block threads are identified by a threadIdx - a 3 dimensional vector (accessing dimensions by x, y and z). However, programmer may use only one or two dimensions (like in the example above).

The number of threads per block is limited (since they must fit into single MP). On most current GPU the limit is 1024. However, the threads can be executed by multiple blocks, which number is practically unlimited. Like threads blocks are organized in one, two or tree-dimensional space. 

The number of threads per block and blocks is specified between `<<< … >>>` operators used when calling kernel function. The arguments can be of type int (for one-dimensional grid) or dim3 (for two- or tree-dimensional).

## Exercises

You can use any CUDA supported graphics card. Most of recent NVIDIA cards (even on notebooks) support it. 

For instructions on how to install CUDA toolkit visit: [http://docs.nvidia.com/cuda/index.html#getting-started-guides](http://docs.nvidia.com/cuda/index.html#getting-started-guides)

CUDA toolkit should be already installed on lab computers and on nvidia1 and nvidia2 hosts. The installation directory on nvidia hosts is `/usr/local/cuda-7.0/`, in labs it is `/opt/cuda-7.0/`.

We use nvcc CUDA compiler, see Makefiles of examples for details.

### DeviceQuery

Copy CUDA programs’ samples to your local directory using `cuda-install-sample-7.0.sh` script, which can be found in the installation directory under bin subdirectory.

Compile the code of `1_Utilities/deviceQuery` example using make. Run the program deviceQuery to verify that the CUDA toolkit and drivers work. You should be able to read specification of the GPU. Check how many SMs is in the card you use.

### 1d Stencil

1d Stencil of radius `D > 1` is a function on vector `X` to obtain vector `Y` of the same size such that `Y[i] = X[i - D] + X[i - D + 1] + … + X[i + D]`, where index addition is modulo `X`’s length.

1. Analyze code in `lab1/1dstencil1.cu`. Add code to measure execution time of kernels (you can use NVIDIA events for this (see [here](http://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/))). Try experimenting with block number and threads per block to improve speed. Which configuration is best? Why? What if we increase the number of elements in the vector?

2. Compare and run codes in `1dStencil1.cu` and `1dStencil2.cu` files. Can you explain why second version is faster?

3. Delete the line `__syncthreads()` in `1dStencil2.cu`. What has changed in the result? Can you explain what `__syncthreads()` does?

### 2d Stencil (2 points)

Write your solution to 2d Stencil problem using CUDA platform. In 2d version input `X` and output `Y` are matrices. Output matrix `Y` is defined as `Y(i, j)` = sum of elements of `X` with index `(k, l)` s.t. `i - D < k < i + D` and `j - D < l < j + D`.

## Additional information

1. Cuda documentation: [http://docs.nvidia.com/cuda/index.html#axzz3W3H2DE3N](http://docs.nvidia.com/cuda/index.html#axzz3W3H2DE3N)

2. Introduction to CUDA programming: [http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz3VcSdDZPm](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz3VcSdDZPm)
