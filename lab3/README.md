# Lab 3 - tunning performance cont'd and MIS

In this lab we cover techniques of improving performance of GPU computations that have not yet been presented. We conclude the CUDA part with implementing PRAM algorithm for classical Maximal Independent Set problem.

## Additional optimization techniques for CUDA GPU

### Memory access techniques (again)

We have already seen that we can improve program speed by avoiding accesses to global memory and using block shared memory instead. While this is usually the most effective, there are also others that may speed up compuatations significantly.

**Memory coalescing (DRAM)**

The data from global memory is read in read in 32-, 64-, or 128-byte transactions. When a warp accesses the global memory it coalesces the accesses to adjacent addresses into one or more of these transactions. Thus, it is generally faster to read data from adjacent memory slots in the single instruction. Consider an example of 2d array, that is read in a single instruction by threads of a single warp from the same column. It will be faster if the matrix is stored by columns as the number of transactions will be lower.

**Bank conflicts (shared memory)**

Shared memory has 32 banks that is organized such that successive 32-bit words map to successive banks. An access to a single bank results in a bank conflict between two threads of the same warp if the threads access different words of the bank in a single instruction. Bank conflicts cause delays in the memory operations. To avoid them, threads of the same warp should in a single instruction should read either data from different banks or same word from the same bank. 

In a 2d arrays case, this can be usually ensured by slightly changing the array dimensions. Consider two 2d arrays `a[1024][32]` and `b[1024][33]` in shared memory and an instruction of threads in a warp that reads the column of each array.

```cuda
__shared__ float a[1024][32];
__shared__ float b[1024][33];
```

Observe that due to the number of banks the instruction on array `a` will result in a bank conflict that will not occur for `b`. Thus, when encountering a bank conflict, programmer can add an never used column to the array to speed up the program.

### Atomic variables

CUDA C language extension supports atomic operations on basic types. Examples include: `atomicAdd`, `atomicMax`, `atomicDec`, `atomicOr` and others. See Cuda C programming guide for details.

### Vector types

Vector types are derived from basic integer and floating types. Theirs 1st, 2nd, 3rd and 4th elements are accessed using `x`, `y`, `z`, `w`, respectively.
Constructors of the form `make_<<type>><<N>>` are use to construct a vector variable of `type` and arity `N`. Use of vector types is presented below:
```cuda
int2 point = make_int2(3, 2);
double norm = sqrt ((point.x * point.x) + (point.y * point.y))
```

### Faster mathematical functions 

CUDA supports supports most of the C/C++ standard library mathematical functions (like `sin`, `cons`, `sqrt` ...). Some of them have less precise but faster varsions that can be used in the device code, i.e.: `__fdividef` (division), `__sinf`, `__logf`, `__powf`. See [here](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions) for a complete list.

### Random numbers

Using CUDA random number generator is a bit complicated, because you should avaiod initializing generator in every thread from seed and you don't want to use the same generator across threads (because you would obtain the same *random* numbers). Hence, you usually initialize multiple random number generator states (in a kernel called in a setup phase) and pass it to your kernel, so that every thread can use different state. See the code below.
```cuda
__global__ void setup_kernel (curandState * state, unsigned long seed)
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
} 

__global__ void generate( curandState* globalState ) 
{
    curandState localState = globalState[threadIdx.x];
    float random = curand_uniform( &localState );
    printf("random: %f\n", random);
    globalState[threadIdx.x] = localState; 
}

int main( int argc, char** argv) 
{
    curandState* devStates;
    cudaMalloc(&devStates, N*sizeof(curandState));
    
    // setup seeds
    setup_kernel <<< 1, N >>> (devStates, time(NULL));

    // generate random numbers
    generate <<< 1, N >>> (devStates);

    return 0;
}
```
### Debugging

In new CUDA versions `printf` function works as expected. Try in kernel code:
``` cuda
 printf("Block(%d, %d, %d), Thread(%d, %d, %d) Hello!", blockIdx.x, 
   blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
```
## Maximal Independent Set

**NC** is a complexity class of problems that can be solved in polylogarithmic time on a PRAM computer with polynomial number of processors. *NC* problems can be thought of as the problems that can be effectively solved on a parallel computer. Obviously, *NC* is in *P*, because a polylogarithmic parallel program can be simulated by a sequential computer in polynomial time. The important, open question is whether *NC* = *P*. In other words, does there exists a problem that can be solved in polynomial time on a sequential machine, but can not be speeded up to polylogarithmic time on a parallel computer. Such a problem would be fundamentally sequential. An example of a problem that is believed to be fundamentally sequential is *Lexicographically First Maximal Independent Set* (LFMIS).

Given undirected graph `G = (V, E)` **Independent Set** of `G` is a set `I` of vertices `V` such that no adjacent vertices are in `I`. **Maximal Independent Set (MIS)** is a maximal *Independent Set* under superset relation. *Independent Set* of the largest cardinality is called *Maximum Independent Set*.

There exists a simple algorithm to find a *MIS* in the graph:
```python
I ← ∅
for v = 1 to n do
  if (v has no neighbors in I) then add v to I
return I
```

The *MIS* returned by the trivial algorithm is called *lexicographically first* MIS. As mentioned before, it appears that solving LFMIS is impossible to do effectively in parallel. However, there exists a simple randomized algorithm for finding any *MIS*.

```python
I ← ∅
while True:
  S ← ∅
  for all v in V do in parallel (I)
    if (d(v) = 0) then add v to I and delete v from V
    else mark v with probability 1/ (2 * d(v))
  for all (u, v) in E do in parallel  (II)
    if both u and v are marked
      then unmark the lower degree vertex
  for all v in V do in parallel
    if v is marked then add v to S
  I ← I ∪ S
  delete S* = S ∪ neigh(S) from V, and all edges incident to S* from E
  if (V = ∅) return I
```

### Exercises

1. Implement the parallel algorithm for finding *MIS* on CUDA platform. Check the minimal graph size for which CUDA implementation is faster than LFMIS. You may simulate random number generator with some constants.

**Hints**

1. There is no other way of synchronization between blocks than launching seperate kernels sequentially. You may consider using kernels that: compute vertex degree (in a graph with some nodes removed), mark nodes with some probability (I), resolve conflicts (II), update graph, check if set is maximal.

2. Start with small graphs of known structure (see `cycle` and `star` functions in the program skeleton). Verification is much easier on this graphs (it's easy to compute degrees and check if a set is independent). You may verify your partial results with files in `results` directory.

3. Use function `fillPseudoRandoms` instead of a proper random generator to make debugging easier. You may need to assume some additional pseudo-randomness using threadIdx or iteration number (algorithm may fail if some node is never marked).
