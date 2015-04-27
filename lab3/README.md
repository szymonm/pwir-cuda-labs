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
  for all v in V do in parallel
    if (d(v) = 0) then add v to I and delete v from V
    else mark v with probability 1/2 * d(v)
  for all (u, v) in E do in parallel
    if both u and v are marked
      then unmark the lower degree vertex
  for all v in V do in parallel
    if v is marked then add v to S
  I ← I ∪ S
  delete S"'"=S ∪ neigh(S) from V, and all edges incident to S"'" from E
  if (V = ∅) return I
```

### Exercises

1. Implement the parallel algorithm for finding *MIS* on CUDA platform. Measure the speed-up over the sequential version (greedy algorithm).

2. Plot a diagram showing running time of your implementation versus graph size. What is the observed complexity of the implementation?

3. Can you verify graph size limit that your CUDA implementation is possible to handle efficiently?
