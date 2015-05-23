# MPI - Graph pattern matching
## Assignment - PWIR 2015

## Introduction
Given directed graph `G=(V, E)` and directed, connected graph `P=(Q, S)` called pattern, *graph pattern matching* is a problem of finding all induced subgraphs of `G` isomorphic with pattern `P` called matches. Your task is to implement a distributed algorithm that solves the graph pattern matching problem using MPI system.

You should evaluate performance of your solution and present results of the evaluation in a report.

## Specification

### Formal definitions
Graph `G' = (V', E')` is a induced subgraph of `G = (V, E)` if `V'` is a subset of `V` and `E'` is a subset of `E` containing all edges between nodes in `V'`.

We say that an induced subgraph `G'` of `G` matches pattern `P` if there exists a bijective function `h` from the nodes of `P` to the nodes of `G'` such that if `(u, v)` is an edge in `P` then `(h(u), h(v))` is an edge in `G'`.

### Input
Your program should accept 2 arguments from the command line. First is the path to the file containing graphs `G` and `P`, second is the path to the output file.

Your program should read 2 graphs (`G` and `P`) from the input file separated with a blank line. First line of graph encoding contains a single integer `N`. Nodes are numbered from `1` to `N` (hence `N` is also maximal node id). Next lines (until empty line) contain information about graph edges grouped by the source node. For a node, first line of the node's outgoing edges encoding consists of two integers: node id `n` and number `k` of outgoing edges. Each of the following `k` lines contains a single integer - the target of an edge from node with id `n`. Note that, if a node has no outgoing edges it is not stored on the list. Hence, set of vertices of a graph is set sum of nodes with outgoing edges and nodes that are a target of some edge.

You can assume that `P` is weakly connected and has at most `10` nodes and that input encoding is correct. You can also assume that `G` has no more than `10 000 000` vertices and its diameter is smaller than `100`.

There are no self loops in `P` and `G`.

Example input:
```
5
1 3
2
3
4
2 2
1
3
3 2
1
4
4 1
5

3
1 1
2
2 1
3
3 1
1
```
The input encodes `G = ({1, 2, 3, 4, 5}, {(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (3, 1), (3, 4), (4, 5)})` and `P = ({7, 8, 9), {(7, 8), (8, 9), (9, 7)})`.

### Output
Your program should print to the output file all induced subgraphs of G isomorphic with P in the following way: each line encode a single induced subgraph and is a list of node ids (separated with space) of the induced subgraph that map to consecutive nodes of the pattern graph (any ordering of induced subgraphs is allowed).

For the input from the previous example, correct output is:
```
1 2 3
2 3 1
3 1 2
```

Your solution should be concise(after doing unix's `sort` on output file) with reference pairs of input and output (see: [here](https://github.com/szymonm/pwir-cuda-labs/tree/master/mpi-assignment/seq)), although, the ordering of matches may be different.

## Solution format
Your solution should compile and run on `notos` cluster. Every process should not use more than 512MB of RAM.

Student should provide archive named with her user id (ex. `ab123456.tgz`), which, when unpacked, should create directory named with the user id, that contains following files:

1. `report.pdf` - your report in the PDF format.
2. `Makefile` - make file compiling your solution to `gpm-par.exe`. You are not allowed to change compilation configuration significantly (ex. flags). You can make changes to allow C++ code.
3. `gpm-seq-naive.c` - original file with sequential solution (see [here](https://github.com/szymonm/pwir-cuda-labs/tree/master/mpi-assignment/seq)) 
4. `gpm-par.c` - your parallel implementation using MPI.

## Report
Report should contain at least:

1. Detailed description of your solution.
2. Tests and their results presented in a readable form.

### Description
Description part should contain general description of your idea, detailed specification of the algorithm including assumptions, used data structures, task allocation to processes, communication, optimalizations etc. Description should be clear and easy to understand. Consider enhancing it with diagrams and pseudo-code listings.

### Tests
You should perform both correctness and performance testing.

Make sure to include following details:

1. Description of test environment.
2. Times of program execution for the two reference examples.

You should measure time of graph distribution and pattern matching separately, but optimize for pattern matching efficiency. Your program should print distribution and computations times to the standard output (see sequential version for reference).

You cannot assume that whole graph fits RAM memory of a single node. We will definitelly check your solutions on graphs that do not fit into space constraints. We will test your solution on high number of nodes and processors (ex. 128x4).

## Grading

1. *6 points* - correctness of your solution (will be checked automatically, so make sure your solution is 100% complaint with specification);
2. *4 points* - performance of your solution. You will receive 4 points, when your solution is correct and runs at least as fast as our parallel benchmark solution - expected running times on 2 big graphs will be published later;
3. *2 points* - report;
4. *3 points* - You will receive 3 points, when your solution is among top 10% of all students that send correct solution; 2 points, when your solution is among top 20% and 1 point, when you are in top 30%. We will test your results on real-life, scale-free graphs from [SNAP library](https://snap.stanford.edu/data/).

## Literature

Following articles may help you solve the problem.

1. Ma S. et. al. *Distributed Graph Pattern Matching.*
2. Fard A. et. al. *Distributed and scalable graph pattern matching: models and algorithms.*
3. Fan W. *Graph Pattern Matching Revised for Social Network Analysis*.

## Assumptions

1. You cannot assume that the whole graph fits into one nodes memory.
2. You can assume that the number of patterns found is lower than `1 000 000`.

## FAQ
Please, send additional questions to: `sm262956@mimuw.edu.pl`.

**Czy krawędzie mogą się powtarzać?**

Nie.

**Czy można założyć, że każdy proces może trzymać listę krawędzi wychodzących kolejnych |V|/(liczba_procesów) wierzchołków?**

Nie można.

**Jaki będzie w testach maksymalny stosunek (|V| + |E|)/(liczba_procesów)? Może to mieć znaczenie, jeśli do tego, aby zmieścić się w pamięci ram 512MB trzeba będzie równomiernie ze względu na stopień wierzchołków rozdystrybuować dane.**

Można założyć, że `(|V| + |E|)/(liczba_procesów) * sizeof(int) < 64 MB`.

**Jakie testy zostaną udostępnione?**

Zostaną udostępnione małe testy poprownościowe (podstawowa weryfikacja, czy Państwa rozwiązanie dobrze interpretuje wejście i wypisuje wyjście) oraz wymagania dotyczące czasu działania na 2 większych grafach (punkt 2.).

Podczas sprawdzania zweryfikujemy Państwa rozwiązania na kolejnych testach poprawnościowych (punkt 1. oceniania) oraz zrobimy konkurs na grafach, które ujawnimy dopiero po uzyskaniu wszystkich rozwiązań.

**Czy można założyć, że program będzie uruchamiany z przynajmniej 4 procesami?**

Tak

**2. Jeśli chodzi o czas dystrybucji oraz czas wykonania - jak dokładnie zdefiniować, gdzie kończy się jedno, a zaczyna drugie? Przykładowo - przyjmijmy, że chciałbym transponować graf G, i żeby każdy proces miał przypisany pewien podzbiór wierzchołków, dla któego trzymałby wszystkie krawędzie wychodzące i wchodzące do tych wierzchołków. Czy rozesłanie tych informacji między procesami można zaliczyć do dystrybucji? Jak efektywne musi być to rozesłanie?**

Tak, rozproszoną transpozycję grafu można uznać za dystrybucję. Będziemy oceniać czas łączny i czas obliczeń (ważniejszy będzie czas obliczeń). Powinno być rozsądne, ale nie oczekujemy wysublimowanego rozwiązania dystrybucji.

