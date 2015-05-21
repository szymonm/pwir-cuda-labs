//
// Created by Szymon Matejczyk on 21.05.15.
//

#ifndef MPI_ASSIGNMENT_SEQ_GRAPH_H
#define MPI_ASSIGNMENT_SEQ_GRAPH_H

typedef struct {
    int** edges;
    int* outDegrees;
    int nodes;
    int maxNodeWithOutEdgesId;
} Graph;

Graph* readGraph(FILE* f);

void printGraph(Graph* g);

void freeGraph(Graph* graph);

Graph* reverseGraph(Graph* graph);

int dfs(int node, int nextId, int parentNode, Graph* graph, Graph* reversed,
        int* numbering, int* parent);

#endif //MPI_ASSIGNMENT_SEQ_GRAPH_H
