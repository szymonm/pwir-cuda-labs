//
// Created by Szymon Matejczyk on 21.05.15.
//

#ifndef MPI_ASSIGNMENT_SEQ_GRAPH_H
#define MPI_ASSIGNMENT_SEQ_GRAPH_H

/**
 * Directed graph data structure.
 */
typedef struct {
    /**
     * For each node an array of outgoing edges.
     */
    int** edges;

    /**
     * Out-degrees for nodes or -1 if the node is not in the graph.
     */
    int* outDegrees;

    /**
     * Number of nodes.
     */
    int nodes;

    /**
     * Maximal id of node that has outgoing edges.
     */
    int maxNodeWithOutEdgesId;
} Graph;

Graph* readGraph(FILE* f);

void printGraph(Graph* g);

void freeGraph(Graph* graph);

Graph* reverseGraph(Graph* graph);

int dfs(int node, int nextId, int parentNode, Graph* graph, Graph* reversed,
        int* numbering, int* parent);

#endif //MPI_ASSIGNMENT_SEQ_GRAPH_H
