//
// Created by Szymon Matejczyk on 21.05.15.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h> /* for isspace */

#include "graph.h"
#include "common.h"


int isLineEmpty(const char *line)
{
    /* check if the string consists only of spaces. */
    while (*line != '\0')
    {
        if (isspace(*line) == 0)
            return 0;
        line++;
    }
    return 1;
}

Graph* readGraph(FILE* f) {
    int from, to, edgesNo, edgeCounter, maxNodeId;
    char line[64];

    Graph* g = malloc(sizeof(Graph));
    if (fscanf(f, "%d\n", &maxNodeId) != 1) {
        error("Can't parse number of nodes.");
    }
    g->nodes = maxNodeId;
    g->edges = malloc((maxNodeId + 1) * sizeof(int*));
    g->outDegrees = malloc((maxNodeId + 1) * sizeof(int));
    for (int i = 0; i <= maxNodeId; i++)
        g->outDegrees[i] = -1;
    int* edgesForNode;
    g->maxNodeWithOutEdgesId = 0;
    while ((fgets(line, sizeof line, f) != NULL) && (isLineEmpty(line) == 0)) {
        if (sscanf(line, "%d %d", &from, &edgesNo) == 2) {
            edgeCounter = 0;
            g->maxNodeWithOutEdgesId = max(g->maxNodeWithOutEdgesId, from);
            g->outDegrees[from] = edgesNo;
            edgesForNode = malloc(edgesNo * sizeof(int));
            while (edgeCounter < edgesNo) {
                if (fgets(line, sizeof line, f) != NULL) {
                    if (sscanf(line, "%d", &to) == 1) {
                        debug_print("Edge %d %d\n", from, to);
                        edgesForNode[edgeCounter] = to;
                        if (g->outDegrees[to] == -1)
                            g->outDegrees[to] = 0;
                        fflush(stdout);
                        edgeCounter++;
                    } else {
                        error("Can't read edge from %d", from);
                    }
                } else {
                    error("Can't read edge from %d", from);
                }
            }
            g->edges[from] = edgesForNode;
            debug_print("node read\n");
        } else {
            error("Can't read node\n");
        }
    }
    return g;
}

void printGraph(Graph* g) {
    printf("Graph: %d nodes, maxNodeWithOutEdgesId: %d\n", g->nodes, g->maxNodeWithOutEdgesId);
    fflush(stdout);
    for (int i = 0; i <= g->maxNodeWithOutEdgesId; i++) {
        if (g->outDegrees[i] == 0) {
            printf("Node %d with no out neighbors\n", i);
        }
        if (g->outDegrees[i] > 0) {
            printf("Node %d, out neighbors [%d]:", i, g->outDegrees[i]);
            fflush(stdout);
            for (int j = 0; j < g->outDegrees[i]; j++) {
                printf("%d ", g->edges[i][j]);
            }
            printf("\n");
            fflush(stdout);
        }
    }
}

void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->maxNodeWithOutEdgesId; i++) {
        if (graph->outDegrees[i] > 0) {
            free(graph->edges[i]);
        }
    }
    free(graph->edges);
    free(graph->outDegrees);
    free(graph);
}

Graph* reverseGraph(Graph* graph) {
    Graph* reversed = malloc(sizeof(Graph));
    reversed->nodes = graph->nodes;
    int* inDegrees = malloc((graph->nodes + 1) * sizeof(int));
    for (int i = 0; i <= graph->nodes; i++) {
        inDegrees[i] = -1;
    }
    for (int i = 0; i <= graph->maxNodeWithOutEdgesId; i++) {
        if (graph->outDegrees[i] > -1 && inDegrees[i] == -1)
            inDegrees[i] = 0;
        for (int j = 0; j < graph->outDegrees[i]; j++) {
            int target = graph->edges[i][j];
            if (inDegrees[target] == -1)
                inDegrees[target] = 0;
            inDegrees[target]++;
        }
    }
    reversed->outDegrees = inDegrees;
    int maxNodeWithInEdges = 0;
    for (int i = 0; i <= graph->nodes; i++) {
        if (inDegrees[i] > 0)
            maxNodeWithInEdges = i;
    }
    reversed->maxNodeWithOutEdgesId = maxNodeWithInEdges;
    int** edges = malloc((maxNodeWithInEdges + 1) * sizeof(int*));
    for (int i = 0; i <= maxNodeWithInEdges; i++) {
        edges[i] = malloc(inDegrees[i] * sizeof(int));
    }
    int* counters = malloc((maxNodeWithInEdges + 1) * sizeof(int));
    memset(counters, 0, (maxNodeWithInEdges + 1) * sizeof(int));
    for (int i = 0; i <= graph->maxNodeWithOutEdgesId; i++) {
        for (int j = 0; j < graph->outDegrees[i]; j++) {
            int target = graph->edges[i][j];
            edges[target][counters[target]++] = i;
        }
    }
    free(counters);
    reversed->edges = edges;
    return reversed;
}

int dfs(int node, int nextId, int parentNode, Graph* graph, Graph* reversed,
        int* numbering, int* parent, int viaReverseEdge) {
    int currentId = nextId;
    if (viaReverseEdge) {
        numbering[node] = -currentId;
    } else {
        numbering[node] = currentId;
    }
    parent[node] = parentNode;
    for (int i = 0; i < graph->outDegrees[node]; i++) {
        if (numbering[graph->edges[node][i]] == 0) {
            currentId = dfs(graph->edges[node][i], currentId + 1, node, graph,
                            reversed, numbering, parent, 0);
        }
    }
    for (int i = 0; i < reversed->outDegrees[node]; i++) {
        if (numbering[reversed->edges[node][i]] == 0) {
            currentId = dfs(reversed->edges[node][i], currentId + 1, node, graph,
                            reversed, numbering, parent, 1);
        }
    }
    return currentId;
}

