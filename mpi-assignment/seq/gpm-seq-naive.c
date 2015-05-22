//
// Created by Szymon Matejczyk on 17.05.15.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "graph.h"

#define MAX_MATCH_SIZE 10

typedef struct {
    int matchedNodes;

    /**
     * `matches[i] contains data graph node id of `i` node in pattern or -1 if the
     * node is not yet matched.
     */
    int matches[MAX_MATCH_SIZE + 1];
} Match;


void printMatch(Match* match, FILE* out) {
    fprintf(out, "%d", match->matches[1]);
    for (int i = 2; i <= match->matchedNodes; i++) {
       fprintf(out, " %d", match->matches[i]);
    }
    fprintf(out, "\n");
}

int graphToPatternNode(int node, Match* match) {
    int res = -1;
    for (int i = 0; i < match->matchedNodes; i++) {
        if (match->matches[i] == node) {
            res = i + 1;
        }
    }
    return res;
}

int patternToGraphNode(int node, Match* match) {
    return match->matches[node];
}

int containsEdge(int from, int to, Graph* graph) {
    for (int i = 0; i < graph->outDegrees[from]; i++) {
        if (graph->edges[from][i] == to) {
            return 1;
        }
    }
    return 0;
}

int matchContains(int nodeId, Match* match) {
    for (int i = 1; i <= MAX_MATCH_SIZE; i++) {
       if (match->matches[i] == nodeId) {
           return 1;
       }
    }
    return 0;
}

int checkNodeMatches(int nodeData, int nodePattern, Graph* dataGraph, Graph* pattern,
                     Graph* patternReversed, Match* match) {
    if (matchContains(nodeData, match)) {
        return 0;
    }

    // check out edges - every edge from the pattern must be in the graph
    for (int i = 0; i < pattern->outDegrees[nodePattern]; i++) {
        int targetPatternNode = pattern->edges[nodePattern][i];
        int target = patternToGraphNode(targetPatternNode, match);
        if (target > -1 && !containsEdge(nodeData, target, dataGraph)) {
            return 0;
        }
    }

    // check in edges
    for (int i = 0; i < patternReversed->outDegrees[nodePattern]; i++) {
        int sourcePatternNode = patternReversed->edges[nodePattern][i];
        int sourceData = patternToGraphNode(sourcePatternNode, match);
        if (sourceData > -1 && !containsEdge(sourceData, nodeData, dataGraph)) {
            return 0;
        }
    }

    return 1;
}

void ordering(int* numbering, int* order, int len) {
    for (int i = 1; i <= len; i++) {
        if (numbering[i] != -1) {
            order[numbering[i]] = i;
        }
    }
}

void printArray(int* arr, int len) {
    debug_print("[%d", arr[0]);
    for (int i = 1; i < len; i++) {
        debug_print("\t%d", arr[i]);
    }
    debug_print("]\n");
}

Match addNode(int node, int patternNode, Match* match) {
    Match m;
    int i;
    for (i = 0; i <= MAX_MATCH_SIZE; i++) {
        m.matches[i] = match->matches[i];
    }
    m.matches[patternNode] = node;
    m.matchedNodes = match->matchedNodes + 1;
    return m;
}

void exploreMatch(Graph* dataGraph, Graph* pattern, Graph* patternReversed, Match match,
                  int* nodesMatchingOrder, int* parents, FILE* out) {
    if (match.matchedNodes == pattern->nodes) {
        printMatch(&match, out);
        return;
    }

    int nextNodePatternId = nodesMatchingOrder[match.matchedNodes + 1];
    int nextNodeParentPatternId = parents[nextNodePatternId];

    int parentId = patternToGraphNode(nextNodeParentPatternId, &match);
    // for neighbors of parent we try to match the new node
    for (int i = 0; i < dataGraph->outDegrees[parentId]; i++) {
        int node = dataGraph->edges[parentId][i];
        if (checkNodeMatches(node, nextNodePatternId, dataGraph, pattern, patternReversed, &match)) {
            Match new = addNode(node, nextNodePatternId, &match);
            exploreMatch(dataGraph, pattern, patternReversed, new, nodesMatchingOrder, parents, out);
        }
    }
}

int  main(int argc, char** argv)
{
    if (argc != 3) {
        error("Wrong number of arguments.");
    }

    time_t startTime = time(NULL);
    char* filename = argv[1];
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        error("Input file not found: %s\n", filename);
    }

    FILE* out = fopen(argv[2], "w");
    if (out == NULL) {
        error("Can't open output file.\n");
    }

    Graph* dataGraph = readGraph(f);

    Graph* pattern = readGraph(f);
    fclose(f);

    Graph* patternReversed = reverseGraph(pattern);

    if (DEBUG_TEST)
        printGraph(dataGraph);

    if (DEBUG_TEST)
        printGraph(pattern);

    if (DEBUG_TEST)
        printGraph(patternReversed);

    int* dfsPatternNumbering = malloc((pattern->nodes + 1) * sizeof(int));
    memset(dfsPatternNumbering, -1, (pattern->nodes + 1) * sizeof(int));

    int* dfsPatternParents = malloc((pattern->nodes + 1) * sizeof(int));
    dfs(1, 1, -1, pattern, patternReversed, dfsPatternNumbering, dfsPatternParents);

    debug_print("Numbering: ");
    printArray(dfsPatternNumbering, pattern->nodes);
    debug_print("Parent: ");
    printArray(dfsPatternParents, pattern->nodes);

    int* patternNodesOrdered = malloc(pattern->nodes * sizeof(int));

    ordering(dfsPatternNumbering, patternNodesOrdered, pattern->nodes);
    debug_print("Ordering: ");
    printArray(patternNodesOrdered, pattern->nodes);

    time_t distributionTime = time(NULL);
    printf("Distribution time[s]: %ld\n", (distributionTime - startTime));

    for (int i = 1; i <= dataGraph->nodes; i++) {
        if (dataGraph->outDegrees[i] > -1) {
            Match m;
            memset(m.matches, -1, (MAX_MATCH_SIZE + 1) * sizeof(int));
            m.matchedNodes = 1;
            m.matches[1] = i;
            exploreMatch(dataGraph, pattern, patternReversed, m, dfsPatternNumbering,
                dfsPatternParents, out);
        }
    }

    time_t computationsTime = time(NULL);
    printf("Computations time[s]: %ld\n", (computationsTime - distributionTime));

    freeGraph(dataGraph);
    freeGraph(pattern);
    freeGraph(patternReversed);
    free(dfsPatternNumbering);
    free(dfsPatternParents);
    free(patternNodesOrdered);

    fclose(out);

    return 0;
}
