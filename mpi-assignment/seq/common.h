//
// Created by Szymon Matejczyk on 21.05.15.
//

#ifndef MPI_ASSIGNMENT_SEQ_COMMON_H
#define MPI_ASSIGNMENT_SEQ_COMMON_H

#ifdef DEBUG
#define DEBUG_TEST 1
#else
#define DEBUG_TEST 0
#endif

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define debug_print(...) \
            do { if (DEBUG_TEST) fprintf(stderr, ##__VA_ARGS__); } while (0)

void error(const char* errorMsgFormat, ...);

#endif //MPI_ASSIGNMENT_SEQ_COMMON_H
