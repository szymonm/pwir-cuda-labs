//
// Created by Szymon Matejczyk on 21.05.15.
//
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

void error(const char* errorMsgFormat, ...) {
    va_list argptr;
    va_start(argptr, errorMsgFormat);
    vfprintf(stderr, errorMsgFormat, argptr);
    fflush(stderr);
    va_end(argptr);
    exit(-1);
}

