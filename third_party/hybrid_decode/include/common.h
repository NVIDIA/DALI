#ifndef COMMON_H_
#define COMMON_H_

#include "nvToolsExt.h"

#include <array>
#include <iostream>
#include <string>
#include <vector>

using std::array;
using std::cout;
using std::endl;
using std::string;
using std::vector;

#ifdef ENABLE_TIMERANGES
// Just for debug in nvvp. No-op if not defined. To enable:
// Add `add_definitions(-DENABLE_TIMERANGES)` to CMakeLists.txt,
// `export LIBRARY_PATH=/usr/local/cuda/lib64` (or something like that)
// and `export CFLAGS=-lnvToolsExt` before calling cmake.
#include "nvToolsExt.h"
#endif

struct TimeRange 
{
    TimeRange(char const * name) 
        {
#ifdef ENABLE_TIMERANGES
            nvtxRangePushA(name);
#endif
        }
    ~TimeRange() 
        {
#ifdef ENABLE_TIMERANGES
            nvtxRangePop();
#endif
        }
};

inline int DivUp(int x, int d) {
    return (x + d - 1) / d;
}

#endif // COMMON_H_
